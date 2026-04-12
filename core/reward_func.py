from __future__ import annotations

"""Reward V1: MRR@k + Recall@k + CopyPenalty + FormatPenalty."""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from app_config import RewardConfig, patch_pyserini_prebuilt_index_urls

TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
MARKER_LINE_RE = re.compile(r"^(?:rewritten\s+query|search\s+query)\s*:\s*(.*)$", flags=re.IGNORECASE)
MARKER_INLINE_RE = re.compile(r"(?:rewritten\s+query|search\s+query)\s*:\s*([^\n\r]+)", flags=re.IGNORECASE)
EXPLANATION_RE = re.compile(
    r"\b("
    r"because|therefore|explanation|reasoning|step[- ]?by[- ]?step|"
    r"user query|search query|rewritten query|i (?:think|believe)|let(?:'s| us)"
    r")\b",
    flags=re.IGNORECASE,
)
TRAILING_PARTIAL_TOKENS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    total: float
    mrr: float
    recall: float
    overlap: float
    copy_penalty: float
    format_penalty: float
    hit_rank: int | None
    retrieved_relevant_count: int
    relevant_total: int
    rewritten_query: str


def compute_mrr_at_k(result_docids: Sequence[str], relevant_docids: Iterable[str], topk: int = 10) -> tuple[float, int | None]:
    relevant = set(relevant_docids)
    if not relevant:
        return 0.0, None

    for rank, docid in enumerate(result_docids[:topk], start=1):
        if docid in relevant:
            return 1.0 / rank, rank
    return 0.0, None


def compute_recall_at_k(
    result_docids: Sequence[str],
    relevant_docids: Iterable[str],
    topk: int = 50,
) -> tuple[float, int, int]:
    relevant = set(relevant_docids)
    relevant_total = len(relevant)
    if relevant_total == 0:
        return 0.0, 0, 0

    retrieved_relevant_count = len(set(result_docids[:topk]) & relevant)
    return retrieved_relevant_count / float(relevant_total), retrieved_relevant_count, relevant_total


def _unreadable_ratio(text: str) -> float:
    if not text:
        return 1.0

    unreadable = 0
    for ch in text:
        if ch.isspace():
            continue
        if ch.isalnum() or ch in "-_.,:;!?()/[]'\"":
            continue
        unreadable += 1
    return unreadable / max(1, len(text))


def compute_unreadable_ratio(text: str) -> float:
    return _unreadable_ratio(text)


def _tokenize_for_overlap(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def compute_lexical_overlap(source_query: str, rewritten_query: str) -> float:
    src = set(_tokenize_for_overlap(source_query))
    rew = set(_tokenize_for_overlap(rewritten_query))
    if not src or not rew:
        return 0.0
    union = src | rew
    if not union:
        return 0.0
    return len(src & rew) / float(len(union))


def compute_copy_penalty(overlap: float, tau: float) -> float:
    return max(0.0, float(overlap) - float(tau))


def _english_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    english = sum(1 for ch in letters if "a" <= ch.lower() <= "z")
    return english / float(len(letters))


def _looks_like_explanation(text: str) -> bool:
    if EXPLANATION_RE.search(text):
        return True

    if ":" in text:
        prefix = text.split(":", 1)[0].strip().lower()
        if prefix in {"query", "search query", "rewritten query", "explanation", "reasoning"}:
            return True
    return False


def compute_format_penalty(text: str, cfg: RewardConfig) -> float:
    cleaned = (text or "").strip()
    if not cleaned:
        return 1.0
    if "\n" in cleaned or "\r" in cleaned:
        return 1.0
    if _looks_like_explanation(cleaned):
        return 1.0
    if len(_tokenize_for_overlap(cleaned)) > cfg.format_max_tokens:
        return 1.0
    if _english_ratio(cleaned) < cfg.format_min_english_ratio:
        return 1.0
    if _unreadable_ratio(cleaned) > cfg.format_max_unreadable_ratio:
        return 1.0
    return 0.0


def compose_reward(
    *,
    mrr: float,
    recall: float,
    copy_penalty: float,
    format_penalty: float,
    cfg: RewardConfig,
) -> float:
    return (
        cfg.w_mrr * mrr
        + cfg.w_recall * recall
        - cfg.w_copy * copy_penalty
        - cfg.w_format * format_penalty
    )


def _normalize_candidate_text(text: str) -> str:
    cleaned = (text or "").strip().strip("'").strip('"').strip("`")
    cleaned = re.sub(r"^[\-\*\d\.\)\]\s]+", "", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned


def _extract_query_candidates(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates: list[str] = []

    for idx, line in enumerate(lines):
        marker_match = MARKER_LINE_RE.match(line)
        if marker_match:
            tail = marker_match.group(1).strip()
            if tail:
                candidates.append(tail)
            elif idx + 1 < len(lines):
                candidates.append(lines[idx + 1])
            continue

        inline_match = MARKER_INLINE_RE.search(line)
        if inline_match:
            candidates.append(inline_match.group(1))
            continue

        lowered = line.lower()
        if lowered in {"assistant:", "search query:", "rewritten query:"}:
            continue
        if line.endswith(":") and len(line.split()) <= 4:
            continue
        candidates.append(line)

    for match in MARKER_INLINE_RE.finditer(text):
        candidates.append(match.group(1))

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = _normalize_candidate_text(candidate)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)
    return normalized


def _candidate_rank_key(candidate: str, source_query: str | None, index: int) -> tuple[float, int, int, int, int]:
    tokens = _tokenize_for_overlap(candidate)
    overlap = compute_lexical_overlap(source_query, candidate) if source_query else 0.0
    tail = tokens[-1] if tokens else ""
    completeness = 0 if tail in TRAILING_PARTIAL_TOKENS else 1
    unique_count = len(set(tokens))
    return (overlap, completeness, -index, unique_count, len(tokens))


def clean_rewritten_query(text: str, source_query: str | None = None) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    candidates = _extract_query_candidates(raw)
    if not candidates:
        return _normalize_candidate_text(raw)

    best = max(enumerate(candidates), key=lambda item: _candidate_rank_key(item[1], source_query, item[0]))[1]
    return best


class Rewarder:
    def __init__(
        self,
        qrels: dict[str, set[str]],
        prebuilt_index: str,
        reward_cfg: RewardConfig,
    ) -> None:
        from pyserini.search.lucene import LuceneSearcher

        patch_pyserini_prebuilt_index_urls()
        searcher = self._build_searcher_with_recovery(LuceneSearcher, prebuilt_index)
        if searcher is None:
            raise RuntimeError(f"Failed to initialize prebuilt index: {prebuilt_index}")

        self.searcher = searcher
        self.qrels = qrels
        self.cfg = reward_cfg
        self.mrr_k = max(1, int(reward_cfg.mrr_k))
        self.recall_k = max(1, int(reward_cfg.recall_k))
        self.retrieval_k = max(self.mrr_k, self.recall_k)
        cpu_count = max(1, os.cpu_count() or 1)
        self.search_threads = max(1, min(int(getattr(reward_cfg, "search_threads", 1)), cpu_count))
        if self.search_threads > 1:
            print(f"[reward] retrieval batch_search enabled: threads={self.search_threads}")

    @staticmethod
    def _extract_corrupted_index_path(error_text: str) -> Path | None:
        marker = " does not match expected file size"
        idx = error_text.find(marker)
        if idx <= 0:
            return None
        path_text = error_text[:idx].strip()
        candidate = Path(path_text)
        return candidate if candidate.suffixes else None

    def _build_searcher_with_recovery(self, lucene_searcher_cls, prebuilt_index: str):
        for attempt in range(2):
            try:
                return lucene_searcher_cls.from_prebuilt_index(prebuilt_index)
            except AssertionError as exc:
                error_text = str(exc)
                bad_file = self._extract_corrupted_index_path(error_text)
                can_recover = (
                    "does not match expected file size" in error_text
                    and bad_file is not None
                    and bad_file.exists()
                )
                if can_recover and attempt == 0:
                    try:
                        bad_file.unlink()
                    except OSError:
                        pass
                    continue
                raise

    def _search_docids(self, query: str) -> list[str]:
        if not query:
            return []
        try:
            hits = self.searcher.search(query, k=self.retrieval_k)
            return [str(hit.docid) for hit in hits]
        except Exception:
            return []

    def _search_docids_batch(self, queries: Sequence[str]) -> list[list[str]]:
        results: list[list[str]] = [[] for _ in queries]
        if not queries:
            return results

        non_empty_pairs = [(idx, q) for idx, q in enumerate(queries) if q]
        if not non_empty_pairs:
            return results

        can_batch = hasattr(self.searcher, "batch_search")
        should_batch = can_batch and len(non_empty_pairs) > 1 and self.search_threads > 1
        if should_batch:
            qids = [str(idx) for idx, _ in non_empty_pairs]
            batch_queries = [q for _, q in non_empty_pairs]
            try:
                batch_hits = self.searcher.batch_search(
                    batch_queries,
                    qids,
                    k=self.retrieval_k,
                    threads=self.search_threads,
                )
                for (idx, _), qid in zip(non_empty_pairs, qids):
                    hits = batch_hits.get(qid, [])
                    results[idx] = [str(hit.docid) for hit in hits]
                return results
            except Exception:
                pass

        for idx, query in non_empty_pairs:
            results[idx] = self._search_docids(query)
        return results

    def _score_one(
        self,
        qid: str,
        cleaned_query: str,
        hits_docids: Sequence[str],
        source_query: str | None,
    ) -> RewardBreakdown:
        relevant_docids = self.qrels.get(str(qid), set())
        mrr, hit_rank = compute_mrr_at_k(hits_docids, relevant_docids, topk=self.mrr_k)
        recall, retrieved_relevant_count, relevant_total = compute_recall_at_k(
            hits_docids,
            relevant_docids,
            topk=self.recall_k,
        )
        overlap = compute_lexical_overlap(source_query or "", cleaned_query) if source_query else 0.0
        copy_penalty = compute_copy_penalty(overlap, self.cfg.copy_tau)
        format_penalty = compute_format_penalty(cleaned_query, self.cfg)
        total = compose_reward(
            mrr=mrr,
            recall=recall,
            copy_penalty=copy_penalty,
            format_penalty=format_penalty,
            cfg=self.cfg,
        )
        return RewardBreakdown(
            total=total,
            mrr=mrr,
            recall=recall,
            overlap=overlap,
            copy_penalty=copy_penalty,
            format_penalty=format_penalty,
            hit_rank=hit_rank,
            retrieved_relevant_count=retrieved_relevant_count,
            relevant_total=relevant_total,
            rewritten_query=cleaned_query,
        )

    def score(self, qid: str, rewritten_query: str, source_query: str | None = None) -> RewardBreakdown:
        query = (rewritten_query or "").strip()
        hits_docids = self._search_docids(query)
        return self._score_one(qid, query, hits_docids, source_query)

    def score_batch(
        self,
        qid: str,
        rewritten_queries: Sequence[str],
        source_query: str | None = None,
    ) -> list[RewardBreakdown]:
        raw_queries = [(text or "").strip() for text in rewritten_queries]
        hits_docids_batch = self._search_docids_batch(raw_queries)
        outputs: list[RewardBreakdown] = []
        for query, hits_docids in zip(raw_queries, hits_docids_batch):
            outputs.append(self._score_one(qid, query, hits_docids, source_query))
        return outputs
