from __future__ import annotations

"""Dense reward for BM25 query rewriting."""

import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from app_config import RewardConfig, patch_pyserini_prebuilt_index_urls

TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
MARKER_LINE_RE = re.compile(
    r"^(?:rewritten\s+query|search\s+query|better\s+bm25\s+query)\s*:\s*(.*)$",
    flags=re.IGNORECASE,
)
MARKER_INLINE_RE = re.compile(
    r"(?:rewritten\s+query|search\s+query|better\s+bm25\s+query)\s*:\s*([^\n\r]+)",
    flags=re.IGNORECASE,
)
EXPLANATION_RE = re.compile(
    r"\b("
    r"because|therefore|explanation|reasoning|step[- ]?by[- ]?step|"
    r"user query|search query|rewritten query|i (?:think|believe)|let(?:'s| us)"
    r")\b",
    flags=re.IGNORECASE,
)
USER_QUERY_LINE_RE = re.compile(r"^user\s+query\s*:\s*.*$", flags=re.IGNORECASE)
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
ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?(?:%|[a-z]+)?\b", flags=re.IGNORECASE)
QUESTION_TOKENS = {
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "can",
    "could",
    "would",
    "should",
    "please",
}
NEGATION_TOKENS = {"no", "not", "without", "except", "excluding", "exclude"}
STOPWORD_TOKENS = QUESTION_TOKENS | {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "between",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "those",
    "to",
    "was",
    "were",
    "will",
    "with",
}
POLLUTION_RE = re.compile(
    r"(?:<think|thinking process|analysis:|assistant:|search query:|rewritten query:)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    total: float
    mrr: float
    recall: float
    recall_dense: float = 0.0
    overlap: float = 0.0
    term_preserve: float = 1.0
    keyword_preserve: float = 1.0
    locked_term_preserve: float = 1.0
    number_preserve: float = 1.0
    acronym_preserve: float = 1.0
    negation_preserve: float = 1.0
    length_score: float = 0.0
    clean_format: float = 0.0
    bad_format_penalty: float = 0.0
    unsafe_copy_penalty: float = 0.0
    hit_rank: int | None = None
    retrieved_relevant_count: int = 0
    relevant_total: int = 0
    rewritten_query: str = ""


@dataclass(frozen=True, slots=True)
class StabilizedRewrite:
    raw_query: str
    cleaned_query: str
    final_query: str
    fallback_to_original: bool
    fallback_reasons: tuple[str, ...]
    raw_contains_think: bool
    raw_contains_label: bool
    raw_multiline: bool
    raw_format_penalty: float
    final_overlap: float
    final_term_count: int


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


def compute_exact_copy_penalty(source_query: str, rewritten_query: str, penalty_value: float) -> float:
    source_clean = _normalize_query_text(source_query)
    rewritten_clean = _normalize_query_text(rewritten_query)
    if not source_clean or not rewritten_clean:
        return 0.0
    return float(penalty_value) if source_clean.lower() == rewritten_clean.lower() else 0.0


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
    return compute_bad_format_penalty(text, cfg)


def compute_bad_format_penalty(text: str, cfg: RewardConfig) -> float:
    cleaned = (text or "").strip()
    if not cleaned:
        return min(float(cfg.bad_format_cap), 1.0)

    penalty = 0.0
    if "\n" in cleaned or "\r" in cleaned:
        penalty += 0.5
    if _looks_like_explanation(cleaned) or POLLUTION_RE.search(cleaned):
        penalty += 0.5

    token_count = len(_tokenize_for_overlap(cleaned))
    if token_count > cfg.format_max_tokens:
        penalty += min(0.5, max(0, token_count - cfg.format_max_tokens) * 0.03)

    english_ratio = _english_ratio(cleaned)
    if english_ratio < cfg.format_min_english_ratio:
        gap = cfg.format_min_english_ratio - english_ratio
        penalty += min(0.5, 0.5 * (gap / max(cfg.format_min_english_ratio, 1e-6)))

    unreadable_ratio = _unreadable_ratio(cleaned)
    if unreadable_ratio > cfg.format_max_unreadable_ratio:
        gap = unreadable_ratio - cfg.format_max_unreadable_ratio
        penalty += min(0.5, 0.5 * (gap / max(1.0 - cfg.format_max_unreadable_ratio, 1e-6)))

    return min(float(cfg.bad_format_cap), float(penalty))


def compute_clean_format_score(bad_format_penalty: float) -> float:
    return 1.0 if float(bad_format_penalty) == 0.0 else 0.0


def _preserve_ratio(source_tokens: Sequence[str], rewritten_tokens: Sequence[str]) -> float:
    if not source_tokens:
        return 1.0

    source_counter = Counter(token.lower() for token in source_tokens if token)
    rewritten_counter = Counter(token.lower() for token in rewritten_tokens if token)
    preserved = sum(min(count, rewritten_counter.get(token, 0)) for token, count in source_counter.items())
    total = sum(source_counter.values())
    return preserved / float(total) if total else 1.0


def _extract_keyword_terms(text: str) -> list[str]:
    numeric_tokens = _extract_locked_numeric_tokens(text)
    acronym_tokens = _extract_locked_acronyms(text)
    keywords: list[str] = []
    for token in _tokenize_terms(text):
        if token in STOPWORD_TOKENS or token in NEGATION_TOKENS:
            continue
        if token in numeric_tokens or token in acronym_tokens:
            continue
        if len(token) <= 1:
            continue
        keywords.append(token)
    return keywords


def compute_keyword_preserve(source_query: str, rewritten_query: str) -> float:
    return _preserve_ratio(_extract_keyword_terms(source_query), _tokenize_terms(rewritten_query))


def compute_locked_term_preserve(
    source_query: str,
    rewritten_query: str,
) -> tuple[float, float, float, float]:
    source_terms = _tokenize_terms(source_query)
    rewritten_terms = _tokenize_terms(rewritten_query)
    source_numbers = [match.group(0).lower() for match in NUMERIC_RE.finditer(source_query or "")]
    source_acronyms = [match.group(0).lower() for match in ACRONYM_RE.finditer(source_query or "")]
    source_negations = [token for token in source_terms if token in NEGATION_TOKENS]

    number_preserve = _preserve_ratio(
        source_numbers,
        [match.group(0).lower() for match in NUMERIC_RE.finditer(rewritten_query or "")],
    )
    acronym_preserve = _preserve_ratio(source_acronyms, rewritten_terms)
    negation_preserve = _preserve_ratio(source_negations, rewritten_terms)

    applicable_scores: list[float] = []
    if source_numbers:
        applicable_scores.append(number_preserve)
    if source_acronyms:
        applicable_scores.append(acronym_preserve)
    if source_negations:
        applicable_scores.append(negation_preserve)

    locked_term_preserve = sum(applicable_scores) / float(len(applicable_scores)) if applicable_scores else 1.0
    return locked_term_preserve, number_preserve, acronym_preserve, negation_preserve


def _combine_term_preserve(keyword_preserve: float, locked_term_preserve: float) -> float:
    return 0.5 * float(keyword_preserve) + 0.5 * float(locked_term_preserve)


def compute_term_preserve(
    source_query: str,
    rewritten_query: str,
) -> tuple[float, float, float, float]:
    keyword_preserve = compute_keyword_preserve(source_query, rewritten_query)
    locked_term_preserve, number_preserve, acronym_preserve, negation_preserve = compute_locked_term_preserve(
        source_query,
        rewritten_query,
    )
    term_preserve = _combine_term_preserve(keyword_preserve, locked_term_preserve)
    return term_preserve, number_preserve, acronym_preserve, negation_preserve


def compute_length_score(query: str, cfg: RewardConfig) -> float:
    token_count = len(_tokenize_for_overlap(query))
    min_terms = int(cfg.length_score_min_terms)
    ideal_min = int(cfg.length_score_ideal_min_terms)
    ideal_max = int(cfg.length_score_ideal_max_terms)
    max_terms = int(cfg.length_score_max_terms)

    if token_count <= min_terms or token_count >= max_terms:
        return 0.0
    if ideal_min <= token_count <= ideal_max:
        return 1.0
    if token_count < ideal_min:
        span = max(1, ideal_min - min_terms)
        return max(0.0, min(1.0, (token_count - min_terms) / float(span)))

    span = max(1, max_terms - ideal_max)
    return max(0.0, min(1.0, (max_terms - token_count) / float(span)))


def compute_unsafe_copy_penalty(source_query: str, rewritten_query: str) -> float:
    source_clean = _normalize_query_text(source_query)
    rewritten_clean = _normalize_query_text(rewritten_query)
    if not source_clean or not rewritten_clean:
        return 0.0
    if _source_is_retrieval_ready(source_clean):
        return 0.0
    return 1.0 if source_clean.lower() == rewritten_clean.lower() else 0.0


def compose_reward(
    *,
    mrr: float,
    recall: float,
    recall_dense: float,
    term_preserve: float,
    length_score: float,
    clean_format: float,
    bad_format_penalty: float,
    unsafe_copy_penalty: float,
    cfg: RewardConfig,
) -> float:
    return (
        cfg.w_mrr * mrr
        + cfg.w_recall * recall
        + cfg.w_recall_dense * recall_dense
        + cfg.w_term_preserve * term_preserve
        + cfg.w_length_score * length_score
        + cfg.w_clean_format * clean_format
        - cfg.w_bad_format * bad_format_penalty
        - cfg.w_unsafe_copy * unsafe_copy_penalty
    )


def _normalize_query_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _tokenize_terms(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def _source_is_retrieval_ready(text: str) -> bool:
    normalized = _normalize_query_text(text)
    tokens = _tokenize_terms(normalized)
    if not 2 <= len(tokens) <= 8:
        return False
    if "?" in normalized:
        return False
    return not any(token in QUESTION_TOKENS for token in tokens)


def is_retrieval_ready_query(text: str) -> bool:
    """Return True when the query already looks like a compact BM25-style keyword query."""

    return _source_is_retrieval_ready(text)


def _extract_locked_numeric_tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in NUMERIC_RE.finditer(text or "")}


def _extract_locked_acronyms(text: str) -> set[str]:
    return {match.group(0).lower() for match in ACRONYM_RE.finditer(text or "")}


def _dedupe_keep_order(items: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def stabilize_generated_rewrite(
    raw_query: str,
    *,
    source_query: str,
    guardrail_cfg: object,
    reward_cfg: RewardConfig,
) -> StabilizedRewrite:
    """Clean model output and fall back to the source query when the rewrite is unsafe."""

    source_clean = _normalize_query_text(source_query)
    raw_clean = (raw_query or "").strip()

    stop_marker = getattr(guardrail_cfg, "stop_on", None)
    if stop_marker and stop_marker != "\n":
        stop_idx = raw_clean.find(stop_marker)
        if stop_idx >= 0:
            raw_clean = raw_clean[:stop_idx].strip()

    cleaned = clean_rewritten_query(raw_clean, source_query=source_clean)
    cleaned = _normalize_query_text(cleaned)
    fallback_reasons = ["empty_after_clean"] if not cleaned else []
    final_query = source_clean if fallback_reasons else cleaned
    final_query = _normalize_query_text(final_query or source_clean)
    final_terms = _tokenize_terms(final_query)

    return StabilizedRewrite(
        raw_query=raw_clean,
        cleaned_query=cleaned,
        final_query=final_query,
        fallback_to_original=bool(fallback_reasons),
        fallback_reasons=tuple(fallback_reasons),
        raw_contains_think=("<think" in raw_clean.lower()) or ("thinking process" in raw_clean.lower()),
        raw_contains_label=bool(POLLUTION_RE.search(raw_clean)),
        raw_multiline=("\n" in raw_clean) or ("\r" in raw_clean),
        raw_format_penalty=float(compute_format_penalty(raw_clean, reward_cfg)),
        final_overlap=float(compute_lexical_overlap(source_clean, final_query)) if source_clean else 0.0,
        final_term_count=len(final_terms),
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
        lowered = line.lower()

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

        if USER_QUERY_LINE_RE.match(line):
            continue
        if lowered.startswith("example"):
            continue
        if lowered.startswith("bm25 rules"):
            continue
        if line[:1] in {"-", "*"}:
            continue
        if lowered in {
            "assistant:",
            "search query:",
            "rewritten query:",
            "better bm25 query:",
            "better bm25 query",
            "better bm25",
            "better bm2",
        }:
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
    lower = candidate.lower()
    has_template_noise = int(
        any(marker in lower for marker in ("user query", "better bm25", "example", "bm25 rules"))
    )
    cleanliness = 1 if (has_template_noise == 0 and not _looks_like_explanation(candidate)) else 0
    return (cleanliness, overlap, completeness, -index, unique_count, len(tokens))


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
        self.recall_dense_k = max(1, int(reward_cfg.recall_dense_k))
        self.retrieval_k = max(self.mrr_k, self.recall_k, self.recall_dense_k)
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
        recall_dense, _, _ = compute_recall_at_k(
            hits_docids,
            relevant_docids,
            topk=self.recall_dense_k,
        )
        overlap = compute_lexical_overlap(source_query or "", cleaned_query) if source_query else 0.0
        if source_query:
            keyword_preserve = compute_keyword_preserve(source_query, cleaned_query)
            locked_term_preserve, number_preserve, acronym_preserve, negation_preserve = compute_locked_term_preserve(
                source_query,
                cleaned_query,
            )
            term_preserve = _combine_term_preserve(keyword_preserve, locked_term_preserve)
        else:
            keyword_preserve = 1.0
            locked_term_preserve = 1.0
            term_preserve, number_preserve, acronym_preserve, negation_preserve = (1.0, 1.0, 1.0, 1.0)
        length_score = compute_length_score(cleaned_query, self.cfg)
        bad_format_penalty = compute_bad_format_penalty(cleaned_query, self.cfg)
        clean_format = compute_clean_format_score(bad_format_penalty)
        unsafe_copy_penalty = (
            compute_unsafe_copy_penalty(source_query or "", cleaned_query) if source_query else 0.0
        )
        total = compose_reward(
            mrr=mrr,
            recall=recall,
            recall_dense=recall_dense,
            term_preserve=term_preserve,
            length_score=length_score,
            clean_format=clean_format,
            bad_format_penalty=bad_format_penalty,
            unsafe_copy_penalty=unsafe_copy_penalty,
            cfg=self.cfg,
        )
        return RewardBreakdown(
            total=total,
            mrr=mrr,
            recall=recall,
            recall_dense=recall_dense,
            overlap=overlap,
            term_preserve=term_preserve,
            keyword_preserve=keyword_preserve,
            locked_term_preserve=locked_term_preserve,
            number_preserve=number_preserve,
            acronym_preserve=acronym_preserve,
            negation_preserve=negation_preserve,
            length_score=length_score,
            clean_format=clean_format,
            bad_format_penalty=bad_format_penalty,
            unsafe_copy_penalty=unsafe_copy_penalty,
            hit_rank=hit_rank,
            retrieved_relevant_count=retrieved_relevant_count,
            relevant_total=relevant_total,
            rewritten_query=cleaned_query,
        )

    def score(self, qid: str, rewritten_query: str, source_query: str | None = None) -> RewardBreakdown:
        query = clean_rewritten_query((rewritten_query or "").strip(), source_query=source_query)
        hits_docids = self._search_docids(query)
        return self._score_one(qid, query, hits_docids, source_query)

    def score_batch(
        self,
        qid: str,
        rewritten_queries: Sequence[str],
        source_query: str | None = None,
    ) -> list[RewardBreakdown]:
        cleaned_queries = [
            clean_rewritten_query((text or "").strip(), source_query=source_query) for text in rewritten_queries
        ]
        hits_docids_batch = self._search_docids_batch(cleaned_queries)
        outputs: list[RewardBreakdown] = []
        for query, hits_docids in zip(cleaned_queries, hits_docids_batch):
            outputs.append(self._score_one(qid, query, hits_docids, source_query))
        return outputs
