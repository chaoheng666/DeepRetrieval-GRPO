from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from core.reward_func import compute_mrr_at_k, compute_recall_at_k
from data.loader import QueryExample

# Curriculum 的核心思想：
# A 桶：原 query 在 Recall@100 能找到相关文档，但 MRR@20 不好，最适合学习 top20 改写；
# B 桶：原 query 几乎找不到相关文档，但文本质量还可以，用来扩大探索；
# C 桶：原 query 已经较强，少量保留，避免模型只学会激进改写。
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?(?:%|[a-z]+)?\b", flags=re.IGNORECASE)
POLLUTION_RE = re.compile(
    r"(?:<think|thinking process|analysis:|assistant:|search query:|rewritten query:)",
    flags=re.IGNORECASE,
)
NEGATION_TOKENS = {"no", "not", "without", "except", "excluding", "exclude"}
STOPWORD_TOKENS = {
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
    "can",
    "could",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "please",
    "should",
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
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "will",
    "with",
    "would",
}
PHASE_BUCKET_WEIGHTS: dict[str, dict[str, float]] = {
    "phase1": {"A": 0.80, "C": 0.15, "B": 0.05},
    "phase2": {"A": 0.60, "B": 0.20, "C": 0.20},
}
PHASE_SEED_OFFSETS: dict[str, int] = {
    "phase1": 0,
    "phase2": 50021,
}


@dataclass(frozen=True, slots=True)
class CurriculumQueryMetadata:
    """单个训练 query 的原始检索难度元数据。"""

    qid: str
    text: str
    orig_mrr20: float
    orig_recall20: float
    orig_recall100: float
    orig_best_hit_rank: int | None
    bucket: str


def _normalize_query_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _tokenize_terms(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def _english_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    english = sum(1 for ch in letters if "a" <= ch.lower() <= "z")
    return english / float(len(letters))


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


def _looks_like_bucket_b_candidate(query: str) -> bool:
    raw_query = query or ""
    if "\n" in raw_query or "\r" in raw_query:
        return False

    normalized = _normalize_query_text(raw_query)
    if not normalized:
        return False
    if POLLUTION_RE.search(normalized):
        return False
    if _english_ratio(normalized) < 0.70:
        return False
    if _unreadable_ratio(normalized) > 0.25:
        return False

    tokens = _tokenize_terms(normalized)
    if not 3 <= len(tokens) <= 12:
        return False

    non_stop_tokens = [token for token in tokens if token not in STOPWORD_TOKENS]
    if (len(non_stop_tokens) / float(len(tokens))) < 0.40:
        return False

    numeric_tokens = {match.group(0).lower() for match in NUMERIC_RE.finditer(normalized)}
    acronym_tokens = {match.group(0).lower() for match in ACRONYM_RE.finditer(normalized)}
    locked_terms = set(numeric_tokens) | set(acronym_tokens) | {token for token in tokens if token in NEGATION_TOKENS}
    keyword_terms = [
        token
        for token in tokens
        if token not in STOPWORD_TOKENS
        and token not in NEGATION_TOKENS
        and token not in numeric_tokens
        and token not in acronym_tokens
        and len(token) > 1
    ]
    return len(keyword_terms) >= 2 or len(locked_terms) >= 1


def bucket_for_query(*, orig_mrr20: float, orig_recall100: float, query_text: str) -> str:
    # A: 有可召回相关文档但 top20 排名差；B: 原检索失败但 query 文本可靠；C: 原 query 已经较好。
    if orig_recall100 > 0.0 and orig_mrr20 < 0.35:
        return "A"
    if orig_mrr20 >= 0.35:
        return "C"
    if orig_recall100 == 0.0 and _looks_like_bucket_b_candidate(query_text):
        return "B"
    return "DROP"


def load_curriculum_metadata(path: str | Path) -> dict[str, CurriculumQueryMetadata]:
    metadata_path = Path(path)
    if not metadata_path.exists():
        return {}

    loaded: dict[str, CurriculumQueryMetadata] = {}
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            row = CurriculumQueryMetadata(
                qid=str(payload["qid"]),
                text=str(payload["text"]),
                orig_mrr20=float(payload["orig_mrr20"]),
                orig_recall20=float(payload["orig_recall20"]),
                orig_recall100=float(payload["orig_recall100"]),
                orig_best_hit_rank=payload.get("orig_best_hit_rank"),
                bucket=str(payload["bucket"]),
            )
            loaded[row.qid] = row
    return loaded


def write_curriculum_metadata(path: str | Path, rows: Iterable[CurriculumQueryMetadata]) -> None:
    metadata_path = Path(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def build_curriculum_metadata(
    queries: Sequence[QueryExample],
    rewarder,
    *,
    batch_size: int = 128,
    progress_every: int = 1000,
) -> list[CurriculumQueryMetadata]:
    # 首次训练会扫描 train split 的原 query 检索效果，并写入 jsonl 供后续 phase 复用。
    built: list[CurriculumQueryMetadata] = []
    step = max(1, int(progress_every))
    for start in range(0, len(queries), max(1, int(batch_size))):
        batch = list(queries[start : start + max(1, int(batch_size))])
        hits_docids_batch = rewarder._search_docids_batch([query.text for query in batch], k=100)
        for idx, (query, hits_docids) in enumerate(zip(batch, hits_docids_batch), start=start + 1):
            relevant_docids = rewarder.qrels.get(str(query.qid), set())
            orig_mrr20, orig_best_hit_rank = compute_mrr_at_k(hits_docids, relevant_docids, topk=20)
            orig_recall20, _, _ = compute_recall_at_k(hits_docids, relevant_docids, topk=20)
            orig_recall100, _, _ = compute_recall_at_k(hits_docids, relevant_docids, topk=100)
            built.append(
                CurriculumQueryMetadata(
                    qid=str(query.qid),
                    text=query.text,
                    orig_mrr20=orig_mrr20,
                    orig_recall20=orig_recall20,
                    orig_recall100=orig_recall100,
                    orig_best_hit_rank=orig_best_hit_rank,
                    bucket=bucket_for_query(
                        orig_mrr20=orig_mrr20,
                        orig_recall100=orig_recall100,
                        query_text=query.text,
                    ),
                )
            )
            if idx % step == 0 or idx == len(queries):
                print(f"[curriculum] scanned {idx}/{len(queries)} train queries")
    return built


def ensure_curriculum_metadata(
    queries: Sequence[QueryExample],
    rewarder,
    *,
    metadata_path: str | Path,
    batch_size: int = 128,
    progress_every: int = 1000,
) -> dict[str, CurriculumQueryMetadata]:
    # metadata 存在且覆盖当前 train qid 时直接复用，否则重建。
    metadata_file = Path(metadata_path)
    loaded = load_curriculum_metadata(metadata_file)
    required_qids = {str(query.qid) for query in queries}
    if required_qids and required_qids.issubset(loaded.keys()):
        return {qid: loaded[qid] for qid in required_qids}

    rebuilt = build_curriculum_metadata(
        queries,
        rewarder,
        batch_size=batch_size,
        progress_every=progress_every,
    )
    write_curriculum_metadata(metadata_file, rebuilt)
    return {row.qid: row for row in rebuilt}


def curriculum_bucket_counts(
    queries: Sequence[QueryExample],
    metadata_by_qid: dict[str, CurriculumQueryMetadata],
) -> dict[str, int]:
    counts = {"A": 0, "B": 0, "C": 0, "DROP": 0}
    for query in queries:
        bucket = metadata_by_qid.get(str(query.qid), CurriculumQueryMetadata(
            qid=str(query.qid),
            text=query.text,
            orig_mrr20=0.0,
            orig_recall20=0.0,
            orig_recall100=0.0,
            orig_best_hit_rank=None,
            bucket="DROP",
        )).bucket
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def filter_curriculum_train_queries(
    queries: Sequence[QueryExample],
    metadata_by_qid: dict[str, CurriculumQueryMetadata],
) -> tuple[list[QueryExample], dict[str, int]]:
    # DROP 样本不进入主线训练，避免没有学习信号或文本质量太差的 query 干扰 reward。
    kept: list[QueryExample] = []
    counts = {"A": 0, "B": 0, "C": 0, "DROP": 0}
    for query in queries:
        meta = metadata_by_qid.get(str(query.qid))
        bucket = meta.bucket if meta is not None else "DROP"
        counts[bucket] = counts.get(bucket, 0) + 1
        if bucket != "DROP":
            kept.append(query)
    return kept, counts


def _compute_bucket_quotas(total: int, weights: dict[str, float]) -> dict[str, int]:
    raw = {bucket: total * weight for bucket, weight in weights.items()}
    quotas = {bucket: int(value) for bucket, value in raw.items()}
    remainder = max(0, total - sum(quotas.values()))
    ranked = sorted(
        weights,
        key=lambda bucket: (raw[bucket] - quotas[bucket], weights[bucket], bucket),
        reverse=True,
    )
    for bucket in ranked[:remainder]:
        quotas[bucket] += 1
    return quotas


def _cycled_shuffled_items(items: Sequence[QueryExample], count: int, rng: random.Random) -> list[QueryExample]:
    if count <= 0:
        return []
    pool = list(items)
    if not pool:
        return []

    output: list[QueryExample] = []
    cursor_pool = pool.copy()
    while len(output) < count:
        rng.shuffle(cursor_pool)
        needed = count - len(output)
        output.extend(cursor_pool[: min(needed, len(cursor_pool))])
    return output


def sample_curriculum_queries(
    queries: Sequence[QueryExample],
    metadata_by_qid: dict[str, CurriculumQueryMetadata],
    *,
    phase: str,
    seed: int,
    epoch: int,
) -> list[QueryExample]:
    # 每个 epoch 按 phase 权重重新采样并打乱，phase2 会比 phase1 多看 B/C 桶。
    phase_key = str(phase or "phase1").strip().lower()
    if phase_key not in PHASE_BUCKET_WEIGHTS:
        raise ValueError(f"Unsupported curriculum phase: {phase}")

    bucket_to_queries: dict[str, list[QueryExample]] = {"A": [], "B": [], "C": []}
    for query in queries:
        meta = metadata_by_qid.get(str(query.qid))
        if meta is None or meta.bucket == "DROP":
            continue
        if meta.bucket in bucket_to_queries:
            bucket_to_queries[meta.bucket].append(query)

    active_weights = {
        bucket: weight
        for bucket, weight in PHASE_BUCKET_WEIGHTS[phase_key].items()
        if bucket_to_queries.get(bucket)
    }
    if not active_weights:
        return list(queries)

    weight_total = sum(active_weights.values())
    normalized_weights = {bucket: weight / weight_total for bucket, weight in active_weights.items()}
    total = sum(len(bucket_to_queries[bucket]) for bucket in active_weights)
    quotas = _compute_bucket_quotas(total, normalized_weights)

    # Keep each phase deterministic, but avoid phase2 reusing the same prefix as phase1
    # when max_steps cuts the epoch short.
    rng = random.Random(int(seed) + int(epoch) * 1009 + PHASE_SEED_OFFSETS.get(phase_key, 0))
    bucket_schedule: list[str] = []
    for bucket, quota in quotas.items():
        bucket_schedule.extend([bucket] * quota)
    rng.shuffle(bucket_schedule)

    bucket_draws = {
        bucket: _cycled_shuffled_items(bucket_to_queries[bucket], quotas.get(bucket, 0), rng)
        for bucket in active_weights
    }
    bucket_positions = {bucket: 0 for bucket in active_weights}

    ordered: list[QueryExample] = []
    for bucket in bucket_schedule:
        bucket_items = bucket_draws[bucket]
        position = bucket_positions[bucket]
        if position >= len(bucket_items):
            continue
        ordered.append(bucket_items[position])
        bucket_positions[bucket] += 1
    return ordered
