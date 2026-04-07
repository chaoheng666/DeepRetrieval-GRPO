from __future__ import annotations

"""奖励函数模块。

总体设计：
  total_reward = retrieval_reward - text_penalty

其中：
1. retrieval_reward: 通过 Pyserini BM25 检索得到 MRR@k。
2. text_penalty: 轻量规则惩罚（过短、重复、不可读字符比例过高）。

这样做的目的：
- 让优化目标直接对齐检索质量（MRR）。
- 通过简单文本约束减少模型生成退化 query。
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from app_config import RewardConfig

TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
MARKER_LINE_RE = re.compile(r"^(?:rewritten\s+query|search\s+query)\s*:\s*(.*)$", flags=re.IGNORECASE)
MARKER_INLINE_RE = re.compile(r"(?:rewritten\s+query|search\s+query)\s*:\s*([^\n\r]+)", flags=re.IGNORECASE)
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
class TextPenaltyDetails:
    """文本惩罚细分结果。"""

    total: float
    short: float
    repeat: float
    unreadable: float


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """结构化奖励输出（便于训练日志与调试分析）。"""

    total: float
    mrr: float
    overlap: float
    penalty: float
    hit_rank: int | None
    short_penalty: float
    repeat_penalty: float
    unreadable_penalty: float
    rewritten_query: str


def compute_mrr_at_k(result_docids: Sequence[str], relevant_docids: Iterable[str], topk: int = 10) -> tuple[float, int | None]:
    """计算单条 query 的 MRR@k。

    参数：
    - result_docids: 检索返回的 docid 列表（按相关性降序）
    - relevant_docids: 该 query 的相关文档集合
    - topk: 截断深度

    返回：
    - reciprocal_rank: 若命中则为 1/rank，否则 0
    - hit_rank: 命中的名次；未命中时为 None
    """

    relevant = set(relevant_docids)
    if not relevant:
        return 0.0, None

    for rank, docid in enumerate(result_docids[:topk], start=1):
        if docid in relevant:
            return 1.0 / rank, rank
    return 0.0, None


def _repeat_ratio(tokens: list[str]) -> float:
    """估算 token 重复比例（范围 [0, 1]）。

    定义：
      repeat_ratio = 1 - (unique_token_count / total_token_count)
    当值较高时，通常表示生成退化（循环重复词）。
    """

    if not tokens:
        return 0.0
    unique = len(set(tokens))
    return 1.0 - (unique / float(len(tokens)))


def _unreadable_ratio(text: str) -> float:
    """估算不可读字符比例。

    这里把字母数字、空白和常见标点视作可读字符，其余字符计入不可读。
    这是一个轻量启发式规则，不追求语言学完备，只用于快速质量兜底。
    """

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


def compute_text_penalty(text: str, cfg: RewardConfig) -> TextPenaltyDetails:
    """计算文本惩罚。

    包含三项：
    1. 过短惩罚：长度小于 min_query_chars
    2. 重复惩罚：重复率高于 max_repeat_ratio
    3. 不可读惩罚：不可读字符比例高于 max_unreadable_char_ratio
    """

    cleaned = text.strip()
    short_penalty = cfg.penalty_short if len(cleaned) < cfg.min_query_chars else 0.0

    tokens = TOKEN_RE.findall(cleaned.lower())
    repeat_penalty = cfg.penalty_repeat if _repeat_ratio(tokens) > cfg.max_repeat_ratio else 0.0

    unreadable_penalty = cfg.penalty_unreadable if _unreadable_ratio(cleaned) > cfg.max_unreadable_char_ratio else 0.0

    total = short_penalty + repeat_penalty + unreadable_penalty
    return TextPenaltyDetails(
        total=total,
        short=short_penalty,
        repeat=repeat_penalty,
        unreadable=unreadable_penalty,
    )


def _tokenize_for_overlap(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def compute_lexical_overlap(source_query: str, rewritten_query: str) -> float:
    """计算原 query 与重写 query 的词面重叠分（0~1）。

    这里采用集合 Jaccard：
      |A ∩ B| / |A ∪ B|
    作为轻量语义保真近似项，避免纯 MRR 过于稀疏导致训练无梯度信号。
    """

    src = set(_tokenize_for_overlap(source_query))
    rew = set(_tokenize_for_overlap(rewritten_query))
    if not src or not rew:
        return 0.0
    union = src | rew
    if not union:
        return 0.0
    return len(src & rew) / float(len(union))


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
    """清洗模型输出并选择最可用的一条检索 query。

    策略：
    1) 从 marker 行、普通行中提取候选并标准化；
    2) 若提供 source_query，优先按词面重叠排序；
    3) 同分时优先更完整、信息量更高的候选，避免截断残句。
    """

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
        """初始化奖励器。

        关键点：
        - 只在初始化时加载一次 LuceneSearcher，避免每次打分重复开销。
        - qrels 保存在内存中，打分时 O(1) 访问相关文档集合。
        """

        from pyserini.search.lucene import LuceneSearcher

        searcher = self._build_searcher_with_recovery(LuceneSearcher, prebuilt_index)
        if searcher is None:
            raise RuntimeError(f"Failed to initialize prebuilt index: {prebuilt_index}")

        self.searcher = searcher
        self.qrels = qrels
        self.cfg = reward_cfg
        self.topk = reward_cfg.topk

    @staticmethod
    def _extract_corrupted_index_path(error_text: str) -> Path | None:
        """从 Pyserini 的 size mismatch 报错中提取损坏压缩包路径。"""

        # 典型报错形态：
        # C:\...\file.tar.gz does not match expected file size! ...
        marker = " does not match expected file size"
        idx = error_text.find(marker)
        if idx <= 0:
            return None
        path_text = error_text[:idx].strip()
        candidate = Path(path_text)
        return candidate if candidate.suffixes else None

    def _build_searcher_with_recovery(self, lucene_searcher_cls, prebuilt_index: str):
        """构建检索器，遇到损坏索引缓存时自动清理并重试一次。"""

        for attempt in range(2):
            try:
                return lucene_searcher_cls.from_prebuilt_index(prebuilt_index)
            except AssertionError as exc:
                # Pyserini 下载中断后，缓存 tar.gz 可能尺寸不完整，后续启动会一直失败。
                # 这里自动删除坏包并重试一次。
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

    def score(self, qid: str, rewritten_query: str, source_query: str | None = None) -> RewardBreakdown:
        """为单条重写 query 打分。

        流程：
        1. 用重写文本检索 topk 文档
        2. 根据 qrels 计算 MRR@k
        3. 计算文本惩罚
        4. 汇总 total = mrr - penalty
        """

        query = clean_rewritten_query(rewritten_query, source_query=source_query)
        relevant_docids = self.qrels.get(str(qid), set())

        hits_docids: list[str] = []
        if query:
            try:
                # 按需求直接调用 Pyserini 预编译索引检索接口。
                hits = self.searcher.search(query, k=self.topk)
                hits_docids = [str(hit.docid) for hit in hits]
            except Exception:
                # 检索异常时降级为“无命中”，保证训练流程不中断。
                hits_docids = []

        mrr, hit_rank = compute_mrr_at_k(hits_docids, relevant_docids, topk=self.topk)
        overlap = compute_lexical_overlap(source_query or "", query) if source_query else 0.0
        penalty = compute_text_penalty(query, self.cfg)
        total = self.cfg.mrr_weight * mrr + self.cfg.overlap_weight * overlap - penalty.total
        return RewardBreakdown(
            total=total,
            mrr=mrr,
            overlap=overlap,
            penalty=penalty.total,
            hit_rank=hit_rank,
            short_penalty=penalty.short,
            repeat_penalty=penalty.repeat,
            unreadable_penalty=penalty.unreadable,
            rewritten_query=query,
        )
