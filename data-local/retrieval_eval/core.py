from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def normalize_text(text: str) -> str:
    """标准化文本：压缩空白，便于稳定映射和检索。"""
    return " ".join(str(text).split())


def _stable_doc_id_from_normalized(normalized_text: str) -> str:
    digest = hashlib.sha1(normalized_text.encode("utf-8")).hexdigest()
    return f"D{digest}"


def get_or_create_doc_id(
    text: str,
    text_to_doc: dict[str, str],
    doc_to_text: dict[str, str],
) -> str | None:
    """将 passage 文本映射为稳定 doc_id；空文本返回 None。"""
    normalized = normalize_text(text)
    if not normalized:
        return None

    existing = text_to_doc.get(normalized)
    if existing is not None:
        return existing

    doc_id = _stable_doc_id_from_normalized(normalized)
    text_to_doc[normalized] = doc_id
    doc_to_text[doc_id] = normalized
    return doc_id


def tokenize(text: str) -> list[str]:
    """轻量分词：小写 + 规则切词，用于 rank_bm25。"""
    normalized = normalize_text(text).lower()
    if not normalized:
        return []
    return WORD_RE.findall(normalized)


def compute_reward(
    qid: str,
    results: Sequence[tuple[str, float]],
    qrels: Mapping[str, Iterable[str]],
) -> float:
    """按 MRR 计算单条 query reward。"""
    if qid not in qrels:
        return 0.0

    rel_docs = set(qrels[qid])
    for rank, (doc_id, _) in enumerate(results, start=1):
        if doc_id in rel_docs:
            return 1.0 / rank
    return 0.0


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """逐行读取 JSONL，跳过空行。"""
    data_path = Path(path)
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """写入 JSONL，并自动创建父目录。"""
    data_path = Path(path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
