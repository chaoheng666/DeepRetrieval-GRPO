from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval_eval.core import get_or_create_doc_id, normalize_text, write_jsonl


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Prepare corpus/queries/qrels from HF dataset.")
    parser.add_argument("--dataset", default="ms_marco", help="Dataset name. Default: ms_marco")
    parser.add_argument("--split", default="train[:5000]", help="Split expression for ms_marco")
    parser.add_argument("--seed", type=int, default=42, help="Seed kept for reproducibility")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    return parser.parse_args()


def load_msmarco(split: str):
    """加载 MS MARCO；当 split 形如 train[:N] 时优先走流式读取。"""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: datasets. Run `python -m pip install -r requirements.txt`."
        ) from exc

    match = re.fullmatch(r"([A-Za-z0-9_]+)\[:(\d+)\]", split)
    if match:
        base_split = match.group(1)
        limit = int(match.group(2))
        try:
            stream = load_dataset("ms_marco", "v1.1", split=base_split, streaming=True)
            return itertools.islice(stream, limit)
        except Exception as exc:
            print(f"[warn] Streaming load failed ({exc}), falling back to standard loading.")

    return load_dataset("ms_marco", "v1.1", split=split)


def load_beir_fallback():
    """在 MS MARCO 不可用时，尝试从 BeIR/msmarco 读取 queries/corpus/qrels。"""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: datasets. Run `python -m pip install -r requirements.txt`."
        ) from exc

    attempts: list[tuple[str, str | None, str]] = [
        ("BeIR/msmarco", "queries", "queries"),
        ("BeIR/msmarco", None, "queries"),
    ]
    queries_ds = None
    last_error: Exception | None = None
    for path, name, split in attempts:
        try:
            queries_ds = load_dataset(path, name, split=split) if name else load_dataset(path, split=split)
            break
        except Exception as exc:  # pragma: no cover - fallback only
            last_error = exc
    if queries_ds is None:  # pragma: no cover - fallback only
        raise RuntimeError("Failed to load BeIR/msmarco queries split.") from last_error

    corpus_attempts: list[tuple[str, str | None, str]] = [
        ("BeIR/msmarco", "corpus", "corpus"),
        ("BeIR/msmarco", None, "corpus"),
    ]
    corpus_ds = None
    for path, name, split in corpus_attempts:
        try:
            corpus_ds = load_dataset(path, name, split=split) if name else load_dataset(path, split=split)
            break
        except Exception as exc:  # pragma: no cover - fallback only
            last_error = exc
    if corpus_ds is None:  # pragma: no cover - fallback only
        raise RuntimeError("Failed to load BeIR/msmarco corpus split.") from last_error

    qrels_attempts: list[tuple[str, str | None, str]] = [
        ("BeIR/msmarco", "qrels", "train"),
        ("BeIR/msmarco", "qrels", "qrels"),
        ("BeIR/msmarco-qrels", None, "train"),
    ]
    qrels_ds = None
    for path, name, split in qrels_attempts:
        try:
            qrels_ds = load_dataset(path, name, split=split) if name else load_dataset(path, split=split)
            break
        except Exception as exc:  # pragma: no cover - fallback only
            last_error = exc
    if qrels_ds is None:  # pragma: no cover - fallback only
        raise RuntimeError("Failed to load BeIR/msmarco qrels split.") from last_error

    return corpus_ds, queries_ds, qrels_ds


def _extract_field(obj: dict[str, Any], candidates: list[str], default=None):
    """兼容不同数据集字段命名，按候选 key 顺序取值。"""
    for key in candidates:
        if key in obj:
            return obj[key]
    return default


def prepare_from_msmarco(msm_ds):
    """从 MS MARCO 样本构建 corpus / queries / qrels。"""
    corpus_rows: list[dict[str, str]] = []
    query_rows: list[dict[str, str]] = []
    qrels: dict[str, list[str]] = {}
    text_to_doc: dict[str, str] = {}
    doc_to_text: dict[str, str] = {}

    for idx, item in enumerate(msm_ds):
        qid = f"Q{idx}"
        query_text = normalize_text(item.get("query", ""))
        if not query_text:
            continue
        query_rows.append({"query_id": qid, "text": query_text})

        passages = item.get("passages") or {}
        passage_texts = passages.get("passage_text") or []
        labels = passages.get("is_selected") or []

        rel_docs: list[str] = []
        for j, text in enumerate(passage_texts):
            doc_id = get_or_create_doc_id(text, text_to_doc, doc_to_text)
            if doc_id is None:
                continue
            if len(labels) > j and labels[j] == 1:
                rel_docs.append(doc_id)

        unique_rel_docs = sorted(set(rel_docs))
        if unique_rel_docs:
            qrels[qid] = unique_rel_docs

    for doc_id, text in doc_to_text.items():
        corpus_rows.append({"id": doc_id, "contents": text})

    return corpus_rows, query_rows, qrels


def prepare_from_beir(corpus_ds, queries_ds, qrels_ds):  # pragma: no cover - fallback only
    """从 BeIR 结构构建统一数据格式。"""
    text_to_doc: dict[str, str] = {}
    doc_to_text: dict[str, str] = {}
    raw_doc_to_doc: dict[str, str] = {}
    corpus_rows: list[dict[str, str]] = []

    for row in corpus_ds:
        raw_doc_id = str(_extract_field(row, ["_id", "doc_id", "corpus_id", "id"], ""))
        text = normalize_text(
            f"{_extract_field(row, ['title'], '') or ''} {_extract_field(row, ['text', 'contents'], '') or ''}"
        )
        doc_id = get_or_create_doc_id(text, text_to_doc, doc_to_text)
        if doc_id is None:
            continue
        if raw_doc_id:
            raw_doc_to_doc[raw_doc_id] = doc_id

    for doc_id, text in doc_to_text.items():
        corpus_rows.append({"id": doc_id, "contents": text})

    raw_query_to_qid: dict[str, str] = {}
    query_rows: list[dict[str, str]] = []
    for idx, row in enumerate(queries_ds):
        raw_qid = str(_extract_field(row, ["_id", "query_id", "qid", "id"], idx))
        text = normalize_text(_extract_field(row, ["text", "query"], ""))
        if not text:
            continue
        qid = f"Q{idx}"
        raw_query_to_qid[raw_qid] = qid
        query_rows.append({"query_id": qid, "text": text})

    qrels: dict[str, list[str]] = {}
    for row in qrels_ds:
        raw_qid = str(_extract_field(row, ["query-id", "query_id", "qid", "query"], ""))
        raw_doc_id = str(_extract_field(row, ["corpus-id", "corpus_id", "doc_id", "docid"], ""))
        score = _extract_field(row, ["score", "relevance", "label"], 1)
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            continue
        if not raw_qid or not raw_doc_id or score_value <= 0:
            continue
        qid = raw_query_to_qid.get(raw_qid)
        doc_id = raw_doc_to_doc.get(raw_doc_id)
        if qid is None or doc_id is None:
            continue
        qrels.setdefault(qid, []).append(doc_id)

    for qid, docs in list(qrels.items()):
        qrels[qid] = sorted(set(docs))

    return corpus_rows, query_rows, qrels


def compute_stats(corpus_rows, query_rows, qrels):
    """统计数据规模和 qrels 覆盖率，便于快速排查对齐问题。"""
    corpus_doc_ids = {row["id"] for row in corpus_rows}
    total_rel_docs = sum(len(doc_ids) for doc_ids in qrels.values())
    covered_rel_docs = sum(sum(1 for doc_id in doc_ids if doc_id in corpus_doc_ids) for doc_ids in qrels.values())
    coverage = (covered_rel_docs / total_rel_docs) if total_rel_docs else 0.0
    positives = [len(doc_ids) for doc_ids in qrels.values()]

    return {
        "num_docs": len(corpus_rows),
        "num_queries": len(query_rows),
        "num_qrels_queries": len(qrels),
        "num_empty_qrels_queries": len(query_rows) - len(qrels),
        "avg_relevant_docs_per_qrel_query": fmean(positives) if positives else 0.0,
        "qrels_doc_coverage": coverage,
    }


def main() -> int:
    """脚本入口：拉取数据、构建三大文件并输出统计信息。"""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset.lower()
    source = ""
    if dataset_name in {"ms_marco", "msmarco"}:
        try:
            msm_ds = load_msmarco(args.split)
            corpus_rows, query_rows, qrels = prepare_from_msmarco(msm_ds)
            source = "ms_marco/v1.1"
        except Exception as exc:
            print(f"[warn] Failed to load ms_marco split '{args.split}': {exc}")
            print("[warn] Trying BeIR/msmarco fallback...")
            corpus_ds, queries_ds, qrels_ds = load_beir_fallback()
            corpus_rows, query_rows, qrels = prepare_from_beir(corpus_ds, queries_ds, qrels_ds)
            source = "BeIR/msmarco"
    elif dataset_name in {"beir/msmarco", "beir_msmarco"}:
        corpus_ds, queries_ds, qrels_ds = load_beir_fallback()
        corpus_rows, query_rows, qrels = prepare_from_beir(corpus_ds, queries_ds, qrels_ds)
        source = "BeIR/msmarco"
    else:
        raise ValueError(f"Unsupported dataset '{args.dataset}'. Use ms_marco or BeIR/msmarco.")

    write_jsonl(output_dir / "corpus.jsonl", corpus_rows)
    write_jsonl(output_dir / "queries.jsonl", query_rows)
    with (output_dir / "qrels.json").open("w", encoding="utf-8") as f:
        json.dump(qrels, f, ensure_ascii=False, indent=2)

    stats = compute_stats(corpus_rows, query_rows, qrels)
    stats["dataset"] = source
    stats["split"] = args.split
    stats["seed"] = args.seed
    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[ok] Corpus written to: {output_dir / 'corpus.jsonl'} ({stats['num_docs']} docs)")
    print(f"[ok] Queries written to: {output_dir / 'queries.jsonl'} ({stats['num_queries']} queries)")
    print(f"[ok] Qrels written to: {output_dir / 'qrels.json'} ({stats['num_qrels_queries']} qids)")
    print(
        "[ok] Coverage:"
        f" {stats['qrels_doc_coverage']:.4f}, empty_qrels={stats['num_empty_qrels_queries']},"
        f" avg_pos={stats['avg_relevant_docs_per_qrel_query']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
