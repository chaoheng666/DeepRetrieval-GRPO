from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval_eval.core import compute_reward, iter_jsonl, tokenize


class PyseriniRetriever:
    """基于 Lucene 索引的检索器（高效 BM25）。"""

    def __init__(self, index_dir: Path):
        from pyserini.search import SimpleSearcher

        self.searcher = SimpleSearcher(str(index_dir))

    def retrieve(self, query: str, topk: int) -> list[tuple[str, float]]:
        hits = self.searcher.search(query, k=topk)
        return [(hit.docid, float(hit.score)) for hit in hits]


class RankBM25Retriever:
    """纯 Python 兜底检索器：本地加载 corpus 后用 rank_bm25 检索。"""

    def __init__(self, corpus_path: Path):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: rank-bm25. Run `python -m pip install -r requirements.txt`."
            ) from exc

        self.doc_ids: list[str] = []
        tokenized_docs: list[list[str]] = []
        for row in iter_jsonl(corpus_path):
            doc_id = row.get("id")
            contents = row.get("contents", "")
            tokens = tokenize(contents)
            if not doc_id or not tokens:
                continue
            self.doc_ids.append(doc_id)
            tokenized_docs.append(tokens)

        if not self.doc_ids:
            raise RuntimeError(f"No valid documents loaded from {corpus_path}")

        self.bm25 = BM25Okapi(tokenized_docs)

    def retrieve(self, query: str, topk: int) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
        return [(self.doc_ids[i], float(scores[i])) for i in ranked_idx]


def parse_args() -> argparse.Namespace:
    """解析评估参数。"""
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality with MRR reward.")
    parser.add_argument("--queries", default="data/queries.jsonl")
    parser.add_argument("--qrels", default="data/qrels.json")
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--index", default="index")
    parser.add_argument("--backend", choices=["auto", "pyserini", "rank_bm25"], default="auto")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-mrr", type=float, default=0.2)
    parser.add_argument("--min-nonzero-ratio", type=float, default=0.5)
    parser.add_argument("--report-path", default="runs/eval_report.json")
    return parser.parse_args()


def pyserini_available() -> bool:
    """检测 pyserini 是否可用。"""
    return importlib.util.find_spec("pyserini") is not None


def pick_backend(requested: str, index_dir: Path) -> str:
    """根据用户参数和 index 元数据选择后端。"""
    if requested != "auto":
        return requested

    backend_meta = index_dir / "backend.json"
    if backend_meta.exists():
        with backend_meta.open("r", encoding="utf-8") as f:
            payload = json.load(f)
            backend = payload.get("backend")
            if backend in {"pyserini", "rank_bm25"}:
                return backend
    return "pyserini"


def build_retriever(backend: str, args: argparse.Namespace):
    """按后端名称构建统一检索器实例。"""
    if backend == "pyserini":
        return PyseriniRetriever(Path(args.index)), "pyserini"
    if backend == "rank_bm25":
        return RankBM25Retriever(Path(args.corpus)), "rank_bm25"
    raise ValueError(f"Unsupported backend: {backend}")


def load_queries(path: Path) -> list[tuple[str, str]]:
    """读取 queries.jsonl。"""
    rows = []
    for row in iter_jsonl(path):
        qid = row.get("query_id")
        query = row.get("text")
        if not qid or not query:
            continue
        rows.append((qid, query))
    return rows


def load_qrels(path: Path) -> dict[str, Sequence[str]]:
    """读取 qrels.json。"""
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {str(k): list(v) for k, v in payload.items()}


def main() -> int:
    """脚本入口：采样 query，执行检索并计算 MRR 奖励。"""
    args = parse_args()
    queries = load_queries(Path(args.queries))
    qrels = load_qrels(Path(args.qrels))
    if not queries:
        raise RuntimeError("No queries found.")

    requested_backend = pick_backend(args.backend, Path(args.index))
    try:
        retriever, actual_backend = build_retriever(requested_backend, args)
    except Exception as exc:
        if args.backend == "auto" and requested_backend == "pyserini":
            print(f"[warn] Pyserini backend unavailable ({exc}). Falling back to rank_bm25.")
            retriever, actual_backend = build_retriever("rank_bm25", args)
        else:
            raise

    sample_size = min(args.sample_size, len(queries))
    rng = random.Random(args.seed)
    sampled_queries = rng.sample(queries, sample_size)

    rewards: list[float] = []
    zero_examples: list[dict] = []
    for qid, query in sampled_queries:
        results = retriever.retrieve(query, topk=args.topk)
        reward = compute_reward(qid, results, qrels)
        rewards.append(reward)
        if reward == 0.0 and len(zero_examples) < 20:
            zero_examples.append(
                {
                    "query_id": qid,
                    "query": query,
                    "topk_doc_ids": [doc_id for doc_id, _ in results],
                    "gold_doc_ids": list(qrels.get(qid, [])),
                }
            )

    avg_reward = statistics.fmean(rewards) if rewards else 0.0
    nonzero_ratio = (sum(1 for r in rewards if r > 0) / len(rewards)) if rewards else 0.0
    passed = avg_reward > args.min_mrr and nonzero_ratio > args.min_nonzero_ratio

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "backend": actual_backend,
        "sample_size": sample_size,
        "topk": args.topk,
        "avg_reward": avg_reward,
        "nonzero_ratio": nonzero_ratio,
        "thresholds": {"min_mrr": args.min_mrr, "min_nonzero_ratio": args.min_nonzero_ratio},
        "passed": passed,
        "zero_reward_examples": zero_examples,
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[info] backend={actual_backend}")
    print(f"[info] sample_size={sample_size}, topk={args.topk}")
    print(f"[result] avg_reward: {avg_reward:.4f}")
    print(f"[result] nonzero_ratio: {nonzero_ratio:.4f}")
    print(f"[result] report: {report_path}")

    if not passed:
        print("[fail] Threshold not met. Showing up to 20 zero-reward examples:")
        for item in zero_examples:
            print(f"- qid={item['query_id']} query={item['query']}")
            print(f"  topk={item['topk_doc_ids']}")
            print(f"  gold={item['gold_doc_ids']}")
        return 1

    print("[ok] Thresholds passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
