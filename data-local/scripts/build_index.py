from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析索引构建参数。"""
    parser = argparse.ArgumentParser(description="Build BM25 index (Pyserini first, fallback to rank_bm25).")
    parser.add_argument("--input", default="data", help="Input data directory containing corpus.jsonl")
    parser.add_argument("--index", default="index", help="Output index directory")
    parser.add_argument("--backend", choices=["auto", "pyserini", "rank_bm25"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def pyserini_available() -> bool:
    """检测当前环境是否已安装 pyserini。"""
    return importlib.util.find_spec("pyserini") is not None


def write_backend_metadata(index_dir: Path, backend: str, corpus_file: Path, note: str | None = None) -> None:
    """记录实际检索后端，供评估阶段自动选择。"""
    index_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": backend,
        "corpus_file": str(corpus_file.resolve()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if note:
        payload["note"] = note
    with (index_dir / "backend.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_with_pyserini(corpus_file: Path, index_dir: Path, threads: int) -> bool:
    """调用 pyserini 构建 Lucene 索引。"""
    staging_dir = index_dir / "_pyserini_input"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging_corpus = staging_dir / "corpus.jsonl"
    shutil.copy2(corpus_file, staging_corpus)

    cmd = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(staging_dir),
        "--index",
        str(index_dir),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(threads),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    print("[info] Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def main() -> int:
    """脚本入口：优先 Pyserini，失败时回退 rank_bm25。"""
    args = parse_args()
    input_dir = Path(args.input)
    index_dir = Path(args.index)
    corpus_file = input_dir / "corpus.jsonl"

    if not corpus_file.exists():
        raise FileNotFoundError(f"Missing corpus file: {corpus_file}")

    requested = args.backend
    if requested == "pyserini" and not pyserini_available():
        print("[error] Backend was forced to pyserini but pyserini is not installed.")
        return 2

    backend = requested
    if requested == "auto":
        backend = "pyserini" if pyserini_available() else "rank_bm25"

    if backend == "pyserini":
        ok = build_with_pyserini(corpus_file, index_dir, args.threads)
        if ok:
            write_backend_metadata(index_dir, "pyserini", corpus_file)
            print(f"[ok] Pyserini BM25 index built at: {index_dir}")
            return 0
        if requested != "auto":
            print("[error] Pyserini indexing failed and fallback is disabled.")
            return 3
        print("[warn] Pyserini indexing failed, falling back to rank_bm25.")
        backend = "rank_bm25"

    write_backend_metadata(
        index_dir,
        backend,
        corpus_file,
        note="rank_bm25 backend uses in-memory index built during evaluation.",
    )
    print("[ok] Backend set to rank_bm25. No Lucene index is required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
