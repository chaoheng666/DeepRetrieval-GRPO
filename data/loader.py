from __future__ import annotations

"""数据加载工具。

主线直接使用 Pyserini 的 topics/qrels：
1. 不做自定义数据格式转换，避免额外依赖和中间状态；
2. 只保留有正相关 qrels 的 query，保证 MRR/Recall 奖励有监督信号；
3. 输出最小训练结构 QueryExample，后续 curriculum 和 rewarder 共用。
"""

import os
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from app_config import patch_pyserini_prebuilt_index_urls


@dataclass(frozen=True, slots=True)
class QueryExample:
    """单条 query 样本。

    qid 来自 Pyserini topic/qrels，text 是用于 prompt 和检索的原始 query 文本。
    """

    qid: str
    text: str


def _parse_java_major(version_output: str) -> int | None:
    """从 `java -version` 输出中解析主版本号。"""

    match = re.search(r'version\s+"([^"]+)"', version_output)
    if not match:
        return None

    raw = match.group(1).strip()
    if raw.startswith("1."):
        # 旧格式：1.8 -> Java 8。
        parts = raw.split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1])
        return None

    first = raw.split(".")[0]
    return int(first) if first.isdigit() else None


def _run_java_version(java_exec: str) -> tuple[int | None, str]:
    """执行 `<java_exec> -version`，返回 (major, raw_output)。"""

    try:
        proc = subprocess.run(
            [java_exec, "-version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None, ""

    raw = (proc.stderr or "") + "\n" + (proc.stdout or "")
    return _parse_java_major(raw), raw.strip()


def _java_home_from_java_exec(java_exec_path: str) -> str | None:
    """根据 java 可执行文件路径推断 JAVA_HOME。"""

    exe = Path(java_exec_path).resolve()
    # 常见结构：<JAVA_HOME>/bin/java(.exe)。
    if exe.parent.name.lower() == "bin":
        return str(exe.parent.parent)
    return None


def ensure_java_runtime(min_major: int = 21) -> None:
    """确保 Pyserini 使用兼容的 Java 版本。

    优先使用 JAVA_HOME；如果 JAVA_HOME 不合格但 PATH 中 java 合格，
    就自动把 JAVA_HOME 切到 PATH java 对应的 JDK 根目录。
    """

    java_home = os.environ.get("JAVA_HOME", "").strip()
    home_major = None
    home_raw = ""
    if java_home:
        java_in_home = str(Path(java_home) / "bin" / ("java.exe" if os.name == "nt" else "java"))
        home_major, home_raw = _run_java_version(java_in_home)

    java_exec = shutil.which("java")
    path_major = None
    path_raw = ""
    if java_exec:
        path_major, path_raw = _run_java_version(java_exec)

    if home_major is not None and home_major >= min_major:
        return

    if path_major is not None and path_major >= min_major and java_exec:
        inferred_home = _java_home_from_java_exec(java_exec)
        if inferred_home:
            os.environ["JAVA_HOME"] = inferred_home
            bin_path = str(Path(inferred_home) / "bin")
            current_path = os.environ.get("PATH", "")
            if not current_path.lower().startswith(bin_path.lower()):
                os.environ["PATH"] = bin_path + os.pathsep + current_path
            return

    raise RuntimeError(
        "Java runtime is incompatible for Pyserini. "
        f"Require Java >= {min_major}, but got JAVA_HOME={java_home!r} (major={home_major}) "
        f"and PATH java={java_exec!r} (major={path_major}).\n"
        "Please install JDK 21+ and ensure JAVA_HOME points to that JDK.\n"
        "Current JAVA_HOME java -version output:\n"
        f"{home_raw or '(unavailable)'}\n"
        "Current PATH java -version output:\n"
        f"{path_raw or '(unavailable)'}"
    )


def _extract_topic_text(topic: Mapping[str, str]) -> str:
    """从 topic 字典中提取可用 query 文本。

    不同 benchmark 字段名可能不同，因此按常见优先级查找：
    title -> query -> question -> description -> text。
    """

    preferred_keys = ("title", "query", "question", "description", "text")
    for key in preferred_keys:
        value = topic.get(key)
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned

    for value in topic.values():
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
    return ""


def _to_positive_qrels(qrels_raw: Mapping) -> dict[str, set[str]]:
    """把原始 qrels 转成 `{qid: {positive_docid}}`。

    只保留 judgement > 0 的 docid，因为 MRR/Recall 只依赖正相关集合。
    """

    qrels: dict[str, set[str]] = {}
    for qid, doc_judgements in qrels_raw.items():
        qid_str = str(qid)
        positive: set[str] = set()

        if isinstance(doc_judgements, Mapping):
            for docid, judgement in doc_judgements.items():
                try:
                    score = float(judgement)
                except (TypeError, ValueError):
                    continue
                if score > 0:
                    positive.add(str(docid))

        if positive:
            qrels[qid_str] = positive
    return qrels


def _sort_key_qid(qid: str) -> tuple[int, int | str]:
    """qid 排序键：纯数字按数值排序，否则按字符串排序。"""

    return (0, int(qid)) if qid.isdigit() else (1, qid)


def load_topics_qrels(topic_name: str) -> tuple[list[QueryExample], dict[str, set[str]]]:
    """通过 Pyserini 加载 topics 和 qrels。

    返回：
    - queries: 按 qid 排序后的 QueryExample，只保留有正相关 qrels 的 query；
    - qrels: `{qid -> 正相关 docid 集合}`。
    """

    # 导入 pyserini 前先确保 Java/JAVA_HOME 可用，并把预建索引 URL 切到 HF_ENDPOINT。
    ensure_java_runtime(min_major=21)
    patch_pyserini_prebuilt_index_urls()

    from pyserini.search import get_qrels, get_topics

    topics = get_topics(topic_name)
    qrels = _to_positive_qrels(get_qrels(topic_name))

    queries: list[QueryExample] = []
    for qid, payload in topics.items():
        qid_str = str(qid)
        # 没有正相关文档的 query 奖励恒为 0，训练时会引入噪声，直接过滤。
        if qid_str not in qrels:
            continue

        if isinstance(payload, Mapping):
            query_text = _extract_topic_text(payload)
        else:
            query_text = str(payload).strip()

        if not query_text:
            continue
        queries.append(QueryExample(qid=qid_str, text=query_text))

    queries.sort(key=lambda item: _sort_key_qid(item.qid))
    return queries, qrels


def split_queries(
    queries: Sequence[QueryExample],
    train_ratio: float,
    seed: int,
) -> tuple[list[QueryExample], list[QueryExample]]:
    """按比例切分 train/val，并保持可复现。"""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if len(queries) < 2:
        raise ValueError("Need at least 2 queries to split train/val.")

    items = list(queries)
    random.Random(seed).shuffle(items)
    split_idx = int(len(items) * train_ratio)
    split_idx = max(1, min(split_idx, len(items) - 1))
    return items[:split_idx], items[split_idx:]


def maybe_limit(queries: Iterable[QueryExample], limit: int | None) -> list[QueryExample]:
    """可选截断，用于快速调试或限制训练/评估规模。"""

    if limit is None:
        return list(queries)
    return list(queries)[: max(0, limit)]
