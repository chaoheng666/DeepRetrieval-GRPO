from __future__ import annotations

"""数据加载工具（直接基于 Pyserini 预编译资源）。

设计原则：
1. 不做自定义数据集转换，不引入额外数据依赖。
2. 直接读取 Pyserini 的 topics + qrels，保证与检索评测生态对齐。
3. 只产出训练需要的最小结构，避免复杂中间格式。
"""

import os
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class QueryExample:
    """单条查询样本。

    字段说明：
    - qid: 查询唯一标识（来自 topics/qrels）
    - text: 查询文本
    """

    qid: str
    text: str


def _parse_java_major(version_output: str) -> int | None:
    """从 `java -version` 输出中提取主版本号。

    兼容示例：
    - java version "23" ...
    - openjdk version "21.0.2" ...
    - java version "1.8.0_372" ...
    """

    match = re.search(r'version\s+"([^"]+)"', version_output)
    if not match:
        return None

    raw = match.group(1).strip()
    if raw.startswith("1."):
        # 旧格式：1.8 -> Java 8
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
    # 常见结构：<JAVA_HOME>/bin/java(.exe)
    if exe.parent.name.lower() == "bin":
        return str(exe.parent.parent)
    return None


def ensure_java_runtime(min_major: int = 21) -> None:
    """确保 Pyserini 使用兼容 Java 版本。

    修复策略：
    1. 若 JAVA_HOME 指向的 Java >= min_major，直接使用。
    2. 若 JAVA_HOME 过低，但 PATH 上 java >= min_major，则自动把 JAVA_HOME
       切到该可执行文件对应的 JDK 根目录。
    3. 若两者都不满足，抛出带操作建议的异常。
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

    # 情况1：JAVA_HOME 本身合格。
    if home_major is not None and home_major >= min_major:
        return

    # 情况2：PATH 上的 java 合格，自动修复 JAVA_HOME。
    if path_major is not None and path_major >= min_major and java_exec:
        inferred_home = _java_home_from_java_exec(java_exec)
        if inferred_home:
            os.environ["JAVA_HOME"] = inferred_home
            # 把对应 bin 放在 PATH 前面，确保子进程也优先使用正确 java。
            current_path = os.environ.get("PATH", "")
            bin_path = str(Path(inferred_home) / "bin")
            if not current_path.lower().startswith(bin_path.lower()):
                os.environ["PATH"] = bin_path + os.pathsep + current_path
            return

    # 情况3：都不满足，给出可执行的修复提示。
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
    """从 topic 字典中提取可用的查询文本。

    由于不同 benchmark 的 topic 字段命名可能不同，这里按优先级尝试：
    title -> query -> question -> description -> text
    若都不存在，则退化为“第一个非空字符串字段”。
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
    """将原始 qrels 转为“仅保留正相关文档”的结构。

    输入可能是：
      {qid: {docid: judgement, ...}, ...}
    输出固定为：
      {qid(str): {docid(str), ...}}

    仅保留 judgement > 0 的 docid。
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
    """qid 排序键。

    - 若 qid 是纯数字字符串，则按数值排序（避免 1,10,2 这种顺序）。
    - 否则按字符串字典序排序。
    """

    return (0, int(qid)) if qid.isdigit() else (1, qid)


def load_topics_qrels(topic_name: str) -> tuple[list[QueryExample], dict[str, set[str]]]:
    """通过 Pyserini 加载 topics 与 qrels。

    返回：
    - queries: 按 qid 排序后的 QueryExample 列表，只保留“有正相关 qrels”的 query
    - qrels:   {qid -> 正相关 docid 集合}

    说明：
    训练阶段如果包含没有正样本的 qid，MRR 奖励会恒为 0，容易引入噪声。
    因此这里直接过滤掉这类样本。
    """

    # 在导入 pyserini 前先确保 Java 版本与 JAVA_HOME 配置可用。
    ensure_java_runtime(min_major=21)

    from pyserini.search import get_qrels, get_topics

    topics = get_topics(topic_name)
    qrels = _to_positive_qrels(get_qrels(topic_name))

    queries: list[QueryExample] = []
    for qid, payload in topics.items():
        qid_str = str(qid)
        # 仅保留存在正相关文档的 qid，否则该样本奖励恒为 0。
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
    """按给定比例切分训练集/验证集（可复现）。

    注意：
    - train_ratio 必须在 (0, 1)。
    - 即使样本数很少，也会保证 train/val 都至少有 1 条（前提是总数 >= 2）。
    """

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
    """可选截断，用于快速调试。"""

    if limit is None:
        return list(queries)
    return list(queries)[: max(0, limit)]
