from __future__ import annotations

"""Prompt sweep evaluation on the same random query sample."""

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import dataclass
import json
import random
import re
from statistics import fmean
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

# Support direct execution: `python prompt_eval/run_prompt_eval.py`
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from app_config import AppConfig, get_default_config
from core.model_wrapper import ModelWrapper
from core.reward_func import (
    RewardBreakdown,
    Rewarder,
    clean_rewritten_query,
    compute_format_penalty,
    compute_lexical_overlap,
    stabilize_generated_rewrite as shared_stabilize_generated_rewrite,
)
from data.loader import QueryExample, load_topics_qrels, split_queries
from eval_compare import evaluate_original

try:
    # Works when imported as a package module.
    from prompt_eval.prompt_bank import PromptSpec, build_prompt_specs
except ImportError:  # pragma: no cover - fallback for direct script execution.
    from prompt_bank import PromptSpec, build_prompt_specs


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
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
POLLUTION_RE = re.compile(
    r"(?:<think|thinking process|analysis:|assistant:|search query:|rewritten query:)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RewriteRecord:
    """One generated rewrite after cleaning and guardrail fallback."""

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
    prompt_spec: PromptSpec,
    reward_cfg,
) -> RewriteRecord:
    """Use the shared rewrite guardrail so prompt-eval matches train/eval behavior."""

    shared = shared_stabilize_generated_rewrite(
        raw_query,
        source_query=source_query,
        guardrail_cfg=prompt_spec,
        reward_cfg=reward_cfg,
    )
    return RewriteRecord(
        raw_query=shared.raw_query,
        cleaned_query=shared.cleaned_query,
        final_query=shared.final_query,
        fallback_to_original=shared.fallback_to_original,
        fallback_reasons=shared.fallback_reasons,
        raw_contains_think=shared.raw_contains_think,
        raw_contains_label=shared.raw_contains_label,
        raw_multiline=shared.raw_multiline,
        raw_format_penalty=shared.raw_format_penalty,
        final_overlap=shared.final_overlap,
        final_term_count=shared.final_term_count,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI args for prompt evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate multiple zero-shot prompts on a fixed random query sample."
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--topic-name", type=str, default=None)
    parser.add_argument("--prebuilt-index", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Used for split + random sampling.")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--disable-4bit",
        action="store_true",
        help="Disable 4-bit quantization when loading the 0.8B model.",
    )
    parser.add_argument("--search-threads", type=int, default=None)
    parser.add_argument("--reward-mrr-k", type=int, default=None)
    parser.add_argument("--reward-recall-k", type=int, default=None)
    parser.add_argument("--reward-recall-dense-k", type=int, default=None)
    parser.add_argument("--reward-w-mrr", type=float, default=None)
    parser.add_argument("--reward-w-recall", type=float, default=None)
    parser.add_argument("--reward-w-recall-dense", type=float, default=None)
    parser.add_argument("--reward-w-term-preserve", type=float, default=None)
    parser.add_argument("--reward-w-length-score", type=float, default=None)
    parser.add_argument("--reward-w-clean-format", type=float, default=None)
    parser.add_argument("--reward-w-bad-format", type=float, default=None)
    parser.add_argument("--reward-w-unsafe-copy", type=float, default=None)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="Print progress every N queries per stage.",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=5,
        help="Batch size for model rewrite generation (increase to raise GPU utilization).",
    )
    parser.add_argument(
        "--prompt-model-parallel",
        type=int,
        default=5,
        help="Number of model instances to load for prompt-level parallel evaluation.",
    )
    parser.add_argument(
        "--sample-print",
        type=int,
        default=3,
        help="How many sampled examples to keep in report per prompt.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Optional custom path for JSON report output.",
    )
    parser.add_argument(
        "--prompt-ids",
        type=str,
        default=None,
        help="Comma-separated prompt ids to evaluate (for slow incremental runs).",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Evaluate only first N prompts after filtering.",
    )
    parser.add_argument(
        "--history-path",
        type=str,
        default=None,
        help="Path to history JSON file. Default: <report_dir>/prompt_eval_history.json",
    )
    parser.add_argument(
        "--no-save-each-prompt",
        dest="save_each_prompt",
        action="store_false",
        help="Disable checkpoint save after each prompt.",
    )
    parser.add_argument(
        "--no-append-history",
        dest="append_history",
        action="store_false",
        help="Disable appending summary records to history JSON.",
    )
    parser.set_defaults(save_each_prompt=True, append_history=True)
    return parser.parse_args()


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Apply CLI overrides to runtime config."""

    config.model.model_name = args.model_name
    config.data.seed = args.seed
    if args.disable_4bit:
        config.model.load_in_4bit = False
    if args.topic_name is not None:
        config.data.topic_name = args.topic_name
    if args.prebuilt_index is not None:
        config.data.prebuilt_index = args.prebuilt_index
    if args.train_ratio is not None:
        config.data.train_ratio = args.train_ratio
    if args.max_new_tokens is not None:
        config.train.max_new_tokens = max(1, args.max_new_tokens)
    if args.search_threads is not None:
        config.reward.search_threads = max(1, args.search_threads)
    if args.reward_mrr_k is not None:
        config.reward.mrr_k = max(1, args.reward_mrr_k)
    if args.reward_recall_k is not None:
        config.reward.recall_k = max(1, args.reward_recall_k)
    if args.reward_recall_dense_k is not None:
        config.reward.recall_dense_k = max(1, args.reward_recall_dense_k)
    if args.reward_w_mrr is not None:
        config.reward.w_mrr = args.reward_w_mrr
    if args.reward_w_recall is not None:
        config.reward.w_recall = args.reward_w_recall
    if args.reward_w_recall_dense is not None:
        config.reward.w_recall_dense = args.reward_w_recall_dense
    if args.reward_w_term_preserve is not None:
        config.reward.w_term_preserve = args.reward_w_term_preserve
    if args.reward_w_length_score is not None:
        config.reward.w_length_score = args.reward_w_length_score
    if args.reward_w_clean_format is not None:
        config.reward.w_clean_format = args.reward_w_clean_format
    if args.reward_w_bad_format is not None:
        config.reward.w_bad_format = args.reward_w_bad_format
    if args.reward_w_unsafe_copy is not None:
        config.reward.w_unsafe_copy = args.reward_w_unsafe_copy
    return config


def resolve_report_path(raw_path: str | None, *, seed: int, sample_size: int) -> Path:
    """Resolve output report path."""

    if raw_path:
        return Path(raw_path)
    return Path("prompt_eval") / "artifacts" / f"seed{seed}_n{sample_size}" / "prompt_eval_report.json"


def resolve_history_path(raw_path: str | None, *, report_path: Path) -> Path:
    """Resolve history path."""

    if raw_path:
        return Path(raw_path)
    return report_path.parent / "prompt_eval_history.json"


def sample_eval_queries(
    queries: Sequence[QueryExample],
    *,
    sample_size: int,
    seed: int,
) -> list[QueryExample]:
    """Sample queries with deterministic randomness."""

    if sample_size < 1:
        raise ValueError(f"sample_size must be >= 1, got {sample_size}")

    pool = list(queries)
    if not pool:
        return []

    rng = random.Random(seed)
    if sample_size >= len(pool):
        sampled = pool.copy()
        rng.shuffle(sampled)
        return sampled
    return rng.sample(pool, sample_size)


def compute_metric_deltas(
    baseline_metrics: dict[str, float],
    prompt_metrics: dict[str, float],
) -> dict[str, float]:
    """Compute aggregate metric deltas against Original baseline."""

    return {
        "delta_mrr": float(prompt_metrics.get("mrr", 0.0) - baseline_metrics.get("mrr", 0.0)),
        "delta_recall": float(prompt_metrics.get("recall", 0.0) - baseline_metrics.get("recall", 0.0)),
        "delta_reward_mean": float(
            prompt_metrics.get("reward_mean", 0.0) - baseline_metrics.get("reward_mean", 0.0)
        ),
    }


def compute_win_tie_loss(
    baseline_by_qid: dict[str, RewardBreakdown],
    prompt_by_qid: dict[str, tuple[str, RewardBreakdown]],
    *,
    tol: float = 1e-12,
) -> dict[str, float]:
    """Compute query-level win/tie/loss by MRR versus baseline."""

    win = 0
    tie = 0
    loss = 0

    for qid, baseline_score in baseline_by_qid.items():
        prompt_item = prompt_by_qid.get(qid)
        if prompt_item is None:
            continue
        prompt_score = prompt_item[1]
        delta = float(prompt_score.mrr - baseline_score.mrr)
        if delta > tol:
            win += 1
        elif delta < -tol:
            loss += 1
        else:
            tie += 1

    total = win + tie + loss
    return {
        "win": win,
        "tie": tie,
        "loss": loss,
        "total": total,
        "win_rate": (win / total) if total else 0.0,
    }


def _prompt_rank_key(result: dict[str, Any]) -> tuple[float, float, float, str]:
    metrics = result.get("metrics", {})
    return (
        -float(metrics.get("mrr", 0.0)),
        -float(metrics.get("recall", 0.0)),
        -float(metrics.get("reward_mean", 0.0)),
        str(result.get("prompt_id", "")),
    )


def rank_prompt_results(prompt_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build leaderboard sorted by mrr, recall, reward_mean."""

    ordered = sorted(prompt_results, key=_prompt_rank_key)
    leaderboard: list[dict[str, Any]] = []
    for idx, item in enumerate(ordered, start=1):
        metrics = item["metrics"]
        deltas = item["deltas"]
        wtl = item["win_tie_loss"]
        leaderboard.append(
            {
                "rank": idx,
                "prompt_id": item["prompt_id"],
                "prompt_name": item["prompt_name"],
                "mrr": metrics["mrr"],
                "recall": metrics["recall"],
                "reward_mean": metrics["reward_mean"],
                "delta_mrr": deltas["delta_mrr"],
                "delta_recall": deltas["delta_recall"],
                "delta_reward_mean": deltas["delta_reward_mean"],
                "win": wtl["win"],
                "tie": wtl["tie"],
                "loss": wtl["loss"],
            }
        )
    return leaderboard


def select_best_prompt(prompt_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select best prompt by mrr, recall, reward_mean."""

    if not prompt_results:
        return None
    return sorted(prompt_results, key=_prompt_rank_key)[0]


def filter_prompt_specs(
    prompt_specs: Sequence[PromptSpec],
    *,
    prompt_ids_csv: str | None,
    max_prompts: int | None,
) -> list[PromptSpec]:
    """Filter prompts by id and count for incremental testing."""

    selected = list(prompt_specs)
    if prompt_ids_csv:
        requested_ids = [item.strip() for item in prompt_ids_csv.split(",") if item.strip()]
        id_to_spec = {item.id: item for item in prompt_specs}
        missing = [item for item in requested_ids if item not in id_to_spec]
        if missing:
            raise ValueError(f"Unknown prompt ids: {', '.join(missing)}")
        selected = [id_to_spec[item] for item in requested_ids]

    if max_prompts is not None:
        if max_prompts < 1:
            raise ValueError(f"max_prompts must be >= 1, got {max_prompts}")
        selected = selected[:max_prompts]

    return selected


def _build_sample_preview(
    sampled_queries: Sequence[QueryExample],
    per_qid: dict[str, tuple[str, RewardBreakdown]],
    rewrite_records: dict[str, RewriteRecord],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for query in sampled_queries[: max(0, limit)]:
        rewritten, score = per_qid[query.qid]
        record = rewrite_records[query.qid]
        preview.append(
            {
                "qid": query.qid,
                "original_query": query.text,
                "raw_generation": record.raw_query,
                "cleaned_rewrite": record.cleaned_query,
                "rewritten_query": rewritten,
                "fallback_to_original": record.fallback_to_original,
                "fallback_reasons": list(record.fallback_reasons),
                "mrr": score.mrr,
                "recall": score.recall,
            }
        )
    return preview


def _summarize_rewrite_records(
    sampled_queries: Sequence[QueryExample],
    rewrite_records: dict[str, RewriteRecord],
) -> dict[str, Any]:
    records: list[RewriteRecord] = [rewrite_records[item.qid] for item in sampled_queries if item.qid in rewrite_records]
    if not records:
        return {
            "changed_rate": 0.0,
            "effective_rewrite_rate": 0.0,
            "fallback_rate": 0.0,
            "raw_format_fail_rate": 0.0,
            "raw_multiline_rate": 0.0,
            "raw_think_rate": 0.0,
            "raw_label_rate": 0.0,
            "mean_final_overlap": 0.0,
            "mean_final_terms": 0.0,
            "fallback_reason_counts": {},
        }

    source_by_qid = {item.qid: _normalize_query_text(item.text) for item in sampled_queries}
    fallback_counter = Counter(reason for record in records for reason in record.fallback_reasons)
    changed_count = sum(1 for item in sampled_queries if rewrite_records[item.qid].final_query != source_by_qid[item.qid])
    effective_rewrite_count = sum(
        1
        for item in sampled_queries
        if rewrite_records[item.qid].final_query != source_by_qid[item.qid]
        and not rewrite_records[item.qid].fallback_to_original
    )
    total = len(records)
    return {
        "changed_rate": changed_count / total,
        "effective_rewrite_rate": effective_rewrite_count / total,
        "fallback_rate": sum(1 for record in records if record.fallback_to_original) / total,
        "raw_format_fail_rate": sum(1 for record in records if record.raw_format_penalty > 0.0) / total,
        "raw_multiline_rate": sum(1 for record in records if record.raw_multiline) / total,
        "raw_think_rate": sum(1 for record in records if record.raw_contains_think) / total,
        "raw_label_rate": sum(1 for record in records if record.raw_contains_label) / total,
        "mean_final_overlap": fmean(record.final_overlap for record in records),
        "mean_final_terms": fmean(record.final_term_count for record in records),
        "fallback_reason_counts": dict(sorted(fallback_counter.items())),
    }


def _generate_rewrites_for_prompt(
    model: ModelWrapper,
    sampled_queries: Sequence[QueryExample],
    prompt_spec: PromptSpec,
    *,
    query_batch_size: int,
    reward_cfg,
) -> dict[str, RewriteRecord]:
    model.prompt_cfg.system_prompt = prompt_spec.system_prompt
    model.prompt_cfg.template = prompt_spec.template

    rewrites_by_qid: dict[str, RewriteRecord] = {}
    batch_size = max(1, int(query_batch_size))
    total = len(sampled_queries)
    for start in range(0, total, batch_size):
        batch = list(sampled_queries[start : start + batch_size])
        raw_rewrites = model.generate_rewrite_batch(
            [query.text for query in batch],
            policy="actor",
            max_new_tokens=prompt_spec.max_new_tokens,
            temperature=prompt_spec.temperature,
            top_p=prompt_spec.top_p,
        )
        for query, raw in zip(batch, raw_rewrites):
            rewrites_by_qid[query.qid] = stabilize_generated_rewrite(
                raw,
                source_query=query.text,
                prompt_spec=prompt_spec,
                reward_cfg=reward_cfg,
            )
    return rewrites_by_qid


def _aggregate_prompt_metrics(
    sampled_queries: Sequence[QueryExample],
    rewrites_by_qid: dict[str, RewriteRecord],
    rewarder: Rewarder,
) -> tuple[dict[str, float], dict[str, tuple[str, RewardBreakdown]], dict[str, Any]]:
    per_qid: dict[str, tuple[str, RewardBreakdown]] = {}
    for query in sampled_queries:
        record = rewrites_by_qid.get(query.qid)
        if record is None:
            record = stabilize_generated_rewrite(
                "",
                source_query=query.text,
                prompt_spec=PromptSpec(
                    id="fallback",
                    name="Fallback",
                    system_prompt="",
                    template="{query}",
                    stop_on="\n",
                ),
                reward_cfg=rewarder.cfg,
            )
            rewrites_by_qid[query.qid] = record
        score = rewarder.score(query.qid, record.final_query, source_query=query.text)
        per_qid[query.qid] = (record.final_query, score)

    values = [item[1] for item in per_qid.values()]
    mrr_value = fmean(v.mrr for v in values) if values else 0.0
    recall_value = fmean(v.recall for v in values) if values else 0.0
    metrics = {
        "mrr": mrr_value,
        f"mrr@{rewarder.mrr_k}": mrr_value,
        "recall": recall_value,
        f"recall@{rewarder.recall_k}": recall_value,
        "reward_mean": fmean(v.total for v in values) if values else 0.0,
    }
    diagnostics = _summarize_rewrite_records(sampled_queries, rewrites_by_qid)
    return metrics, per_qid, diagnostics


def _is_cuda_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "cublas" in text)


def _load_model_pool(config: AppConfig, requested: int) -> list[ModelWrapper]:
    desired = max(1, int(requested))
    models: list[ModelWrapper] = []
    for idx in range(desired):
        try:
            model = ModelWrapper(
                model_cfg=config.model,
                prompt_cfg=deepcopy(config.prompt),
                train_mode=False,
                enable_lora=False,
                load_ref_model=False,
                strict_tokenizer_model_match=False,
            )
            models.append(model)
            print(f"[model-pool] loaded model {len(models)}/{desired}")
        except RuntimeError as exc:
            if torch.cuda.is_available() and _is_cuda_oom_error(exc):
                gc_text = "CUDA OOM while loading additional model"
                print(f"[warn] {gc_text} at slot {idx + 1}; active_models={len(models)}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if models:
                    break
            raise
    if not models:
        raise RuntimeError("Failed to load any model instance for prompt parallel evaluation.")
    return models


def _build_report_payload(
    *,
    config: AppConfig,
    args: argparse.Namespace,
    val_queries: Sequence[QueryExample],
    sampled_queries: Sequence[QueryExample],
    prompt_specs: Sequence[PromptSpec],
    prompt_results: Sequence[dict[str, Any]],
    baseline_metrics: dict[str, float],
    stage_status: str,
    model_pool_active: int,
) -> dict[str, Any]:
    leaderboard = rank_prompt_results(list(prompt_results))
    best_prompt = select_best_prompt(list(prompt_results))
    return {
        "run_config": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": config.model.model_name,
            "load_in_4bit": config.model.load_in_4bit,
            "topic_name": config.data.topic_name,
            "prebuilt_index": config.data.prebuilt_index,
            "train_ratio": config.data.train_ratio,
            "split_seed": config.data.seed,
            "sample_seed": args.seed,
            "sample_size_requested": args.sample_size,
            "max_new_tokens": config.train.max_new_tokens,
            "default_max_new_tokens": config.train.max_new_tokens,
            "prompt_decoding_source": "per_prompt",
            "progress_every": args.progress_every,
            "query_batch_size": max(1, args.query_batch_size),
            "prompt_model_parallel_requested": max(1, args.prompt_model_parallel),
            "prompt_model_parallel_active": max(1, model_pool_active),
            "num_prompts_total": len(prompt_specs),
            "num_prompts_completed": len(prompt_results),
            "prompt_ids_requested": args.prompt_ids,
            "max_prompts": args.max_prompts,
            "save_each_prompt": bool(args.save_each_prompt),
            "append_history": bool(args.append_history),
            "stage_status": stage_status,
            "rewrite_postprocess": {
                "cleaner": "clean_rewritten_query",
                "fallback_to_original_on_empty": True,
                "locked_terms_scored_softly": True,
            },
            "reward": {
                "mrr_k": config.reward.mrr_k,
                "recall_k": config.reward.recall_k,
                "recall_dense_k": config.reward.recall_dense_k,
                "search_threads": config.reward.search_threads,
                "w_mrr": config.reward.w_mrr,
                "w_recall": config.reward.w_recall,
                "w_recall_dense": config.reward.w_recall_dense,
                "w_term_preserve": config.reward.w_term_preserve,
                "w_length_score": config.reward.w_length_score,
                "w_clean_format": config.reward.w_clean_format,
                "w_bad_format": config.reward.w_bad_format,
                "w_unsafe_copy": config.reward.w_unsafe_copy,
            },
        },
        "sample_info": {
            "available_val_queries": len(val_queries),
            "sample_size": len(sampled_queries),
            "sampled_qids": [item.qid for item in sampled_queries],
        },
        "baseline_original": baseline_metrics,
        "prompt_results": list(prompt_results),
        "leaderboard": leaderboard,
        "best_prompt": best_prompt,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_history_entry(history_path: Path, report_path: Path, final_report: dict[str, Any]) -> None:
    """Append final run summary to history JSON."""

    run_cfg = final_report.get("run_config", {})
    baseline = final_report.get("baseline_original", {})
    best = final_report.get("best_prompt") or {}
    best_metrics = best.get("metrics", {}) if isinstance(best, dict) else {}
    best_deltas = best.get("deltas", {}) if isinstance(best, dict) else {}
    history_entry = {
        "generated_at_utc": run_cfg.get("generated_at_utc"),
        "report_path": str(report_path),
        "model_name": run_cfg.get("model_name"),
        "topic_name": run_cfg.get("topic_name"),
        "sample_seed": run_cfg.get("sample_seed"),
        "sample_size": final_report.get("sample_info", {}).get("sample_size"),
        "num_prompts_total": run_cfg.get("num_prompts_total"),
        "num_prompts_completed": run_cfg.get("num_prompts_completed"),
        "baseline_mrr": baseline.get("mrr"),
        "best_prompt_id": best.get("prompt_id"),
        "best_prompt_name": best.get("prompt_name"),
        "best_mrr": best_metrics.get("mrr"),
        "best_delta_mrr": best_deltas.get("delta_mrr"),
        "best_beats_baseline": bool(best_deltas.get("delta_mrr", 0.0) > 0.0),
    }

    history: list[dict[str, Any]]
    if history_path.exists():
        try:
            old = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(old, list):
                history = old
            elif isinstance(old, dict) and isinstance(old.get("runs"), list):
                history = old["runs"]
            else:
                history = []
        except Exception:
            history = []
    else:
        history = []

    history.append(history_entry)
    write_json(history_path, {"runs": history})


def main() -> int:
    args = parse_args()
    config = apply_overrides(get_default_config(), args)
    report_path = resolve_report_path(args.report_path, seed=args.seed, sample_size=args.sample_size)
    history_path = resolve_history_path(args.history_path, report_path=report_path)

    print(
        "[config] "
        f"model={config.model.model_name}, "
        f"index={config.data.prebuilt_index}, "
        f"topic={config.data.topic_name}, "
        f"sample_size={args.sample_size}, "
        f"query_batch_size={max(1, args.query_batch_size)}, "
        f"prompt_model_parallel={max(1, args.prompt_model_parallel)}, "
        f"seed={args.seed}, "
        f"mrr_k={config.reward.mrr_k}, "
        f"recall_k={config.reward.recall_k}, "
        f"load_in_4bit={config.model.load_in_4bit}, "
        f"save_each_prompt={args.save_each_prompt}, "
        f"append_history={args.append_history}"
    )

    all_queries, qrels = load_topics_qrels(config.data.topic_name)
    _, val_queries = split_queries(all_queries, train_ratio=config.data.train_ratio, seed=config.data.seed)
    sampled_queries = sample_eval_queries(val_queries, sample_size=args.sample_size, seed=args.seed)
    print(
        "[data] "
        f"all_queries={len(all_queries)}, "
        f"val_queries={len(val_queries)}, "
        f"sampled_queries={len(sampled_queries)}"
    )

    rewarder = Rewarder(
        qrels=qrels,
        prebuilt_index=config.data.prebuilt_index,
        reward_cfg=config.reward,
    )

    print("[stage] evaluating original baseline...")
    baseline_metrics, baseline_by_qid = evaluate_original(
        sampled_queries,
        rewarder,
        progress_every=args.progress_every,
    )

    prompt_specs = build_prompt_specs(
        default_system_prompt=config.prompt.system_prompt,
        default_template=config.prompt.template,
    )
    prompt_specs = filter_prompt_specs(
        prompt_specs,
        prompt_ids_csv=args.prompt_ids,
        max_prompts=args.max_prompts,
    )
    print(f"[prompts] total={len(prompt_specs)}")

    print("[stage] loading zero-shot model pool (no adapter)...")
    model_pool = _load_model_pool(config, args.prompt_model_parallel)
    model_pool_active = len(model_pool)
    print(
        "[model-pool] "
        f"requested={max(1, args.prompt_model_parallel)}, "
        f"active={model_pool_active}"
    )

    prompt_results_by_id: dict[str, dict[str, Any]] = {}

    def _ordered_prompt_results() -> list[dict[str, Any]]:
        return [prompt_results_by_id[item.id] for item in prompt_specs if item.id in prompt_results_by_id]

    if args.save_each_prompt:
        running_report = _build_report_payload(
            config=config,
            args=args,
            val_queries=val_queries,
            sampled_queries=sampled_queries,
            prompt_specs=prompt_specs,
            prompt_results=_ordered_prompt_results(),
            baseline_metrics=baseline_metrics,
            stage_status="running",
            model_pool_active=model_pool_active,
        )
        write_json(report_path, running_report)
        print(f"[checkpoint] initialized report: {report_path}")

    prompt_total = len(prompt_specs)
    next_prompt_idx = 0
    active_jobs: dict[Future[dict[str, RewriteRecord]], tuple[int, int, PromptSpec]] = {}

    with ThreadPoolExecutor(max_workers=model_pool_active) as executor:
        for worker_id in range(model_pool_active):
            if next_prompt_idx >= prompt_total:
                break
            prompt_spec = prompt_specs[next_prompt_idx]
            print(
                f"[stage] worker {worker_id + 1}/{model_pool_active} "
                f"evaluating prompt {next_prompt_idx + 1}/{prompt_total} -> {prompt_spec.id}"
            )
            future = executor.submit(
                _generate_rewrites_for_prompt,
                model_pool[worker_id],
                sampled_queries,
                prompt_spec,
                query_batch_size=args.query_batch_size,
                reward_cfg=config.reward,
            )
            active_jobs[future] = (worker_id, next_prompt_idx, prompt_spec)
            next_prompt_idx += 1

        while active_jobs:
            done, _ = wait(active_jobs.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                worker_id, prompt_idx, prompt_spec = active_jobs.pop(future)
                try:
                    rewrites_by_qid = future.result()
                except Exception as exc:  # pragma: no cover - runtime safety
                    raise RuntimeError(
                        f"Prompt worker {worker_id + 1} failed on prompt {prompt_spec.id}"
                    ) from exc

                prompt_metrics, prompt_by_qid, diagnostics = _aggregate_prompt_metrics(
                    sampled_queries,
                    rewrites_by_qid,
                    rewarder,
                )
                deltas = compute_metric_deltas(baseline_metrics, prompt_metrics)
                win_tie_loss = compute_win_tie_loss(baseline_by_qid, prompt_by_qid)
                prompt_result = {
                    "prompt_id": prompt_spec.id,
                    "prompt_name": prompt_spec.name,
                    "tags": list(prompt_spec.tags),
                    "prompt": {
                        "system_prompt": prompt_spec.system_prompt,
                        "template": prompt_spec.template,
                    },
                    "decoding": {
                        "temperature": prompt_spec.temperature,
                        "max_new_tokens": prompt_spec.max_new_tokens,
                        "top_p": prompt_spec.top_p,
                        "stop_on": prompt_spec.stop_on,
                        "enforce_single_line": prompt_spec.enforce_single_line,
                    },
                    "guardrail": {
                        "min_terms": prompt_spec.min_terms,
                        "max_terms": prompt_spec.max_terms,
                        "fallback_mode": prompt_spec.fallback_mode,
                    },
                    "metrics": prompt_metrics,
                    "deltas": deltas,
                    "win_tie_loss": win_tie_loss,
                    "diagnostics": diagnostics,
                    "sample_rewrites": _build_sample_preview(
                        sampled_queries,
                        prompt_by_qid,
                        rewrites_by_qid,
                        limit=args.sample_print,
                    ),
                }
                prompt_results_by_id[prompt_spec.id] = prompt_result
                prompt_results = _ordered_prompt_results()
                print(
                    f"[result] {prompt_spec.id} (worker {worker_id + 1}): "
                    f"mrr={prompt_metrics['mrr']:.4f} "
                    f"(delta {deltas['delta_mrr']:+.4f}), "
                    f"recall={prompt_metrics['recall']:.4f}, "
                    f"reward_mean={prompt_metrics['reward_mean']:.4f}, "
                    f"fallback={diagnostics['fallback_rate']:.2%}, "
                    f"rewrite={diagnostics['effective_rewrite_rate']:.2%}"
                )

                if args.save_each_prompt:
                    running_report = _build_report_payload(
                        config=config,
                        args=args,
                        val_queries=val_queries,
                        sampled_queries=sampled_queries,
                        prompt_specs=prompt_specs,
                        prompt_results=prompt_results,
                        baseline_metrics=baseline_metrics,
                        stage_status="running",
                        model_pool_active=model_pool_active,
                    )
                    write_json(report_path, running_report)
                    print(
                        f"[checkpoint] saved {len(prompt_results)}/{len(prompt_specs)} prompts "
                        f"to {report_path}"
                    )

                if next_prompt_idx < prompt_total:
                    next_prompt = prompt_specs[next_prompt_idx]
                    print(
                        f"[stage] worker {worker_id + 1}/{model_pool_active} "
                        f"evaluating prompt {next_prompt_idx + 1}/{prompt_total} -> {next_prompt.id}"
                    )
                    next_future = executor.submit(
                        _generate_rewrites_for_prompt,
                        model_pool[worker_id],
                        sampled_queries,
                        next_prompt,
                        query_batch_size=args.query_batch_size,
                        reward_cfg=config.reward,
                    )
                    active_jobs[next_future] = (worker_id, next_prompt_idx, next_prompt)
                    next_prompt_idx += 1

    prompt_results = _ordered_prompt_results()

    leaderboard = rank_prompt_results(prompt_results)
    best_prompt = select_best_prompt(prompt_results)

    print("\n=== Prompt Leaderboard (mrr > recall > reward_mean) ===")
    for row in leaderboard:
        print(
            f"{row['rank']:>2}. {row['prompt_id']} | "
            f"mrr={row['mrr']:.4f} ({row['delta_mrr']:+.4f}) | "
            f"recall={row['recall']:.4f} ({row['delta_recall']:+.4f}) | "
            f"reward_mean={row['reward_mean']:.4f} ({row['delta_reward_mean']:+.4f}) | "
            f"W/T/L={row['win']}/{row['tie']}/{row['loss']}"
        )

    if best_prompt is None:
        print("[result] no prompt candidates were evaluated.")
    elif best_prompt["deltas"]["delta_mrr"] > 0.0:
        print(
            "[result] best prompt beats baseline by MRR: "
            f"{best_prompt['prompt_id']} ({best_prompt['deltas']['delta_mrr']:+.4f})"
        )
    else:
        print(
            "[result] no prompt beats baseline MRR; best candidate is "
            f"{best_prompt['prompt_id']} ({best_prompt['deltas']['delta_mrr']:+.4f})."
        )

    report = _build_report_payload(
        config=config,
        args=args,
        val_queries=val_queries,
        sampled_queries=sampled_queries,
        prompt_specs=prompt_specs,
        prompt_results=prompt_results,
        baseline_metrics=baseline_metrics,
        stage_status="completed",
        model_pool_active=model_pool_active,
    )
    write_json(report_path, report)
    print(f"\n[report] {report_path}")
    if args.append_history:
        append_history_entry(history_path, report_path, report)
        print(f"[history] appended run summary: {history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
