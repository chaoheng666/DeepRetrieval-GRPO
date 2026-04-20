from __future__ import annotations

"""三路对比评估脚本。

在同一验证集上比较：
1. Original（原始 query）
2. Zero-shot（基础模型重写）
3. RL（基础模型 + LoRA adapter 重写）
"""

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from app_config import AppConfig, apply_reward_mode_prompt_defaults, get_default_config
from core.model_wrapper import ModelWrapper
from core.reward_func import RewardBreakdown, Rewarder, stabilize_generated_rewrite, summarize_reward_breakdowns
from data.loader import QueryExample, load_topics_qrels, maybe_limit, split_queries


def parse_args() -> argparse.Namespace:
    """解析评估命令行参数。"""

    parser = argparse.ArgumentParser(description="Compare Original vs Zero-shot vs RL-rewritten query MRR@k.")
    parser.add_argument("--rl-adapter-path", type=str, required=True, help="Path to trained LoRA adapter.")
    parser.add_argument("--model-name", type=str, default=None, help="Override base model name for evaluation.")
    parser.add_argument(
        "--disable-4bit",
        action="store_true",
        help="Disable 4-bit quantization for evaluation model loading.",
    )
    parser.add_argument(
        "--strict-tokenizer-model-match",
        action="store_true",
        help="Fail fast if tokenizer/model (or adapter base model) mismatch is detected.",
    )
    parser.add_argument("--topic-name", type=str, default=None)
    parser.add_argument("--prebuilt-index", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-eval-queries", type=int, default=None)
    parser.add_argument("--search-threads", type=int, default=None, help="Pyserini batch_search thread count.")
    parser.add_argument("--reward-mrr-k", type=int, default=None)
    parser.add_argument("--reward-recall-k", type=int, default=None)
    parser.add_argument("--reward-recall-dense-k", type=int, default=None)
    parser.add_argument("--reward-mode", type=str, choices=("legacy", "top20_delta"), default=None)
    parser.add_argument("--reward-w-mrr", type=float, default=None)
    parser.add_argument("--reward-w-recall", type=float, default=None)
    parser.add_argument("--reward-w-recall-dense", type=float, default=None)
    parser.add_argument("--reward-w-rank-bonus", type=float, default=None)
    parser.add_argument("--reward-w-term-preserve", type=float, default=None)
    parser.add_argument("--reward-w-length-score", type=float, default=None)
    parser.add_argument("--reward-w-clean-format", type=float, default=None)
    parser.add_argument("--reward-w-bad-format", type=float, default=None)
    parser.add_argument("--reward-w-unsafe-copy", type=float, default=None)
    parser.add_argument("--reward-w-overedit", type=float, default=None)
    parser.add_argument("--overedit-tau", type=float, default=None)
    parser.add_argument("--length-score-min-terms", type=int, default=None)
    parser.add_argument("--length-score-ideal-min-terms", type=int, default=None)
    parser.add_argument("--length-score-ideal-max-terms", type=int, default=None)
    parser.add_argument("--length-score-max-terms", type=int, default=None)
    parser.add_argument("--format-max-tokens", type=int, default=None)
    parser.add_argument("--format-min-english-ratio", type=float, default=None)
    parser.add_argument("--format-max-unreadable-ratio", type=float, default=None)
    parser.add_argument("--bad-format-cap", type=float, default=None)
    parser.add_argument("--sample-print", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=20, help="Print progress every N queries per stage.")
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=5,
        help="Batch size for model rewrite generation (increase to raise GPU utilization).",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="train_and_eval_data_model/artifacts_default_eval/eval_compare_report.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--low-mem-mode",
        action="store_true",
        help="Use low-memory evaluation preset (0.5B model + slim index + smaller eval set).",
    )
    return parser.parse_args()


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """应用评估配置覆盖参数。"""

    if args.low_mem_mode:
        config.model.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        config.data.prebuilt_index = "msmarco-v1-passage-slim"
        config.data.max_val_queries = 100
        config.prompt.max_new_tokens = min(config.prompt.max_new_tokens, 16)
        config.reward.recall_k = 50

    if args.model_name is not None:
        config.model.model_name = args.model_name
    if args.disable_4bit:
        config.model.load_in_4bit = False
    if args.topic_name is not None:
        config.data.topic_name = args.topic_name
    if args.prebuilt_index is not None:
        config.data.prebuilt_index = args.prebuilt_index
    if args.train_ratio is not None:
        config.data.train_ratio = args.train_ratio
    if args.seed is not None:
        config.data.seed = args.seed
    if args.max_eval_queries is not None:
        config.data.max_val_queries = args.max_eval_queries
    if args.search_threads is not None:
        config.reward.search_threads = max(1, args.search_threads)
    if args.reward_mrr_k is not None:
        config.reward.mrr_k = max(1, args.reward_mrr_k)
    if args.reward_recall_k is not None:
        config.reward.recall_k = max(1, args.reward_recall_k)
    if args.reward_recall_dense_k is not None:
        config.reward.recall_dense_k = max(1, args.reward_recall_dense_k)
    if args.reward_mode is not None:
        config.reward.reward_mode = args.reward_mode
    if args.reward_w_mrr is not None:
        config.reward.w_mrr = args.reward_w_mrr
    if args.reward_w_recall is not None:
        config.reward.w_recall = args.reward_w_recall
    if args.reward_w_recall_dense is not None:
        config.reward.w_recall_dense = args.reward_w_recall_dense
    if args.reward_w_rank_bonus is not None:
        config.reward.w_rank_bonus = args.reward_w_rank_bonus
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
    if args.reward_w_overedit is not None:
        config.reward.w_overedit = args.reward_w_overedit
    if args.overedit_tau is not None:
        config.reward.overedit_tau = args.overedit_tau
    if args.length_score_min_terms is not None:
        config.reward.length_score_min_terms = max(0, args.length_score_min_terms)
    if args.length_score_ideal_min_terms is not None:
        config.reward.length_score_ideal_min_terms = max(0, args.length_score_ideal_min_terms)
    if args.length_score_ideal_max_terms is not None:
        config.reward.length_score_ideal_max_terms = max(0, args.length_score_ideal_max_terms)
    if args.length_score_max_terms is not None:
        config.reward.length_score_max_terms = max(0, args.length_score_max_terms)
    if args.format_max_tokens is not None:
        config.reward.format_max_tokens = max(1, args.format_max_tokens)
    if args.format_min_english_ratio is not None:
        config.reward.format_min_english_ratio = args.format_min_english_ratio
    if args.format_max_unreadable_ratio is not None:
        config.reward.format_max_unreadable_ratio = args.format_max_unreadable_ratio
    if args.bad_format_cap is not None:
        config.reward.bad_format_cap = max(0.0, args.bad_format_cap)
    if args.max_new_tokens is not None:
        config.prompt.max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        config.prompt.temperature = args.temperature
    if args.top_p is not None:
        config.prompt.top_p = args.top_p
    return config


def validate_adapter_path(adapter_path: str) -> dict:
    """校验 adapter 路径并读取 adapter_config.json。"""

    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter path does not exist: {adapter_dir}. "
            "Use the correct folder like train_and_eval_data_model/artifacts_lowmem_train/checkpoints/best."
        )
    if not adapter_dir.is_dir():
        raise NotADirectoryError(f"Adapter path is not a directory: {adapter_dir}")

    adapter_cfg_path = adapter_dir / "adapter_config.json"
    if not adapter_cfg_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in: {adapter_dir}")
    with adapter_cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def reward_breakdown_to_report_dict(score: RewardBreakdown) -> dict[str, Any]:
    return {
        "total": getattr(score, "total", 0.0),
        "mrr": getattr(score, "mrr", 0.0),
        "recall": getattr(score, "recall", 0.0),
        "recall_dense": getattr(score, "recall_dense", 0.0),
        "rank_bonus": getattr(score, "rank_bonus", 0.0),
        "orig_mrr": getattr(score, "orig_mrr", 0.0),
        "orig_recall": getattr(score, "orig_recall", 0.0),
        "orig_recall_aux": getattr(score, "orig_recall_aux", 0.0),
        "orig_rank_bonus": getattr(score, "orig_rank_bonus", 0.0),
        "delta_mrr": getattr(score, "delta_mrr", 0.0),
        "delta_recall": getattr(score, "delta_recall", 0.0),
        "delta_recall_aux": getattr(score, "delta_recall_aux", 0.0),
        "delta_rank_bonus": getattr(score, "delta_rank_bonus", 0.0),
        "main_reward": getattr(score, "main_reward", 0.0),
        "overedit_penalty": getattr(score, "overedit_penalty", 0.0),
        "bad_format_penalty": getattr(score, "bad_format_penalty", 0.0),
        "unsafe_copy_penalty": getattr(score, "unsafe_copy_penalty", 0.0),
        "hit_rank": getattr(score, "hit_rank", None),
        "retrieved_relevant_count": getattr(score, "retrieved_relevant_count", 0),
        "relevant_total": getattr(score, "relevant_total", 0),
    }


def build_per_qid_rows(
    queries: Sequence[QueryExample],
    original_by_qid: dict[str, RewardBreakdown],
    zero_by_qid: dict[str, tuple[str, RewardBreakdown]],
    rl_by_qid: dict[str, tuple[str, RewardBreakdown]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query in queries:
        qid = query.qid
        zero_rewrite, zero_score = zero_by_qid[qid]
        rl_rewrite, rl_score = rl_by_qid[qid]
        original_score = original_by_qid[qid]
        rows.append(
            {
                "qid": qid,
                "original_query": query.text,
                "zero_rewrite": zero_rewrite,
                "rl_rewrite": rl_rewrite,
                "original": reward_breakdown_to_report_dict(original_score),
                "zero_shot": reward_breakdown_to_report_dict(zero_score),
                "rl": reward_breakdown_to_report_dict(rl_score),
            }
        )
    return rows


def select_sample_cases(per_qid_rows: Sequence[dict[str, Any]], sample_count: int) -> list[dict[str, Any]]:
    return list(per_qid_rows[: max(0, int(sample_count))])


def evaluate_original(
    queries: Sequence[QueryExample],
    rewarder: Rewarder,
    *,
    progress_every: int,
) -> tuple[dict[str, float], dict[str, RewardBreakdown]]:
    """评估原始 query 基线。"""

    per_qid: dict[str, RewardBreakdown] = {}
    total = len(queries)
    step = max(1, progress_every)
    for idx, query in enumerate(queries, start=1):
        per_qid[query.qid] = rewarder.score(query.qid, query.text, source_query=query.text)
        if idx % step == 0 or idx == total:
            print(f"[progress] stage=original {idx}/{total}")

    values = list(per_qid.values())
    metrics = summarize_reward_breakdowns(values, mrr_key="mrr_mean", recall_key="recall_mean", recall_aux_key="recall_dense_mean")
    mrr_value = metrics["mrr_mean"]
    recall_value = metrics["recall_mean"]
    recall_dense_value = metrics["recall_dense_mean"]
    return (
        {
            "mrr": mrr_value,
            f"mrr@{rewarder.mrr_k}": mrr_value,
            "recall": recall_value,
            f"recall@{rewarder.recall_k}": recall_value,
            "recall_dense": recall_dense_value,
            f"recall@{rewarder.recall_dense_k}": recall_dense_value,
            **metrics,
        },
        per_qid,
    )


def evaluate_with_model(
    model: ModelWrapper,
    queries: Sequence[QueryExample],
    rewarder: Rewarder,
    *,
    guardrail_cfg=None,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop_on: str | None = None,
    enforce_single_line: bool = False,
    query_batch_size: int = 1,
    stage_name: str,
    progress_every: int,
) -> tuple[dict[str, float], dict[str, tuple[str, RewardBreakdown]]]:
    """评估模型重写结果。"""

    def _postprocess_generated_query(text: str) -> str:
        cleaned = (text or "").strip()
        if stop_on:
            marker_idx = cleaned.find(stop_on)
            if marker_idx >= 0:
                cleaned = cleaned[:marker_idx].strip()
        if enforce_single_line:
            lines = [line.strip() for line in cleaned.replace("\r", "\n").split("\n") if line.strip()]
            cleaned = lines[0] if lines else ""
            cleaned = " ".join(cleaned.split())
        return cleaned

    per_qid: dict[str, tuple[str, RewardBreakdown]] = {}
    total = len(queries)
    step = max(1, progress_every)
    batch_size = max(1, int(query_batch_size))
    processed = 0
    for start in range(0, total, batch_size):
        batch = list(queries[start : start + batch_size])
        if hasattr(model, "generate_rewrite_batch"):
            batch_rewrites = model.generate_rewrite_batch(
                [query.text for query in batch],
                policy="actor",
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            batch_rewrites = [
                model.generate_rewrite(
                    query.text,
                    policy="actor",
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                for query in batch
            ]
        for query, raw_rewritten in zip(batch, batch_rewrites):
            rewritten = _postprocess_generated_query(raw_rewritten)
            final_query = rewritten
            if guardrail_cfg is not None and hasattr(rewarder, "cfg"):
                stabilized = stabilize_generated_rewrite(
                    rewritten,
                    source_query=query.text,
                    guardrail_cfg=guardrail_cfg,
                    reward_cfg=rewarder.cfg,
                )
                final_query = stabilized.final_query
            per_qid[query.qid] = (
                final_query,
                rewarder.score(query.qid, final_query, source_query=query.text),
            )
            processed += 1
        if processed % step == 0 or processed == total:
            print(f"[progress] stage={stage_name} {processed}/{total}")

    values = [item[1] for item in per_qid.values()]
    metrics = summarize_reward_breakdowns(values, mrr_key="mrr_mean", recall_key="recall_mean", recall_aux_key="recall_dense_mean")
    mrr_value = metrics["mrr_mean"]
    recall_value = metrics["recall_mean"]
    recall_dense_value = metrics["recall_dense_mean"]
    return (
        {
            "mrr": mrr_value,
            f"mrr@{rewarder.mrr_k}": mrr_value,
            "recall": recall_value,
            f"recall@{rewarder.recall_k}": recall_value,
            "recall_dense": recall_dense_value,
            f"recall@{rewarder.recall_dense_k}": recall_dense_value,
            **metrics,
        },
        per_qid,
    )


def main() -> int:
    """执行完整三路评估并输出报告。"""

    args = parse_args()
    adapter_cfg = validate_adapter_path(args.rl_adapter_path)
    adapter_base_model = str(adapter_cfg.get("base_model_name_or_path", "")).strip() or None

    config = apply_reward_mode_prompt_defaults(apply_overrides(get_default_config(), args))
    if args.model_name is None and adapter_base_model:
        # 默认优先使用 adapter 对应的 base model，避免错配。
        config.model.model_name = adapter_base_model

    if args.low_mem_mode:
        print("[mode] low-mem eval preset enabled.")
    print(
        "[config] "
        f"model={config.model.model_name}, "
        f"index={config.data.prebuilt_index}, "
        f"reward_mode={config.reward.reward_mode}, "
        f"mrr_k={config.reward.mrr_k}, "
        f"recall_k={config.reward.recall_k}, "
        f"query_batch_size={max(1, args.query_batch_size)}, "
        f"prompt_id={config.prompt.prompt_id}, "
        f"eval_max_new_tokens={config.prompt.max_new_tokens}, "
        f"eval_temperature={config.prompt.temperature}, "
        f"eval_top_p={config.prompt.top_p}, "
        f"load_in_4bit={config.model.load_in_4bit}, "
        f"adapter_base={adapter_base_model or '-'}"
    )

    queries, qrels = load_topics_qrels(config.data.topic_name)
    _, val_queries = split_queries(queries, train_ratio=config.data.train_ratio, seed=config.data.seed)
    val_queries = maybe_limit(val_queries, config.data.max_val_queries)
    print(f"[data] eval_queries={len(val_queries)}")

    rewarder = Rewarder(
        qrels=qrels,
        prebuilt_index=config.data.prebuilt_index,
        reward_cfg=config.reward,
    )

    print("[stage] evaluating original queries...")
    original_metrics, original_by_qid = evaluate_original(
        val_queries,
        rewarder,
        progress_every=args.progress_every,
    )

    print("[stage] loading zero-shot model...")
    zero_shot_model = ModelWrapper(
        model_cfg=config.model,
        prompt_cfg=config.prompt,
        train_mode=False,
        enable_lora=False,
        load_ref_model=False,
        strict_tokenizer_model_match=args.strict_tokenizer_model_match,
    )
    print("[stage] evaluating zero-shot rewrites...")
    zero_metrics, zero_by_qid = evaluate_with_model(
        zero_shot_model,
        val_queries,
        rewarder,
        guardrail_cfg=config.prompt,
        max_new_tokens=config.prompt.max_new_tokens,
        temperature=config.prompt.temperature,
        top_p=config.prompt.top_p,
        stop_on=config.prompt.stop_on,
        enforce_single_line=config.prompt.enforce_single_line,
        query_batch_size=args.query_batch_size,
        stage_name="zero-shot",
        progress_every=args.progress_every,
    )

    # 显式释放 zero-shot 模型，减少后续加载 RL 模型时的显存峰值。
    del zero_shot_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[stage] loading RL model...")
    rl_model = ModelWrapper(
        model_cfg=config.model,
        prompt_cfg=config.prompt,
        train_mode=False,
        enable_lora=True,
        load_ref_model=False,
        adapter_path=args.rl_adapter_path,
        strict_tokenizer_model_match=args.strict_tokenizer_model_match,
    )
    print("[stage] evaluating RL rewrites...")
    rl_metrics, rl_by_qid = evaluate_with_model(
        rl_model,
        val_queries,
        rewarder,
        guardrail_cfg=config.prompt,
        max_new_tokens=config.prompt.max_new_tokens,
        temperature=config.prompt.temperature,
        top_p=config.prompt.top_p,
        stop_on=config.prompt.stop_on,
        enforce_single_line=config.prompt.enforce_single_line,
        query_batch_size=args.query_batch_size,
        stage_name="rl",
        progress_every=args.progress_every,
    )

    mrr_label = f"mrr@{config.reward.mrr_k}"
    delta_zero = zero_metrics["mrr"] - original_metrics["mrr"]
    delta_rl = rl_metrics["mrr"] - original_metrics["mrr"]
    delta_rl_vs_zero = rl_metrics["mrr"] - zero_metrics["mrr"]
    recall_label = f"recall@{config.reward.recall_k}"

    print(f"\n=== {mrr_label} Comparison ===")
    print(f"Original : {original_metrics['mrr']:.4f}")
    print(f"Zero-shot: {zero_metrics['mrr']:.4f} (delta vs original {delta_zero:+.4f})")
    print(f"RL       : {rl_metrics['mrr']:.4f} (delta vs original {delta_rl:+.4f})")
    print(f"RL vs Zero-shot delta: {delta_rl_vs_zero:+.4f}")
    print(f"\n=== {recall_label} Comparison ===")
    print(f"Original : {original_metrics['recall']:.4f}")
    print(f"Zero-shot: {zero_metrics['recall']:.4f}")
    print(f"RL       : {rl_metrics['recall']:.4f}")

    per_qid_rows = build_per_qid_rows(
        val_queries,
        original_by_qid,
        zero_by_qid,
        rl_by_qid,
    )

    print("\n=== Sample Cases ===")
    sample_cases = select_sample_cases(per_qid_rows, args.sample_print)
    for row in sample_cases:
        original_score = row["original"]
        zero_score = row["zero_shot"]
        rl_score = row["rl"]
        print(f"[{row['qid']}]")
        print(f"  original_query : {row['original_query']}")
        print(f"  zero_rewrite   : {row['zero_rewrite']}")
        print(f"  rl_rewrite     : {row['rl_rewrite']}")
        print(
            f"  mrr(original/zero/rl): "
            f"{float(original_score['mrr']):.4f}/{float(zero_score['mrr']):.4f}/{float(rl_score['mrr']):.4f}"
        )

    report = {
        "config": config.to_dict(),
        "num_eval_queries": len(val_queries),
        "original": original_metrics,
        "zero_shot": zero_metrics,
        "rl": rl_metrics,
        "per_qid": per_qid_rows,
        "deltas": {
            "zero_minus_original": delta_zero,
            "rl_minus_original": delta_rl,
            "rl_minus_zero": delta_rl_vs_zero,
            "zero_reward_minus_original": zero_metrics.get("reward_mean", 0.0) - original_metrics.get("reward_mean", 0.0),
            "rl_reward_minus_original": rl_metrics.get("reward_mean", 0.0) - original_metrics.get("reward_mean", 0.0),
            "rl_main_reward_minus_zero": rl_metrics.get("main_reward_mean", 0.0) - zero_metrics.get("main_reward_mean", 0.0),
        },
        "samples": sample_cases,
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[report] {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
