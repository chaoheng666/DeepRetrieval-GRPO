from __future__ import annotations

"""Top20_delta full eval：Original / Zero-shot / RL adapter 三路对比。

评估只做推理和检索打分，不训练：
1. original：原 query 直接 BM25；
2. zero-shot：基座模型按 prompt 改写；
3. RL：加载训练出的 LoRA adapter 改写；
4. 汇总 per-qid 明细和整体 delta，写入 JSON 报告。
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
    parser = argparse.ArgumentParser(description="Compare Original, zero-shot, and top20_delta RL rewrites.")
    parser.add_argument("--rl-adapter-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--strict-tokenizer-model-match", action="store_true")
    parser.add_argument("--topic-name", type=str, default=None)
    parser.add_argument("--prebuilt-index", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-eval-queries", type=int, default=None)
    parser.add_argument("--search-threads", type=int, default=None)
    parser.add_argument("--reward-mrr-k", type=int, default=None)
    parser.add_argument("--reward-recall-k", type=int, default=None)
    parser.add_argument("--reward-recall-dense-k", type=int, default=None)
    parser.add_argument("--reward-w-mrr", type=float, default=None)
    parser.add_argument("--reward-w-recall", type=float, default=None)
    parser.add_argument("--reward-w-recall-dense", type=float, default=None)
    parser.add_argument("--reward-w-rank-bonus", type=float, default=None)
    parser.add_argument("--reward-w-bad-format", type=float, default=None)
    parser.add_argument("--reward-w-unsafe-copy", type=float, default=None)
    parser.add_argument("--reward-w-overedit", type=float, default=None)
    parser.add_argument("--overedit-tau", type=float, default=None)
    parser.add_argument("--recall-drop-lambda", type=float, default=None)
    parser.add_argument("--anchor-bonus-value", type=float, default=None)
    parser.add_argument("--format-max-tokens", type=int, default=None)
    parser.add_argument("--format-min-english-ratio", type=float, default=None)
    parser.add_argument("--format-max-unreadable-ratio", type=float, default=None)
    parser.add_argument("--bad-format-cap", type=float, default=None)
    parser.add_argument("--sample-print", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument(
        "--report-path",
        type=str,
        default="train_and_eval_data_model_0421/artifacts_4b_top20_delta_curriculum/eval/eval_compare_report_full.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    return parser.parse_args()


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    # eval 入口保留必要覆盖项，但固定 top20_delta 和 4bit 主线。
    if args.model_name is not None:
        config.model.model_name = args.model_name
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
    if args.reward_w_mrr is not None:
        config.reward.w_mrr = args.reward_w_mrr
    if args.reward_w_recall is not None:
        config.reward.w_recall = args.reward_w_recall
    if args.reward_w_recall_dense is not None:
        config.reward.w_recall_dense = args.reward_w_recall_dense
    if args.reward_w_rank_bonus is not None:
        config.reward.w_rank_bonus = args.reward_w_rank_bonus
    if args.reward_w_bad_format is not None:
        config.reward.w_bad_format = args.reward_w_bad_format
    if args.reward_w_unsafe_copy is not None:
        config.reward.w_unsafe_copy = args.reward_w_unsafe_copy
    if args.reward_w_overedit is not None:
        config.reward.w_overedit = args.reward_w_overedit
    if args.overedit_tau is not None:
        config.reward.overedit_tau = args.overedit_tau
    if args.recall_drop_lambda is not None:
        config.reward.recall_drop_lambda = max(0.0, args.recall_drop_lambda)
    if args.anchor_bonus_value is not None:
        config.reward.anchor_bonus_value = max(0.0, args.anchor_bonus_value)
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

    config.reward.reward_mode = "top20_delta"
    config.model.ref_precision_mode = "4bit"
    return config


def validate_adapter_path(adapter_path: str) -> dict:
    # RL 评估必须显式给 adapter；这里提前验证目录和 adapter_config。
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_dir}")
    if not adapter_dir.is_dir():
        raise NotADirectoryError(f"Adapter path is not a directory: {adapter_dir}")

    adapter_cfg_path = adapter_dir / "adapter_config.json"
    if not adapter_cfg_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in: {adapter_dir}")
    with adapter_cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def reward_breakdown_to_report_dict(score: RewardBreakdown) -> dict[str, Any]:
    # 报告里保留足够的 reward 拆解，方便定位是 MRR、recall 还是 penalty 在变化。
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
        "anchor_bonus": getattr(score, "anchor_bonus", 0.0),
        "recall_drop_penalty": getattr(score, "recall_drop_penalty", 0.0),
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
    # 每个 qid 保留三路 query 和三路 reward，方便后续人工抽样检查。
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
    # 原 query baseline：不经过模型，直接检索打分。
    per_qid: dict[str, RewardBreakdown] = {}
    total = len(queries)
    step = max(1, progress_every)
    for idx, query in enumerate(queries, start=1):
        per_qid[query.qid] = rewarder.score(query.qid, query.text, source_query=query.text)
        if idx % step == 0 or idx == total:
            print(f"[progress] stage=original {idx}/{total}")

    values = list(per_qid.values())
    metrics = summarize_reward_breakdowns(values, mrr_key="mrr_mean", recall_key="recall_mean", recall_aux_key="recall_dense_mean")
    return (
        {
            "mrr": metrics["mrr_mean"],
            f"mrr@{rewarder.mrr_k}": metrics["mrr_mean"],
            "recall": metrics["recall_mean"],
            f"recall@{rewarder.recall_k}": metrics["recall_mean"],
            "recall_dense": metrics["recall_dense_mean"],
            f"recall@{rewarder.recall_dense_k}": metrics["recall_dense_mean"],
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
    # zero-shot/RL 共用这条路径：批量生成 -> 截断/guardrail -> BM25 reward。
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
            per_qid[query.qid] = (final_query, rewarder.score(query.qid, final_query, source_query=query.text))
            processed += 1
        if processed % step == 0 or processed == total:
            print(f"[progress] stage={stage_name} {processed}/{total}")

    values = [item[1] for item in per_qid.values()]
    metrics = summarize_reward_breakdowns(values, mrr_key="mrr_mean", recall_key="recall_mean", recall_aux_key="recall_dense_mean")
    return (
        {
            "mrr": metrics["mrr_mean"],
            f"mrr@{rewarder.mrr_k}": metrics["mrr_mean"],
            "recall": metrics["recall_mean"],
            f"recall@{rewarder.recall_k}": metrics["recall_mean"],
            "recall_dense": metrics["recall_dense_mean"],
            f"recall@{rewarder.recall_dense_k}": metrics["recall_dense_mean"],
            **metrics,
        },
        per_qid,
    )


def main() -> int:
    args = parse_args()
    # 1) adapter_config 里通常记录了基座模型路径；未显式传 model-name 时复用它。
    adapter_cfg = validate_adapter_path(args.rl_adapter_path)
    adapter_base_model = str(adapter_cfg.get("base_model_name_or_path", "")).strip() or None

    config = apply_reward_mode_prompt_defaults(apply_overrides(get_default_config(), args))
    if args.model_name is None and adapter_base_model:
        config.model.model_name = adapter_base_model

    print(
        "[config] "
        f"model={config.model.model_name}, "
        f"index={config.data.prebuilt_index}, "
        f"mrr_k={config.reward.mrr_k}, "
        f"recall_k={config.reward.recall_k}, "
        f"recall_dense_k={config.reward.recall_dense_k}, "
        f"query_batch_size={max(1, args.query_batch_size)}, "
        f"prompt_id={config.prompt.prompt_id}, "
        f"eval_max_new_tokens={config.prompt.max_new_tokens}, "
        f"eval_temperature={config.prompt.temperature}, "
        f"eval_top_p={config.prompt.top_p}, "
        f"load_in_4bit={config.model.load_in_4bit}, "
        f"adapter_base={adapter_base_model or '-'}"
    )

    # 2) 数据只取 validation split，和训练入口保持同一切分逻辑。
    queries, qrels = load_topics_qrels(config.data.topic_name)
    _, val_queries = split_queries(queries, train_ratio=config.data.train_ratio, seed=config.data.seed)
    val_queries = maybe_limit(val_queries, config.data.max_val_queries)
    print(f"[data] eval_queries={len(val_queries)}")

    rewarder = Rewarder(qrels=qrels, prebuilt_index=config.data.prebuilt_index, reward_cfg=config.reward)

    # 3) 三路评估：original -> zero-shot -> RL adapter。
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

    del zero_shot_model
    if torch.cuda.is_available():
        # 先释放 zero-shot 模型显存，再加载 RL adapter。
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

    # 4) 汇总核心 delta，并写出完整 per-qid 报告。
    mrr_label = f"mrr@{config.reward.mrr_k}"
    recall_label = f"recall@{config.reward.recall_k}"
    delta_zero = zero_metrics["mrr"] - original_metrics["mrr"]
    delta_rl = rl_metrics["mrr"] - original_metrics["mrr"]
    delta_rl_vs_zero = rl_metrics["mrr"] - zero_metrics["mrr"]

    print(f"\n=== {mrr_label} Comparison ===")
    print(f"Original : {original_metrics['mrr']:.4f}")
    print(f"Zero-shot: {zero_metrics['mrr']:.4f} (delta vs original {delta_zero:+.4f})")
    print(f"RL       : {rl_metrics['mrr']:.4f} (delta vs original {delta_rl:+.4f})")
    print(f"RL vs Zero-shot delta: {delta_rl_vs_zero:+.4f}")
    print(f"\n=== {recall_label} Comparison ===")
    print(f"Original : {original_metrics['recall']:.4f}")
    print(f"Zero-shot: {zero_metrics['recall']:.4f}")
    print(f"RL       : {rl_metrics['recall']:.4f}")

    per_qid_rows = build_per_qid_rows(val_queries, original_by_qid, zero_by_qid, rl_by_qid)
    sample_cases = select_sample_cases(per_qid_rows, args.sample_print)
    print("\n=== Sample Cases ===")
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
