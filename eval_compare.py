from __future__ import annotations

"""三路对比评估脚本。

在同一份验证集上比较三种 query 形式：
1. Original：原始 query（不重写）
2. Zero-shot：基础模型提示重写（未经过 RL）
3. RL：加载 GRPO 训练后的 LoRA adapter 重写
"""

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Sequence

import torch

from app_config import AppConfig, get_default_config
from core.model_wrapper import ModelWrapper
from core.reward_func import RewardBreakdown, Rewarder
from data.loader import QueryExample, load_topics_qrels, maybe_limit, split_queries


def parse_args() -> argparse.Namespace:
    """解析评测命令行参数。"""

    parser = argparse.ArgumentParser(description="Compare Original vs Zero-shot vs RL-rewritten query MRR@10.")
    parser.add_argument("--rl-adapter-path", type=str, required=True, help="Path to trained LoRA adapter.")
    parser.add_argument("--model-name", type=str, default=None, help="Override base model name for evaluation.")
    parser.add_argument("--topic-name", type=str, default=None)
    parser.add_argument("--prebuilt-index", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-eval-queries", type=int, default=None)
    parser.add_argument("--sample-print", type=int, default=5)
    parser.add_argument("--report-path", type=str, default="artifacts/eval_compare_report.json")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--low-mem-mode",
        action="store_true",
        help="Use low-memory evaluation preset (0.5B model + slim index + smaller eval set).",
    )
    return parser.parse_args()


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """应用配置覆盖参数。"""

    if args.low_mem_mode:
        config.model.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        config.data.prebuilt_index = "msmarco-v1-passage-slim"
        config.data.max_val_queries = 100
        config.train.max_new_tokens = 16
        config.reward.topk = 20

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
    if args.max_new_tokens is not None:
        config.train.max_new_tokens = args.max_new_tokens
    return config


def evaluate_original(queries: Sequence[QueryExample], rewarder: Rewarder) -> tuple[dict[str, float], dict[str, RewardBreakdown]]:
    """评估原始 query 基线。"""

    per_qid: dict[str, RewardBreakdown] = {}
    for query in queries:
        per_qid[query.qid] = rewarder.score(query.qid, query.text, source_query=query.text)

    values = list(per_qid.values())
    return (
        {
            "mrr@10": fmean(v.mrr for v in values) if values else 0.0,
            "reward_mean": fmean(v.total for v in values) if values else 0.0,
        },
        per_qid,
    )


def evaluate_with_model(
    model: ModelWrapper,
    queries: Sequence[QueryExample],
    rewarder: Rewarder,
    *,
    max_new_tokens: int,
) -> tuple[dict[str, float], dict[str, tuple[str, RewardBreakdown]]]:
    """评估模型重写后的 query。"""

    per_qid: dict[str, tuple[str, RewardBreakdown]] = {}
    for query in queries:
        rewritten = model.generate_rewrite(
            query.text,
            policy="actor",
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        per_qid[query.qid] = (rewritten, rewarder.score(query.qid, rewritten, source_query=query.text))

    values = [item[1] for item in per_qid.values()]
    return (
        {
            "mrr@10": fmean(v.mrr for v in values) if values else 0.0,
            "reward_mean": fmean(v.total for v in values) if values else 0.0,
        },
        per_qid,
    )


def main() -> int:
    """执行完整三路评估并输出报告。"""

    args = parse_args()
    config = apply_overrides(get_default_config(), args)
    if args.low_mem_mode:
        print("[mode] low-mem eval preset enabled.")
    print(f"[config] model={config.model.model_name}, index={config.data.prebuilt_index}, topk={config.reward.topk}")

    # 与训练保持同样的数据切分策略，确保比较公平。
    queries, qrels = load_topics_qrels(config.data.topic_name)
    _, val_queries = split_queries(queries, train_ratio=config.data.train_ratio, seed=config.data.seed)
    val_queries = maybe_limit(val_queries, config.data.max_val_queries)
    print(f"[data] eval_queries={len(val_queries)}")

    rewarder = Rewarder(
        qrels=qrels,
        prebuilt_index=config.data.prebuilt_index,
        reward_cfg=config.reward,
    )

    original_metrics, original_by_qid = evaluate_original(val_queries, rewarder)

    # Zero-shot：基础模型直接重写，不加载 LoRA。
    zero_shot_model = ModelWrapper(
        model_cfg=config.model,
        prompt_cfg=config.prompt,
        train_mode=False,
        enable_lora=False,
        load_ref_model=False,
    )
    zero_metrics, zero_by_qid = evaluate_with_model(
        zero_shot_model,
        val_queries,
        rewarder,
        max_new_tokens=config.train.max_new_tokens,
    )
    # 显式释放 zero-shot 模型，降低后续加载 RL 模型的峰值显存占用。
    del zero_shot_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # RL：基础模型 + 已训练 adapter。
    rl_model = ModelWrapper(
        model_cfg=config.model,
        prompt_cfg=config.prompt,
        train_mode=False,
        enable_lora=True,
        load_ref_model=False,
        adapter_path=args.rl_adapter_path,
    )
    rl_metrics, rl_by_qid = evaluate_with_model(
        rl_model,
        val_queries,
        rewarder,
        max_new_tokens=config.train.max_new_tokens,
    )

    delta_zero = zero_metrics["mrr@10"] - original_metrics["mrr@10"]
    delta_rl = rl_metrics["mrr@10"] - original_metrics["mrr@10"]
    delta_rl_vs_zero = rl_metrics["mrr@10"] - zero_metrics["mrr@10"]

    print("\n=== MRR@10 Comparison ===")
    print(f"Original : {original_metrics['mrr@10']:.4f}")
    print(f"Zero-shot: {zero_metrics['mrr@10']:.4f} (delta vs original {delta_zero:+.4f})")
    print(f"RL       : {rl_metrics['mrr@10']:.4f} (delta vs original {delta_rl:+.4f})")
    print(f"RL vs Zero-shot delta: {delta_rl_vs_zero:+.4f}")

    print("\n=== Sample Cases ===")
    sample_count = max(0, args.sample_print)
    for query in val_queries[:sample_count]:
        qid = query.qid
        zero_rewrite, zero_score = zero_by_qid[qid]
        rl_rewrite, rl_score = rl_by_qid[qid]
        original_score = original_by_qid[qid]
        print(f"[{qid}]")
        print(f"  original_query : {query.text}")
        print(f"  zero_rewrite   : {zero_rewrite}")
        print(f"  rl_rewrite     : {rl_rewrite}")
        print(
            f"  mrr(original/zero/rl): "
            f"{original_score.mrr:.4f}/{zero_score.mrr:.4f}/{rl_score.mrr:.4f}"
        )

    report = {
        "config": config.to_dict(),
        "num_eval_queries": len(val_queries),
        "original": original_metrics,
        "zero_shot": zero_metrics,
        "rl": rl_metrics,
        "deltas": {
            "zero_minus_original": delta_zero,
            "rl_minus_original": delta_rl,
            "rl_minus_zero": delta_rl_vs_zero,
        },
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[report] {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
