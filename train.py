from __future__ import annotations

"""Top20-delta curriculum GRPO 训练入口。

主线流程：
1. 读取默认配置和 CLI 覆盖；
2. 加载 topics/qrels，并生成 curriculum metadata；
3. 按 phase 采样训练 query；
4. GRPOEngine 执行采样、奖励、advantage、PPO/GRPO 更新；
5. 周期性验证并保存 latest/best adapter。
"""

import argparse
import json
import os
import random
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import torch

from app_config import AppConfig, apply_reward_mode_prompt_defaults, ensure_runtime_dirs, get_default_config
from core.grpo_engine import GRPOEngine
from core.model_wrapper import ModelWrapper
from core.reward_func import Rewarder, stabilize_generated_rewrite, summarize_reward_breakdowns
from data.curriculum import ensure_curriculum_metadata, filter_curriculum_train_queries, sample_curriculum_queries
from data.loader import QueryExample, load_topics_qrels, maybe_limit, split_queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 4B top20_delta curriculum query rewriter.")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--topic-name", type=str, default=None)
    parser.add_argument("--prebuilt-index", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--max-group-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--clip-range", type=float, default=None)
    parser.add_argument("--kl-beta", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--eval-max-new-tokens", type=int, default=None)
    parser.add_argument("--eval-temperature", type=float, default=None)
    parser.add_argument("--eval-top-p", type=float, default=None)
    parser.add_argument("--eval-query-batch-size", type=int, default=None)

    parser.add_argument("--group-temperature-stride", type=float, default=None)
    parser.add_argument("--group-top-p-stride", type=float, default=None)
    parser.add_argument("--min-unique-final-queries", type=int, default=None)
    parser.add_argument("--max-regen-rounds", type=int, default=None)
    parser.add_argument("--reward-gap-threshold", type=float, default=None)
    parser.add_argument("--gap-sampling-temperature-delta", type=float, default=None)
    parser.add_argument("--actor-chunk-size", type=int, default=None)
    parser.add_argument("--projection-chunk-size", type=int, default=None)

    parser.add_argument("--reward-mrr-k", type=int, default=None)
    parser.add_argument("--reward-recall-k", type=int, default=None)
    parser.add_argument("--reward-recall-dense-k", type=int, default=None)
    parser.add_argument("--search-threads", type=int, default=None)
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

    parser.add_argument("--eval-every-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-queries", type=int, default=None)
    parser.add_argument("--max-val-queries", type=int, default=None)
    parser.add_argument("--curriculum-phase", choices=("phase1", "phase2"), default=None)
    parser.add_argument("--curriculum-metadata-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--group-trace-log-path", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--strict-tokenizer-model-match", action="store_true")
    return parser.parse_args()


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    # CLI 只覆盖目标 .sh 需要的主线参数；legacy/low-mem 分支已经移除。
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
    if args.max_train_queries is not None:
        config.data.max_train_queries = args.max_train_queries
    if args.max_val_queries is not None:
        config.data.max_val_queries = args.max_val_queries

    if args.num_epochs is not None:
        config.train.num_epochs = args.num_epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.group_size is not None:
        config.train.group_size = args.group_size
    if args.max_group_size is not None:
        config.train.max_group_size = args.max_group_size
    if args.learning_rate is not None:
        config.train.learning_rate = args.learning_rate
    if args.clip_range is not None:
        config.train.clip_range = args.clip_range
    if args.kl_beta is not None:
        config.train.kl_beta = args.kl_beta
    if args.max_new_tokens is not None:
        config.train.max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        config.train.temperature = args.temperature
    if args.top_p is not None:
        config.train.top_p = args.top_p
    if args.group_temperature_stride is not None:
        config.train.group_temperature_stride = args.group_temperature_stride
    if args.group_top_p_stride is not None:
        config.train.group_top_p_stride = args.group_top_p_stride
    if args.min_unique_final_queries is not None:
        config.train.min_unique_final_queries = args.min_unique_final_queries
    if args.max_regen_rounds is not None:
        config.train.max_regen_rounds = args.max_regen_rounds
    if args.reward_gap_threshold is not None:
        config.train.reward_gap_threshold = args.reward_gap_threshold
    if args.gap_sampling_temperature_delta is not None:
        config.train.gap_sampling_temperature_delta = args.gap_sampling_temperature_delta
    if args.actor_chunk_size is not None:
        config.train.actor_chunk_size = args.actor_chunk_size
    if args.projection_chunk_size is not None:
        config.model.projection_chunk_size = args.projection_chunk_size

    if args.reward_mrr_k is not None:
        config.reward.mrr_k = args.reward_mrr_k
    if args.reward_recall_k is not None:
        config.reward.recall_k = args.reward_recall_k
    if args.reward_recall_dense_k is not None:
        config.reward.recall_dense_k = args.reward_recall_dense_k
    if args.search_threads is not None:
        config.reward.search_threads = max(1, args.search_threads)
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
        config.reward.recall_drop_lambda = args.recall_drop_lambda
    if args.anchor_bonus_value is not None:
        config.reward.anchor_bonus_value = args.anchor_bonus_value
    if args.format_max_tokens is not None:
        config.reward.format_max_tokens = args.format_max_tokens
    if args.format_min_english_ratio is not None:
        config.reward.format_min_english_ratio = args.format_min_english_ratio
    if args.format_max_unreadable_ratio is not None:
        config.reward.format_max_unreadable_ratio = args.format_max_unreadable_ratio
    if args.bad_format_cap is not None:
        config.reward.bad_format_cap = args.bad_format_cap

    if args.eval_every_steps is not None:
        config.train.eval_every_steps = args.eval_every_steps
    if args.max_steps is not None:
        config.train.max_steps = args.max_steps
    if args.curriculum_phase is not None:
        config.train.curriculum_phase = args.curriculum_phase
    if args.curriculum_metadata_path is not None:
        config.train.curriculum_metadata_path = args.curriculum_metadata_path
    if args.save_dir is not None:
        config.train.save_dir = args.save_dir
    if args.log_path is not None:
        config.train.log_path = args.log_path
    if args.group_trace_log_path is not None:
        config.train.group_trace_log_path = args.group_trace_log_path

    config.train.curriculum_enable = True
    config.reward.reward_mode = "top20_delta"
    config.model.ref_precision_mode = "4bit"
    return config


def apply_runtime_mode_adjustments(config: AppConfig, args: argparse.Namespace | None = None) -> AppConfig:
    # 运行前做一次兜底修正，避免无效参数进入训练循环。
    del args
    if torch.cuda.is_available():
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if not 0.0 < config.data.train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {config.data.train_ratio}")
    if config.train.curriculum_phase not in {"phase1", "phase2"}:
        raise ValueError(f"Unsupported curriculum_phase: {config.train.curriculum_phase}")

    if config.train.num_epochs < 1:
        print(f"[warn] num_epochs={config.train.num_epochs} is invalid; auto-adjusting to 1.")
        config.train.num_epochs = 1
    if config.train.batch_size < 1:
        print(f"[warn] batch_size={config.train.batch_size} is invalid; auto-adjusting to 1.")
        config.train.batch_size = 1
    if config.train.eval_every_steps < 1:
        print(f"[warn] eval_every_steps={config.train.eval_every_steps} is invalid; auto-adjusting to 1.")
        config.train.eval_every_steps = 1
    if config.train.max_new_tokens < 1:
        print(f"[warn] max_new_tokens={config.train.max_new_tokens} is invalid; auto-adjusting to 1.")
        config.train.max_new_tokens = 1
    if config.reward.mrr_k < 1:
        print(f"[warn] reward_mrr_k={config.reward.mrr_k} is invalid; auto-adjusting to 1.")
        config.reward.mrr_k = 1
    if config.reward.recall_k < 1:
        print(f"[warn] reward_recall_k={config.reward.recall_k} is invalid; auto-adjusting to 1.")
        config.reward.recall_k = 1
    if config.reward.recall_dense_k < 1:
        print(f"[warn] reward_recall_dense_k={config.reward.recall_dense_k} is invalid; auto-adjusting to 1.")
        config.reward.recall_dense_k = 1
    if config.data.max_train_queries is not None and config.data.max_train_queries < 0:
        print(f"[warn] max_train_queries={config.data.max_train_queries} is invalid; auto-adjusting to 0.")
        config.data.max_train_queries = 0
    if config.data.max_val_queries is not None and config.data.max_val_queries < 0:
        print(f"[warn] max_val_queries={config.data.max_val_queries} is invalid; auto-adjusting to 0.")
        config.data.max_val_queries = 0
    if config.train.max_steps is not None and config.train.max_steps < 1:
        print(f"[warn] max_steps={config.train.max_steps} is invalid; auto-adjusting to 1.")
        config.train.max_steps = 1
    if config.train.group_size < 2:
        print(f"[warn] group_size={config.train.group_size} is invalid; auto-adjusting to 2.")
        config.train.group_size = 2
    if config.train.max_group_size < config.train.group_size:
        print(
            f"[warn] max_group_size={config.train.max_group_size} is smaller than "
            f"group_size={config.train.group_size}; auto-adjusting to {config.train.group_size}."
        )
        config.train.max_group_size = config.train.group_size
    if config.train.actor_chunk_size < 1:
        print(f"[warn] actor_chunk_size={config.train.actor_chunk_size} is invalid; auto-adjusting to 1.")
        config.train.actor_chunk_size = 1
    if config.model.projection_chunk_size < 1:
        print(f"[warn] projection_chunk_size={config.model.projection_chunk_size} is invalid; auto-adjusting to 1.")
        config.model.projection_chunk_size = 1
    if config.train.reward_gap_threshold < 0.0:
        print(f"[warn] reward_gap_threshold={config.train.reward_gap_threshold} is invalid; auto-adjusting to 0.0.")
        config.train.reward_gap_threshold = 0.0
    if config.train.gap_sampling_temperature_delta < 0.0:
        print(
            f"[warn] gap_sampling_temperature_delta={config.train.gap_sampling_temperature_delta} "
            "is invalid; auto-adjusting to 0.0."
        )
        config.train.gap_sampling_temperature_delta = 0.0
    if config.train.group_temperature_stride < 0.0:
        print(f"[warn] group_temperature_stride={config.train.group_temperature_stride} is invalid; auto-adjusting to 0.0.")
        config.train.group_temperature_stride = 0.0
    if config.train.group_top_p_stride < 0.0:
        print(f"[warn] group_top_p_stride={config.train.group_top_p_stride} is invalid; auto-adjusting to 0.0.")
        config.train.group_top_p_stride = 0.0
    if config.train.min_unique_final_queries < 1:
        print(f"[warn] min_unique_final_queries={config.train.min_unique_final_queries} is invalid; auto-adjusting to 1.")
        config.train.min_unique_final_queries = 1
    if config.train.max_regen_rounds < 0:
        print(f"[warn] max_regen_rounds={config.train.max_regen_rounds} is invalid; auto-adjusting to 0.")
        config.train.max_regen_rounds = 0
    if config.reward.bad_format_cap < 0.0:
        print(f"[warn] bad_format_cap={config.reward.bad_format_cap} is invalid; auto-adjusting to 0.0.")
        config.reward.bad_format_cap = 0.0
    if config.reward.recall_drop_lambda < 0.0:
        print(f"[warn] recall_drop_lambda={config.reward.recall_drop_lambda} is invalid; auto-adjusting to 0.0.")
        config.reward.recall_drop_lambda = 0.0
    if config.reward.anchor_bonus_value < 0.0:
        print(f"[warn] anchor_bonus_value={config.reward.anchor_bonus_value} is invalid; auto-adjusting to 0.0.")
        config.reward.anchor_bonus_value = 0.0

    config.train.curriculum_enable = True
    config.reward.reward_mode = "top20_delta"
    config.model.ref_precision_mode = "4bit"
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def iter_batches(items: Sequence[QueryExample], batch_size: int) -> Iterable[list[QueryExample]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def evaluate_policy(
    model: ModelWrapper,
    rewarder: Rewarder,
    queries: Sequence[QueryExample],
    *,
    guardrail_cfg,
    max_queries: int | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    query_batch_size: int = 1,
) -> dict[str, float]:
    # 训练中验证：actor 生成 rewrite -> guardrail 清洗/兜底 -> rewarder 统计 top20 指标。
    eval_queries = list(queries[:max_queries]) if max_queries is not None else list(queries)
    scored_values = []

    actor_model = getattr(model, "actor_model", None)
    previous_mode = getattr(actor_model, "training", None)
    if actor_model is not None:
        actor_model.eval()

    try:
        batch_size = max(1, int(query_batch_size))
        for start in range(0, len(eval_queries), batch_size):
            batch = eval_queries[start : start + batch_size]
            if hasattr(model, "generate_rewrite_batch"):
                rewrites = model.generate_rewrite_batch(
                    [query.text for query in batch],
                    policy="actor",
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                rewrites = [
                    model.generate_rewrite(
                        query.text,
                        policy="actor",
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    for query in batch
                ]

            for query, rewritten in zip(batch, rewrites):
                stabilized = stabilize_generated_rewrite(
                    rewritten,
                    source_query=query.text,
                    guardrail_cfg=guardrail_cfg,
                    reward_cfg=rewarder.cfg,
                )
                scored_values.append(rewarder.score(query.qid, stabilized.final_query, source_query=query.text))
    finally:
        if actor_model is not None and previous_mode is not None:
            actor_model.train(previous_mode)

    return summarize_reward_breakdowns(scored_values)


def resolve_eval_decode_settings(config: AppConfig, args: argparse.Namespace) -> dict[str, float | int]:
    # 训练采样和验证解码可以分开覆盖；默认沿用训练解码参数。
    return {
        "max_new_tokens": int(
            args.eval_max_new_tokens if getattr(args, "eval_max_new_tokens", None) is not None else config.train.max_new_tokens
        ),
        "temperature": float(
            args.eval_temperature if getattr(args, "eval_temperature", None) is not None else config.train.temperature
        ),
        "top_p": float(args.eval_top_p if getattr(args, "eval_top_p", None) is not None else config.train.top_p),
        "query_batch_size": max(
            1,
            int(
                args.eval_query_batch_size
                if getattr(args, "eval_query_batch_size", None) is not None
                else config.train.batch_size
            ),
        ),
    }


def evaluate_original(
    rewarder: Rewarder,
    queries: Sequence[QueryExample],
    *,
    max_queries: int | None,
) -> dict[str, float]:
    eval_queries = list(queries[:max_queries]) if max_queries is not None else list(queries)
    scored_values = [rewarder.score(query.qid, query.text, source_query=query.text) for query in eval_queries]
    return summarize_reward_breakdowns(scored_values)


def main() -> int:
    args = parse_args()
    # 1) 配置收敛：默认 top20_delta，CLI 只做主线覆盖。
    config = apply_reward_mode_prompt_defaults(apply_overrides(get_default_config(), args))
    config = apply_runtime_mode_adjustments(config)
    ensure_runtime_dirs(config)
    set_seed(config.data.seed)

    print("[config]")
    print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
    eval_decode = resolve_eval_decode_settings(config, args)
    print(
        f"[decode] train max_new_tokens={config.train.max_new_tokens} "
        f"temperature={config.train.temperature} top_p={config.train.top_p}"
    )
    print(
        f"[decode] eval  max_new_tokens={eval_decode['max_new_tokens']} "
        f"temperature={eval_decode['temperature']} top_p={eval_decode['top_p']} "
        f"query_batch_size={eval_decode['query_batch_size']}"
    )

    # 2) 加载数据并切分 train/val；max_* 参数只截断数量，不改变随机切分规则。
    queries, qrels = load_topics_qrels(config.data.topic_name)
    train_queries, val_queries = split_queries(queries, train_ratio=config.data.train_ratio, seed=config.data.seed)
    train_queries = maybe_limit(train_queries, config.data.max_train_queries)
    val_queries = maybe_limit(val_queries, config.data.max_val_queries)

    # 3) Curriculum metadata 会记录原 query 难度，phase1/phase2 用它挑不同训练样本。
    rewarder = Rewarder(qrels=qrels, prebuilt_index=config.data.prebuilt_index, reward_cfg=config.reward)
    metadata_path = Path(
        config.train.curriculum_metadata_path
        or (Path(config.train.save_dir).resolve().parent.parent / "curriculum_query_metadata.jsonl")
    )
    config.train.curriculum_metadata_path = str(metadata_path)
    metadata_by_qid = ensure_curriculum_metadata(train_queries, rewarder, metadata_path=metadata_path)
    train_queries, curriculum_bucket_counts = filter_curriculum_train_queries(train_queries, metadata_by_qid)
    if not train_queries:
        raise RuntimeError("Curriculum filtering left no train queries.")

    print(
        f"[curriculum] phase={config.train.curriculum_phase} "
        f"metadata={metadata_path} bucket_counts={curriculum_bucket_counts}"
    )
    print(f"[data] train_queries={len(train_queries)}, val_queries={len(val_queries)}, qrels_qids={len(qrels)}")

    # 4) 原 query baseline 用来判断 rewrite 是否真正提升，而不是只看绝对 MRR。
    mrr_label = f"MRR@{config.reward.mrr_k}"
    base_original_val = evaluate_original(rewarder, val_queries, max_queries=config.data.max_val_queries)
    print(f"[baseline] original_val_{mrr_label}={base_original_val['mrr_mean']:.4f}")

    # 5) 加载 actor/ref。actor 可带 adapter 热启动，ref 冻结只用于 KL。
    model = ModelWrapper(
        model_cfg=config.model,
        prompt_cfg=config.prompt,
        train_mode=True,
        enable_lora=True,
        load_ref_model=True,
        adapter_path=args.adapter_path,
        strict_tokenizer_model_match=args.strict_tokenizer_model_match,
    )
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    # 6) GRPOEngine 封装每个 train_step 的采样、奖励、advantage 和反传。
    engine = GRPOEngine(
        model_wrapper=model,
        rewarder=rewarder,
        optimizer=optimizer,
        group_size=config.train.group_size,
        clip_range=config.train.clip_range,
        kl_beta=config.train.kl_beta,
        grad_clip_norm=config.train.grad_clip_norm,
        max_new_tokens=config.train.max_new_tokens,
        temperature=config.train.temperature,
        top_p=config.train.top_p,
        group_temperature_stride=config.train.group_temperature_stride,
        group_top_p_stride=config.train.group_top_p_stride,
        max_group_size=config.train.max_group_size,
        min_unique_final_queries=config.train.min_unique_final_queries,
        max_regen_rounds=config.train.max_regen_rounds,
        regen_temperature_delta=config.train.regen_temperature_delta,
        reward_gap_threshold=config.train.reward_gap_threshold,
        gap_sampling_temperature_delta=config.train.gap_sampling_temperature_delta,
        actor_chunk_size=config.train.actor_chunk_size,
    )

    log_path = Path(config.train.log_path)
    group_trace_log_path = Path(config.train.group_trace_log_path)
    ckpt_root = Path(config.train.save_dir)
    best_path = ckpt_root / "best"
    latest_path = ckpt_root / "latest"

    global_step = 0
    best_val_mrr = float("-inf")
    should_stop = False
    is_phase1 = config.train.curriculum_phase == "phase1"
    early_stop_patience = 5 if is_phase1 else max(1, int(config.train.early_stop_patience))
    early_stop_metric_key = "rewrite_recall20_mean" if is_phase1 else "rewrite_mrr20_mean"
    eval_history: deque[dict[str, float]] = deque(maxlen=early_stop_patience)
    print(f"[early-stop] metric={early_stop_metric_key} patience={early_stop_patience}")

    # 7) 主训练循环：每个 epoch 先按 curriculum phase 重排 query，再按 batch 更新。
    for epoch in range(1, config.train.num_epochs + 1):
        epoch_queries = sample_curriculum_queries(
            train_queries,
            metadata_by_qid,
            phase=config.train.curriculum_phase,
            seed=config.data.seed,
            epoch=epoch,
        )
        for batch in iter_batches(epoch_queries, config.train.batch_size):
            global_step += 1
            metrics = engine.train_step(batch)
            group_query_summaries = metrics.pop("group_query_summaries", [])
            timestamp_utc = datetime.now(timezone.utc).isoformat()

            for summary in group_query_summaries:
                append_jsonl(
                    group_trace_log_path,
                    {
                        "phase": "train_group_trace",
                        "epoch": epoch,
                        "step": global_step,
                        "timestamp_utc": timestamp_utc,
                        **summary,
                    },
                )

            metrics.update({"phase": "train", "epoch": epoch, "step": global_step, "timestamp_utc": timestamp_utc})
            append_jsonl(log_path, metrics)

            print(
                f"[train] step={global_step} loss={metrics['loss']:.4f} "
                f"pg={metrics['loss_pg']:.4f} pg_abs={metrics.get('loss_pg_abs_mean', 0.0):.4f} "
                f"kl={metrics['loss_kl']:.4f} kl_dom={metrics.get('kl_dominance_ratio', 0.0):.3f} "
                f"reward={metrics['reward_mean']:.4f} "
                f"orig_mrr20={metrics.get('orig_mrr20_mean', 0.0):.4f} "
                f"rewrite_mrr20={metrics.get('rewrite_mrr20_mean', 0.0):.4f} "
                f"recall20={metrics.get('recall_mean', 0.0):.4f} "
                f"recall50={metrics.get('recall_dense_mean', 0.0):.4f} "
                f"main_reward={metrics.get('main_reward_mean', 0.0):.4f} "
                f"delta_mrr20_pos_ratio={metrics.get('delta_mrr20_positive_ratio', 0.0):.4f} "
                f"keyword_preserve={metrics.get('keyword_preserve_mean', 0.0):.4f} "
                f"overedit_penalty={metrics.get('overedit_penalty_mean', 0.0):.4f} "
                f"bad_format_penalty={metrics.get('bad_format_penalty_mean', 0.0):.4f} "
                f"unsafe_copy_penalty={metrics.get('unsafe_copy_penalty_mean', 0.0):.4f} "
                f"unique_final_query_mean={metrics.get('unique_final_query_mean', 0.0):.4f} "
                f"rollout_batch_prompts_mean={metrics.get('rollout_batch_prompt_count_mean', 0.0):.1f} "
                f"rollout_batch_prompts_max={metrics.get('rollout_batch_prompt_count_max', 0.0):.0f} "
                f"rollout_batch_calls={metrics.get('rollout_batch_call_count', 0.0):.0f} "
                f"rollout_batch_splits={metrics.get('rollout_batch_fallback_split_count', 0.0):.0f} "
                f"extra_sample_ratio={metrics.get('extra_sample_ratio', 0.0):.4f} "
                f"reward_gap_raw_mean={metrics.get('reward_gap_raw_mean', 0.0):.4f} "
                f"reward_gap_met_ratio={metrics.get('reward_gap_met_ratio', 0.0):.4f} "
                f"max_group_size_hit_ratio={metrics.get('max_group_size_hit_ratio', 0.0):.4f} "
                f"collapsed_group_ratio={metrics.get('collapsed_group_ratio', 0.0):.4f} "
                f"flat_main_reward_group_ratio={metrics.get('flat_main_reward_group_ratio', 0.0):.4f} "
                f"best_reward_hit_best_mrr20_ratio={metrics.get('best_reward_hit_best_mrr20_ratio', 0.0):.4f}"
            )

            if global_step % config.train.eval_every_steps == 0:
                # 周期性验证并保存 adapter；best 以 rewrite_mrr20 为准。
                eval_metrics = evaluate_policy(
                    model,
                    rewarder,
                    val_queries,
                    guardrail_cfg=config.prompt,
                    max_queries=config.data.max_val_queries,
                    max_new_tokens=int(eval_decode["max_new_tokens"]),
                    temperature=float(eval_decode["temperature"]),
                    top_p=float(eval_decode["top_p"]),
                    query_batch_size=int(eval_decode["query_batch_size"]),
                )
                eval_metrics.update(
                    {
                        "phase": "eval",
                        "epoch": epoch,
                        "step": global_step,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
                append_jsonl(log_path, eval_metrics)
                print(
                    f"[eval] step={global_step} "
                    f"val_orig_mrr20={eval_metrics.get('orig_mrr20_mean', 0.0):.4f} "
                    f"val_rewrite_mrr20={eval_metrics.get('rewrite_mrr20_mean', 0.0):.4f} "
                    f"val_reward={eval_metrics['reward_mean']:.4f} "
                    f"val_main_reward={eval_metrics.get('main_reward_mean', 0.0):.4f} "
                    f"val_delta_mrr20_pos_ratio={eval_metrics.get('delta_mrr20_positive_ratio', 0.0):.4f} "
                    f"val_recall50={eval_metrics.get('recall_dense_mean', 0.0):.4f} "
                    f"val_keyword_preserve={eval_metrics.get('keyword_preserve_mean', 0.0):.4f} "
                    f"val_overedit_penalty={eval_metrics.get('overedit_penalty_mean', 0.0):.4f} "
                    f"val_bad_format_penalty={eval_metrics.get('bad_format_penalty_mean', 0.0):.4f} "
                    f"val_unsafe_copy_penalty={eval_metrics.get('unsafe_copy_penalty_mean', 0.0):.4f}"
                )

                model.save_adapter(str(latest_path))
                current_eval_mrr = float(eval_metrics.get("rewrite_mrr20_mean", eval_metrics["mrr_mean"]))
                if current_eval_mrr > best_val_mrr:
                    best_val_mrr = current_eval_mrr
                    model.save_adapter(str(best_path))
                    print(f"[ckpt] best updated: rewrite_mrr20={best_val_mrr:.4f} -> {best_path}")

                eval_history.append(
                    {
                        "early_stop_score": float(
                            eval_metrics.get(
                                early_stop_metric_key,
                                eval_metrics["recall_mean"] if is_phase1 else eval_metrics["mrr_mean"],
                            )
                        ),
                        "rewrite_mrr20_mean": current_eval_mrr,
                        "flat_main_reward_group_ratio": float(metrics.get("flat_main_reward_group_ratio", 0.0)),
                        "kl_dominance_ratio": float(metrics.get("kl_dominance_ratio", 0.0)),
                        "delta_mrr20_positive_ratio": float(metrics.get("delta_mrr20_positive_ratio", 0.0)),
                    }
                )
                if len(eval_history) >= early_stop_patience:
                    # phase1 waits for Recall@20 to stall; phase2 keeps the stricter MRR/flat/KL guard.
                    history_rows = list(eval_history)
                    primary_stalled = all(
                        history_rows[idx]["early_stop_score"] <= history_rows[idx - 1]["early_stop_score"]
                        for idx in range(1, len(history_rows))
                    )
                    flat_stalled = all(
                        history_rows[idx]["flat_main_reward_group_ratio"]
                        >= history_rows[idx - 1]["flat_main_reward_group_ratio"]
                        for idx in range(1, len(history_rows))
                    )
                    kl_bad = all(
                        history_rows[idx]["kl_dominance_ratio"] >= history_rows[idx - 1]["kl_dominance_ratio"]
                        for idx in range(1, len(history_rows))
                    ) and all(
                        history_rows[idx]["delta_mrr20_positive_ratio"]
                        <= history_rows[idx - 1]["delta_mrr20_positive_ratio"]
                        for idx in range(1, len(history_rows))
                    )
                    should_early_stop = primary_stalled if is_phase1 else (primary_stalled or flat_stalled or kl_bad)
                    if should_early_stop:
                        reason = (
                            "rewrite_recall20_stalled"
                            if is_phase1
                            else "rewrite_mrr20_stalled"
                            if primary_stalled
                            else "flat_main_reward_not_improving"
                            if flat_stalled
                            else "kl_up_without_delta_mrr_gain"
                        )
                        print(f"[early-stop] step={global_step} reason={reason}")
                        should_stop = True
                        break

            if config.train.max_steps is not None and global_step >= config.train.max_steps:
                should_stop = True
                break
        if should_stop:
            break

    model.save_adapter(str(latest_path))
    # 8) 训练结束再做一次 final eval，防止最后几个 step 没赶上 eval_every_steps。
    final_eval = evaluate_policy(
        model,
        rewarder,
        val_queries,
        guardrail_cfg=config.prompt,
        max_queries=config.data.max_val_queries,
        max_new_tokens=int(eval_decode["max_new_tokens"]),
        temperature=float(eval_decode["temperature"]),
        top_p=float(eval_decode["top_p"]),
        query_batch_size=int(eval_decode["query_batch_size"]),
    )
    final_rewrite_mrr20 = float(final_eval.get("rewrite_mrr20_mean", final_eval["mrr_mean"]))
    if final_rewrite_mrr20 > best_val_mrr or not best_path.exists():
        model.save_adapter(str(best_path))
        best_val_mrr = final_rewrite_mrr20
        print(f"[ckpt] best updated from final eval: rewrite_mrr20={best_val_mrr:.4f} -> {best_path}")

    print(
        f"[done] final_val_{mrr_label}={final_eval['mrr_mean']:.4f} "
        f"original_val_{mrr_label}={base_original_val['mrr_mean']:.4f} "
        f"delta={final_eval['mrr_mean'] - base_original_val['mrr_mean']:+.4f}"
    )
    print(f"[done] checkpoints: best={best_path}, latest={latest_path}")
    print(f"[done] train log: {log_path}")
    print(f"[done] group trace log: {group_trace_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
