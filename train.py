from __future__ import annotations

"""GRPO 训练主入口。

主要职责：
1. 解析命令行并合并默认配置/覆盖配置
2. 构建数据、奖励器、Actor/Ref 模型与优化器
3. 执行训练循环并周期评估
4. 写入两类日志：训练指标日志 + group 采样明细日志
"""

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

import torch

from app_config import AppConfig, ensure_runtime_dirs, get_default_config
from core.grpo_engine import GRPOEngine
from core.model_wrapper import ModelWrapper
from core.reward_func import Rewarder
from data.loader import QueryExample, maybe_limit, load_topics_qrels, split_queries


def parse_args() -> argparse.Namespace:
    """解析训练命令行参数。"""

    parser = argparse.ArgumentParser(description="Train Qwen2.5 query rewriter with custom GRPO.")
    parser.add_argument("--model-name", type=str, default=None, help="Override base model name.")
    parser.add_argument("--topic-name", type=str, default=None)
    parser.add_argument("--prebuilt-index", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--clip-range", type=float, default=None)
    parser.add_argument("--kl-beta", type=float, default=None)
    parser.add_argument(
        "--disable-4bit",
        action="store_true",
        help="Disable 4-bit quantization for actor model loading.",
    )
    parser.add_argument(
        "--strict-tokenizer-model-match",
        action="store_true",
        help="Fail fast if tokenizer/model or adapter base-model mismatch is detected.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--reward-topk", type=int, default=None, help="MRR@k reward cutoff, e.g. 10/20/50.")
    parser.add_argument("--reward-overlap-weight", type=float, default=None, help="Weight for lexical-overlap shaping reward.")
    parser.add_argument("--reward-mrr-weight", type=float, default=None, help="Weight for MRR reward term.")
    parser.add_argument("--eval-every-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-queries", type=int, default=None)
    parser.add_argument("--max-val-queries", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--group-trace-log-path", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, default=None, help="Optional LoRA adapter for warm start.")
    parser.add_argument(
        "--print-best-query",
        action="store_true",
        help="Print each input query and the best rewritten query (by reward) in every training step.",
    )
    parser.add_argument(
        "--low-mem-mode",
        action="store_true",
        help="Use an aggressive low-memory preset (for 6GB-class GPU smoke runs).",
    )
    return parser.parse_args()


def apply_low_mem_mode(config: AppConfig) -> AppConfig:
    """应用低显存预设（以 0.5B 快速跑通链路为目标）。"""

    # 低显存模式默认切到更小模型。
    config.model.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    config.model.load_in_4bit = True
    config.model.lora_r = 8
    config.model.lora_alpha = 16
    config.model.lora_dropout = 0.05

    # 减少单步显存占用与训练时长。
    config.train.batch_size = 1
    config.train.group_size = 8
    config.train.max_new_tokens = 20
    config.train.temperature = 0.1
    config.train.top_p = 0.95
    config.train.eval_every_steps = 10
    config.train.max_steps = 50
    config.train.num_epochs = 1

    # 缩小样本规模并使用 slim 索引，提升启动速度。
    config.data.max_train_queries = 64
    config.data.max_val_queries = 32
    config.data.prebuilt_index = "msmarco-v1-passage-slim"
    config.reward.topk = 50
    config.reward.overlap_weight = 0.3

    # 将低显存实验输出隔离到单独目录。
    config.train.save_dir = "artifacts_lowmem/checkpoints"
    config.train.log_path = "artifacts_lowmem/train_log.jsonl"
    config.train.group_trace_log_path = "artifacts_lowmem/group_trace_log.jsonl"
    return config


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """应用 CLI 覆盖参数。"""

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
    if args.learning_rate is not None:
        config.train.learning_rate = args.learning_rate
    if args.clip_range is not None:
        config.train.clip_range = args.clip_range
    if args.kl_beta is not None:
        config.train.kl_beta = args.kl_beta
    if args.disable_4bit:
        config.model.load_in_4bit = False
    if args.max_new_tokens is not None:
        config.train.max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        config.train.temperature = args.temperature
    if args.top_p is not None:
        config.train.top_p = args.top_p
    if args.reward_topk is not None:
        config.reward.topk = args.reward_topk
    if args.reward_overlap_weight is not None:
        config.reward.overlap_weight = args.reward_overlap_weight
    if args.reward_mrr_weight is not None:
        config.reward.mrr_weight = args.reward_mrr_weight
    if args.eval_every_steps is not None:
        config.train.eval_every_steps = args.eval_every_steps
    if args.max_steps is not None:
        config.train.max_steps = args.max_steps
    if args.save_dir is not None:
        config.train.save_dir = args.save_dir
    if args.log_path is not None:
        config.train.log_path = args.log_path
    if args.group_trace_log_path is not None:
        config.train.group_trace_log_path = args.group_trace_log_path

    return config


def apply_runtime_mode_adjustments(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """应用运行时安全调整（在预设与 CLI 合并之后）。"""

    # 低显存模式下若 CUDA 不可用，自动切到 CPU 兼容加载。
    if args.low_mem_mode and not torch.cuda.is_available():
        config.model.load_in_4bit = False
        config.model.actor_device_map = "cpu"
        config.model.ref_device_map = "cpu"

    # 全精度调试时给出更稳妥的 CUDA 分配策略。
    if args.disable_4bit:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # GRPO 组内标准化至少需要 2 个样本。
    if config.train.group_size < 2:
        print(
            f"[warn] group_size={config.train.group_size} is invalid for GRPO advantage normalization; "
            "auto-adjusting to 2."
        )
        config.train.group_size = 2

    return config


def set_seed(seed: int) -> None:
    """设置随机种子，尽量保证可复现。"""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def iter_batches(items: Sequence[QueryExample], batch_size: int) -> Iterable[list[QueryExample]]:
    """按 batch_size 生成 mini-batch。"""

    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def append_jsonl(path: Path, payload: dict) -> None:
    """向 JSONL 文件追加一条记录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def evaluate_policy(
    model: ModelWrapper,
    rewarder: Rewarder,
    queries: Sequence[QueryExample],
    *,
    max_queries: int | None,
    max_new_tokens: int,
) -> dict[str, float]:
    """评估当前 actor 策略在验证集上的效果。"""

    eval_queries = list(queries[:max_queries]) if max_queries is not None else list(queries)
    rewards: list[float] = []
    mrr_scores: list[float] = []
    penalties: list[float] = []

    for query in eval_queries:
        rewritten = model.generate_rewrite(
            query.text,
            policy="actor",
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        score = rewarder.score(query.qid, rewritten, source_query=query.text)
        rewards.append(score.total)
        mrr_scores.append(score.mrr)
        penalties.append(score.penalty)

    return {
        "reward_mean": fmean(rewards) if rewards else 0.0,
        "mrr_mean": fmean(mrr_scores) if mrr_scores else 0.0,
        "penalty_mean": fmean(penalties) if penalties else 0.0,
        "count": float(len(eval_queries)),
    }


def evaluate_original(
    rewarder: Rewarder,
    queries: Sequence[QueryExample],
    *,
    max_queries: int | None,
) -> dict[str, float]:
    """评估原始 query（不重写）基线。"""

    eval_queries = list(queries[:max_queries]) if max_queries is not None else list(queries)
    mrr_scores = [rewarder.score(q.qid, q.text, source_query=q.text).mrr for q in eval_queries]
    return {
        "mrr_mean": fmean(mrr_scores) if mrr_scores else 0.0,
        "count": float(len(eval_queries)),
    }


def main() -> int:
    """训练主流程。"""

    # 1) 合并默认配置 + 低显存预设 + CLI 覆盖 + 运行时调整。
    args = parse_args()
    config = get_default_config()
    if args.low_mem_mode:
        config = apply_low_mem_mode(config)
    config = apply_overrides(config, args)
    config = apply_runtime_mode_adjustments(config, args)

    # 2) 准备运行目录和随机种子。
    ensure_runtime_dirs(config)
    set_seed(config.data.seed)

    print("[config]")
    print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
    if args.low_mem_mode:
        print("[mode] low-mem preset enabled (intended for smoke tests on limited VRAM).")
        if not torch.cuda.is_available():
            print("[mode] CUDA is unavailable -> switched to CPU-compatible loading (much slower).")

    # 3) 加载数据并切分 train/val。
    queries, qrels = load_topics_qrels(config.data.topic_name)
    train_queries, val_queries = split_queries(queries, train_ratio=config.data.train_ratio, seed=config.data.seed)
    # train_queries = maybe_limit(train_queries, config.data.max_train_queries)
    # val_queries = maybe_limit(val_queries, config.data.max_val_queries)

    print(f"[data] train_queries={len(train_queries)}, val_queries={len(val_queries)}, qrels_qids={len(qrels)}")

    # 4) 初始化奖励器与模型组件。
    rewarder = Rewarder(
        qrels=qrels,
        prebuilt_index=config.data.prebuilt_index,
        reward_cfg=config.reward,
    )

    # 5) 基线评估（原始 query）。
    base_original_val = evaluate_original(rewarder, val_queries, max_queries=config.data.max_val_queries)
    print(f"[baseline] original_val_mrr@10={base_original_val['mrr_mean']:.4f}")

    # 6) 构建训练引擎。
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
    )

    # 7) 日志与 checkpoint 路径。
    log_path = Path(config.train.log_path)
    group_trace_log_path = Path(config.train.group_trace_log_path)
    ckpt_root = Path(config.train.save_dir)
    best_path = ckpt_root / "best"
    latest_path = ckpt_root / "latest"

    # 8) 训练循环：每 step 更新参数、写日志、按周期评估与保存 best/latest。
    global_step = 0
    best_val_mrr = float("-inf")
    should_stop = False

    for epoch in range(1, config.train.num_epochs + 1):
        random.Random(config.data.seed + epoch).shuffle(train_queries)
        for batch in iter_batches(train_queries, config.train.batch_size):
            global_step += 1
            metrics = engine.train_step(batch, collect_best_queries=args.print_best_query)
            best_query_pairs = metrics.pop("best_query_pairs", [])
            group_query_summaries = metrics.pop("group_query_summaries", [])
            timestamp_utc = datetime.now(timezone.utc).isoformat()

            # 独立落盘：每个 query 一条 group 采样明细。
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

            metrics.update(
                {
                    "phase": "train",
                    "epoch": epoch,
                    "step": global_step,
                    "timestamp_utc": timestamp_utc,
                }
            )
            append_jsonl(log_path, metrics)

            print(
                f"[train] step={global_step} loss={metrics['loss']:.4f} "
                f"pg={metrics['loss_pg']:.4f} kl={metrics['loss_kl']:.4f} "
                f"reward={metrics['reward_mean']:.4f} mrr={metrics['mrr_mean']:.4f} "
                f"overlap={metrics.get('overlap_mean', 0.0):.4f} "
                f"unreadable={metrics.get('unreadable_ratio_mean', 0.0):.4f}"
            )
            if args.print_best_query and best_query_pairs:
                for pair in best_query_pairs:
                    print(
                        f"[best-query] qid={pair['qid']} "
                        f"input={pair['input_query']} "
                        f"best={pair['best_rewritten_query']} "
                        f"reward={pair['best_reward']:.4f}"
                    )

            if global_step % config.train.eval_every_steps == 0:
                eval_metrics = evaluate_policy(
                    model,
                    rewarder,
                    val_queries,
                    max_queries=config.data.max_val_queries,
                    max_new_tokens=config.train.max_new_tokens,
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
                    f"[eval] step={global_step} val_mrr={eval_metrics['mrr_mean']:.4f} "
                    f"val_reward={eval_metrics['reward_mean']:.4f}"
                )

                model.save_adapter(str(latest_path))
                if eval_metrics["mrr_mean"] > best_val_mrr:
                    best_val_mrr = eval_metrics["mrr_mean"]
                    model.save_adapter(str(best_path))
                    print(f"[ckpt] best updated: mrr={best_val_mrr:.4f} -> {best_path}")

            if config.train.max_steps is not None and global_step >= config.train.max_steps:
                should_stop = True
                break
        if should_stop:
            break

    # 9) 训练结束后做最终评估与输出汇总。
    model.save_adapter(str(latest_path))
    final_eval = evaluate_policy(
        model,
        rewarder,
        val_queries,
        max_queries=config.data.max_val_queries,
        max_new_tokens=config.train.max_new_tokens,
    )
    print(
        f"[done] final_val_mrr={final_eval['mrr_mean']:.4f} "
        f"original_val_mrr={base_original_val['mrr_mean']:.4f} "
        f"delta={final_eval['mrr_mean'] - base_original_val['mrr_mean']:+.4f}"
    )
    print(f"[done] checkpoints: best={best_path}, latest={latest_path}")
    print(f"[done] train log: {log_path}")
    print(f"[done] group trace log: {group_trace_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
