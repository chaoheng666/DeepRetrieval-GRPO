from __future__ import annotations

"""GRPO 训练主入口。

这个脚本把训练编排集中在一个文件，便于快速理解和调试：
1. 读取配置与 CLI 覆盖参数
2. 加载 Pyserini 数据与检索奖励器
3. 构建模型（Actor + Ref）与优化器
4. 执行 GRPO 训练循环
5. 周期评估并保存 best/latest adapter
"""

import argparse
import json
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
from data.loader import QueryExample, load_topics_qrels, maybe_limit, split_queries


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    约定：所有参数均为“可选覆盖项”，若不提供则沿用 app_config 默认值。
    """

    parser = argparse.ArgumentParser(description="Train Qwen2.5-3B query rewriter with custom GRPO.")
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
    parser.add_argument("--adapter-path", type=str, default=None, help="Optional LoRA adapter for warm start.")
    parser.add_argument(
        "--low-mem-mode",
        action="store_true",
        help="Use an aggressive low-memory preset (for 6GB-class GPU quick smoke runs).",
    )
    return parser.parse_args()


def apply_low_mem_mode(config: AppConfig) -> AppConfig:
    """Apply a conservative low-memory preset.

    This preset is designed for machines with very limited VRAM (e.g. 6GB).
    It targets *pipeline validation* rather than final-quality training.
    """

    # 使用更小模型，降低 actor+ref 双模型同时驻留带来的显存压力。
    config.model.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    config.model.load_in_4bit = True
    config.model.lora_r = 8
    config.model.lora_alpha = 16
    config.model.lora_dropout = 0.05

    # 尽量降低每步显存占用。
    config.train.batch_size = 1
    # NOTE: group_size must be >=2 for GRPO to produce non-zero advantage.
    config.train.group_size = 2
    config.train.max_new_tokens = 16
    config.train.temperature = 0.9
    config.train.top_p = 0.95
    config.train.eval_every_steps = 10
    config.train.max_steps = 20
    config.train.num_epochs = 1

    # 缩小数据规模以快速跑通端到端链路。
    config.data.max_train_queries = 64
    config.data.max_val_queries = 32
    # 使用 slim 预编译索引，首次下载更快（约 0.5GB 级别）。
    config.data.prebuilt_index = "msmarco-v1-passage-slim"
    # 低显存快速实验里把奖励窗口放大到 top-20，增加命中概率。
    config.reward.topk = 20
    config.reward.overlap_weight = 0.3

    # Keep outputs separate from normal runs.
    config.train.save_dir = "artifacts_lowmem/checkpoints"
    config.train.log_path = "artifacts_lowmem/train_log.jsonl"
    return config


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """将 CLI 非空字段覆盖到默认配置对象上。"""

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
    """向 JSONL 日志追加一条记录。"""

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
    """评估当前 actor 策略在验证集上的表现。

    返回值包含：
    - reward_mean: 平均总奖励（mrr - penalty）
    - mrr_mean: 平均 MRR@10
    - penalty_mean: 平均惩罚
    - count: 实际评估样本数
    """

    eval_queries = list(queries[:max_queries]) if max_queries is not None else list(queries)
    rewards: list[float] = []
    mrr_scores: list[float] = []
    penalties: list[float] = []

    for query in eval_queries:
        # 评估时用贪心解码，减少随机性干扰。
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
    """原始 query 基线评估（不做重写）。"""

    eval_queries = list(queries[:max_queries]) if max_queries is not None else list(queries)
    mrr_scores = [rewarder.score(q.qid, q.text, source_query=q.text).mrr for q in eval_queries]
    return {
        "mrr_mean": fmean(mrr_scores) if mrr_scores else 0.0,
        "count": float(len(eval_queries)),
    }


def main() -> int:
    """训练主流程。"""

    args = parse_args()
    config = get_default_config()
    if args.low_mem_mode:
        config = apply_low_mem_mode(config)
    # CLI explicit values should still win over low-mem preset.
    config = apply_overrides(config, args)

    # 低显存模式下，如果 CUDA 不可用，自动切到 CPU 兼容配置，避免 4bit 加载失败。
    if args.low_mem_mode and not torch.cuda.is_available():
        config.model.load_in_4bit = False
        config.model.actor_device_map = "cpu"
        config.model.ref_device_map = "cpu"

    # GRPO 组内标准化需要至少两个样本。否则 advantage 恒为 0，loss_pg 失效。
    if config.train.group_size < 2:
        print(
            f"[warn] group_size={config.train.group_size} is invalid for GRPO advantage normalization; "
            "auto-adjusting to 2."
        )
        config.train.group_size = 2

    ensure_runtime_dirs(config)
    set_seed(config.data.seed)

    print("[config]")
    print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
    if args.low_mem_mode:
        print("[mode] low-mem preset enabled (intended for smoke tests on limited VRAM).")
        if not torch.cuda.is_available():
            print("[mode] CUDA is unavailable -> switched to CPU-compatible loading (much slower).")

    # 数据来源严格使用 Pyserini 预编译 topics/qrels。
    queries, qrels = load_topics_qrels(config.data.topic_name)
    train_queries, val_queries = split_queries(queries, train_ratio=config.data.train_ratio, seed=config.data.seed)
    train_queries = maybe_limit(train_queries, config.data.max_train_queries)
    val_queries = maybe_limit(val_queries, config.data.max_val_queries)

    print(f"[data] train_queries={len(train_queries)}, val_queries={len(val_queries)}, qrels_qids={len(qrels)}")

    # 奖励器内部持有 Pyserini 预编译索引检索器，负责 MRR 计算。
    rewarder = Rewarder(
        qrels=qrels,
        prebuilt_index=config.data.prebuilt_index,
        reward_cfg=config.reward,
    )

    base_original_val = evaluate_original(rewarder, val_queries, max_queries=config.data.max_val_queries)
    print(f"[baseline] original_val_mrr@10={base_original_val['mrr_mean']:.4f}")

    # 训练时同时加载 actor（可训练）和 ref（冻结）以支持 KL 项。
    model = ModelWrapper(
        model_cfg=config.model,
        prompt_cfg=config.prompt,
        train_mode=True,
        enable_lora=True,
        load_ref_model=True,
        adapter_path=args.adapter_path,
    )
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    # GRPO 引擎封装了采样、优势归一化、loss 计算与反传更新。
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

    log_path = Path(config.train.log_path)
    ckpt_root = Path(config.train.save_dir)
    best_path = ckpt_root / "best"
    latest_path = ckpt_root / "latest"

    global_step = 0
    best_val_mrr = float("-inf")
    should_stop = False

    for epoch in range(1, config.train.num_epochs + 1):
        # 每个 epoch 重新乱序，seed 采用可复现的偏移策略。
        random.Random(config.data.seed + epoch).shuffle(train_queries)
        for batch in iter_batches(train_queries, config.train.batch_size):
            global_step += 1
            # 执行一次参数更新。
            metrics = engine.train_step(batch)
            metrics.update(
                {
                    "phase": "train",
                    "epoch": epoch,
                    "step": global_step,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            append_jsonl(log_path, metrics)

            print(
                f"[train] step={global_step} loss={metrics['loss']:.4f} "
                f"pg={metrics['loss_pg']:.4f} kl={metrics['loss_kl']:.4f} "
                f"reward={metrics['reward_mean']:.4f} mrr={metrics['mrr_mean']:.4f} "
                f"overlap={metrics.get('overlap_mean', 0.0):.4f}"
            )

            # 周期性做验证并保存 checkpoint。
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

                # latest 始终刷新，便于中断后恢复。
                model.save_adapter(str(latest_path))
                if eval_metrics["mrr_mean"] > best_val_mrr:
                    best_val_mrr = eval_metrics["mrr_mean"]
                    # best 仅在验证 MRR 提升时更新。
                    model.save_adapter(str(best_path))
                    print(f"[ckpt] best updated: mrr={best_val_mrr:.4f} -> {best_path}")

            if config.train.max_steps is not None and global_step >= config.train.max_steps:
                should_stop = True
                break
        if should_stop:
            break

    model.save_adapter(str(latest_path))
    # 训练结束后做一次最终验证汇总。
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
