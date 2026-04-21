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
from collections import deque
from typing import Iterable, Sequence

import torch

from app_config import AppConfig, apply_reward_mode_prompt_defaults, ensure_runtime_dirs, get_default_config
from core.grpo_engine import GRPOEngine
from core.model_wrapper import ModelWrapper
from core.reward_func import Rewarder, is_retrieval_ready_query, stabilize_generated_rewrite, summarize_reward_breakdowns
from data.curriculum import ensure_curriculum_metadata, filter_curriculum_train_queries, sample_curriculum_queries
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
    parser.add_argument("--max-group-size", type=int, default=None)
    parser.add_argument(
        "--parallel-group-generate",
        action="store_true",
        help="Enable one-shot group sampling via num_return_sequences (may be unstable on some CUDA/transformers stacks).",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--clip-range", type=float, default=None)
    parser.add_argument("--kl-beta", type=float, default=None)
    parser.add_argument(
        "--ref-precision-mode",
        type=str,
        choices=("auto", "full", "4bit"),
        default=None,
        help="Reference model precision mode used for the KL anchor.",
    )
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
    parser.add_argument(
        "--actor-chunk-size",
        type=int,
        default=None,
        help="Actor logprob recompute chunk size during GRPO backward.",
    )
    parser.add_argument(
        "--projection-chunk-size",
        type=int,
        default=None,
        help="lm_head projection chunk size during batched logprob recomputation.",
    )
    parser.add_argument("--reward-mrr-k", type=int, default=None, help="MRR@k reward cutoff, e.g. 50.")
    parser.add_argument("--reward-recall-k", type=int, default=None, help="Recall@k reward cutoff, e.g. 50.")
    parser.add_argument("--reward-recall-dense-k", type=int, default=None, help="Dense Recall@k reward cutoff, e.g. 100.")
    parser.add_argument(
        "--reward-mode",
        type=str,
        choices=("legacy", "top20_delta"),
        default=None,
        help="Reward composition mode.",
    )
    parser.add_argument("--search-threads", type=int, default=None, help="Pyserini batch_search thread count.")
    parser.add_argument("--reward-w-mrr", type=float, default=None, help="Weight for MRR reward term.")
    parser.add_argument("--reward-w-recall", type=float, default=None, help="Weight for Recall reward term.")
    parser.add_argument("--reward-w-recall-dense", type=float, default=None, help="Weight for dense Recall reward term.")
    parser.add_argument("--reward-w-rank-bonus", type=float, default=None, help="Weight for rank-bonus reward term.")
    parser.add_argument("--reward-w-term-preserve", type=float, default=None, help="Weight for term-preserve reward term.")
    parser.add_argument("--reward-w-length-score", type=float, default=None, help="Weight for length-score reward term.")
    parser.add_argument("--reward-w-clean-format", type=float, default=None, help="Weight for clean-format reward term.")
    parser.add_argument("--reward-w-bad-format", type=float, default=None, help="Weight for bad-format penalty term.")
    parser.add_argument("--reward-w-unsafe-copy", type=float, default=None, help="Weight for unsafe-copy penalty term.")
    parser.add_argument("--reward-w-overedit", type=float, default=None, help="Weight for overedit penalty term.")
    parser.add_argument("--overedit-tau", type=float, default=None, help="Keyword preserve threshold before overedit penalty applies.")
    parser.add_argument("--recall-drop-lambda", type=float, default=None, help="Recall@20 drop penalty scale for top20_delta mode.")
    parser.add_argument("--anchor-bonus-value", type=float, default=None, help="Bonus added when rewrite is no worse than original on MRR@20 and Recall@20.")
    parser.add_argument("--length-score-min-terms", type=int, default=None, help="Token count where length score starts above zero.")
    parser.add_argument("--length-score-ideal-min-terms", type=int, default=None, help="Lower bound of the ideal token-count plateau.")
    parser.add_argument("--length-score-ideal-max-terms", type=int, default=None, help="Upper bound of the ideal token-count plateau.")
    parser.add_argument("--length-score-max-terms", type=int, default=None, help="Token count where length score returns to zero.")
    parser.add_argument("--format-max-tokens", type=int, default=None, help="Hard cap for token count in strict format check.")
    parser.add_argument(
        "--format-min-english-ratio",
        type=float,
        default=None,
        help="Minimum English-letter ratio in strict format check.",
    )
    parser.add_argument(
        "--format-max-unreadable-ratio",
        type=float,
        default=None,
        help="Maximum unreadable-char ratio in strict format check.",
    )
    parser.add_argument("--bad-format-cap", type=float, default=None, help="Maximum accumulated bad-format penalty.")
    parser.add_argument("--eval-every-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-queries", type=int, default=None)
    parser.add_argument("--max-val-queries", type=int, default=None)
    parser.add_argument("--curriculum-enable", action="store_true", help="Enable A/B/C bucket curriculum sampling.")
    parser.add_argument(
        "--curriculum-phase",
        type=str,
        choices=("phase1", "phase2"),
        default=None,
        help="Curriculum phase controls bucket mix and phase-specific metadata reuse.",
    )
    parser.add_argument(
        "--curriculum-metadata-path",
        type=str,
        default=None,
        help="Optional cache path for curriculum query metadata JSONL.",
    )
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
    config.model.model_name = "Qwen/Qwen3.5-0.8B"
    config.model.load_in_4bit = True
    config.model.lora_r = 8
    config.model.lora_alpha = 16
    config.model.lora_dropout = 0.05

    # 减少单步显存占用与训练时长。
    config.train.batch_size = 1
    config.train.group_size = 8
    config.train.max_group_size = 24
    config.train.max_new_tokens = 20
    config.train.temperature = 0.1
    config.train.top_p = 0.95
    config.train.reward_gap_threshold = 0.10
    config.train.gap_sampling_temperature_delta = 0.15
    config.train.actor_chunk_size = 1
    config.model.projection_chunk_size = 32
    config.train.eval_every_steps = 10
    config.train.max_steps = 50
    config.train.num_epochs = 1

    # 缩小样本规模并使用 slim 索引，提升启动速度。
    config.data.max_train_queries = 64
    config.data.max_val_queries = 32
    config.data.prebuilt_index = "msmarco-v1-passage-slim"
    config.reward.recall_k = 50

    # 将低显存实验输出隔离到单独目录。
    config.train.save_dir = "train_and_eval_data_model/artifacts_lowmem_train/checkpoints"
    config.train.log_path = "train_and_eval_data_model/artifacts_lowmem_train/train_log.jsonl"
    config.train.group_trace_log_path = "train_and_eval_data_model/artifacts_lowmem_train/group_trace_log.jsonl"
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
    if args.curriculum_enable:
        config.train.curriculum_enable = True
    if args.curriculum_phase is not None:
        config.train.curriculum_phase = args.curriculum_phase
    if args.curriculum_metadata_path is not None:
        config.train.curriculum_metadata_path = args.curriculum_metadata_path

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
    if args.ref_precision_mode is not None:
        config.model.ref_precision_mode = args.ref_precision_mode
    if args.disable_4bit:
        config.model.load_in_4bit = False
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
    if args.reward_mode is not None:
        config.reward.reward_mode = args.reward_mode
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
    if args.recall_drop_lambda is not None:
        config.reward.recall_drop_lambda = args.recall_drop_lambda
    if args.anchor_bonus_value is not None:
        config.reward.anchor_bonus_value = args.anchor_bonus_value
    if args.length_score_min_terms is not None:
        config.reward.length_score_min_terms = args.length_score_min_terms
    if args.length_score_ideal_min_terms is not None:
        config.reward.length_score_ideal_min_terms = args.length_score_ideal_min_terms
    if args.length_score_ideal_max_terms is not None:
        config.reward.length_score_ideal_max_terms = args.length_score_ideal_max_terms
    if args.length_score_max_terms is not None:
        config.reward.length_score_max_terms = args.length_score_max_terms
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

    # CUDA 训练时默认启用更稳妥的内存分配策略，降低碎片化导致的假性 OOM。
    if torch.cuda.is_available():
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # 24G 级别显卡上，4bit actor + full ref 容易在首个 KL/logprob 前向时 OOM。
    # 对 auto 模式做更保守的运行时收紧，优先保证训练能稳定跑起来。
    if (
        torch.cuda.is_available()
        and config.model.load_in_4bit
        and str(config.model.ref_precision_mode).strip().lower() == "auto"
    ):
        try:
            total_gib = torch.cuda.get_device_properties(0).total_memory / float(1024**3)
        except Exception:
            total_gib = None
        if total_gib is not None and total_gib <= 24.5:
            print(
                f"[mode] detected ~{total_gib:.1f} GiB GPU with 4-bit actor training; "
                "forcing ref_precision_mode=4bit for stability."
            )
            config.model.ref_precision_mode = "4bit"

    if not 0.0 < config.data.train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {config.data.train_ratio}")

    if config.train.num_epochs < 1:
        print(f"[warn] num_epochs={config.train.num_epochs} is invalid; auto-adjusting to 1.")
        config.train.num_epochs = 1

    if config.train.batch_size < 1:
        print(f"[warn] batch_size={config.train.batch_size} is invalid; auto-adjusting to 1.")
        config.train.batch_size = 1

    if config.train.eval_every_steps < 1:
        print(
            f"[warn] eval_every_steps={config.train.eval_every_steps} is invalid; "
            "auto-adjusting to 1."
        )
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
        print(
            f"[warn] max_train_queries={config.data.max_train_queries} is invalid; "
            "auto-adjusting to 0."
        )
        config.data.max_train_queries = 0

    if config.data.max_val_queries is not None and config.data.max_val_queries < 0:
        print(
            f"[warn] max_val_queries={config.data.max_val_queries} is invalid; "
            "auto-adjusting to 0."
        )
        config.data.max_val_queries = 0

    if config.train.max_steps is not None and config.train.max_steps < 1:
        print(f"[warn] max_steps={config.train.max_steps} is invalid; auto-adjusting to 1.")
        config.train.max_steps = 1

    # GRPO 组内标准化至少需要 2 个样本。
    if config.train.group_size < 2:
        print(
            f"[warn] group_size={config.train.group_size} is invalid for GRPO advantage normalization; "
            "auto-adjusting to 2."
        )
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
        print(
            f"[warn] projection_chunk_size={config.model.projection_chunk_size} is invalid; auto-adjusting to 1."
        )
        config.model.projection_chunk_size = 1

    if config.train.reward_gap_threshold < 0.0:
        print(
            f"[warn] reward_gap_threshold={config.train.reward_gap_threshold} is invalid; "
            "auto-adjusting to 0.0."
        )
        config.train.reward_gap_threshold = 0.0

    if config.train.gap_sampling_temperature_delta < 0.0:
        print(
            f"[warn] gap_sampling_temperature_delta={config.train.gap_sampling_temperature_delta} is invalid; "
            "auto-adjusting to 0.0."
        )
        config.train.gap_sampling_temperature_delta = 0.0

    if config.train.group_temperature_stride < 0.0:
        print(
            f"[warn] group_temperature_stride={config.train.group_temperature_stride} is invalid; "
            "auto-adjusting to 0.0."
        )
        config.train.group_temperature_stride = 0.0

    if config.train.group_top_p_stride < 0.0:
        print(
            f"[warn] group_top_p_stride={config.train.group_top_p_stride} is invalid; "
            "auto-adjusting to 0.0."
        )
        config.train.group_top_p_stride = 0.0

    if config.train.min_unique_final_queries < 1:
        print(
            f"[warn] min_unique_final_queries={config.train.min_unique_final_queries} is invalid; "
            "auto-adjusting to 1."
        )
        config.train.min_unique_final_queries = 1

    if config.reward.length_score_min_terms < 0:
        print(
            f"[warn] length_score_min_terms={config.reward.length_score_min_terms} is invalid; "
            "auto-adjusting to 0."
        )
        config.reward.length_score_min_terms = 0
    if config.reward.length_score_ideal_min_terms < config.reward.length_score_min_terms:
        print(
            f"[warn] length_score_ideal_min_terms={config.reward.length_score_ideal_min_terms} is smaller than "
            f"length_score_min_terms={config.reward.length_score_min_terms}; auto-adjusting."
        )
        config.reward.length_score_ideal_min_terms = config.reward.length_score_min_terms
    if config.reward.length_score_ideal_max_terms < config.reward.length_score_ideal_min_terms:
        print(
            f"[warn] length_score_ideal_max_terms={config.reward.length_score_ideal_max_terms} is smaller than "
            f"length_score_ideal_min_terms={config.reward.length_score_ideal_min_terms}; auto-adjusting."
        )
        config.reward.length_score_ideal_max_terms = config.reward.length_score_ideal_min_terms
    if config.reward.length_score_max_terms < config.reward.length_score_ideal_max_terms:
        print(
            f"[warn] length_score_max_terms={config.reward.length_score_max_terms} is smaller than "
            f"length_score_ideal_max_terms={config.reward.length_score_ideal_max_terms}; auto-adjusting."
        )
        config.reward.length_score_max_terms = config.reward.length_score_ideal_max_terms
    if config.reward.bad_format_cap < 0.0:
        print(f"[warn] bad_format_cap={config.reward.bad_format_cap} is invalid; auto-adjusting to 0.0.")
        config.reward.bad_format_cap = 0.0
    if config.reward.recall_drop_lambda < 0.0:
        print(f"[warn] recall_drop_lambda={config.reward.recall_drop_lambda} is invalid; auto-adjusting to 0.0.")
        config.reward.recall_drop_lambda = 0.0
    if config.reward.anchor_bonus_value < 0.0:
        print(f"[warn] anchor_bonus_value={config.reward.anchor_bonus_value} is invalid; auto-adjusting to 0.0.")
        config.reward.anchor_bonus_value = 0.0

    if config.train.max_regen_rounds < 0:
        print(
            f"[warn] max_regen_rounds={config.train.max_regen_rounds} is invalid; "
            "auto-adjusting to 0."
        )
        config.train.max_regen_rounds = 0

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


def filter_retrieval_ready_train_queries(queries: Sequence[QueryExample]) -> tuple[list[QueryExample], int]:
    """Skip already-compact BM25-style keyword queries during training."""

    filtered = [query for query in queries if not is_retrieval_ready_query(query.text)]
    skipped = len(queries) - len(filtered)
    if filtered:
        return filtered, skipped
    return list(queries), 0


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
    """评估当前 actor 策略在验证集上的效果。"""

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
                scored_values.append(
                    rewarder.score(query.qid, stabilized.final_query, source_query=query.text)
                )
    finally:
        if actor_model is not None and previous_mode is not None:
            actor_model.train(previous_mode)

    return summarize_reward_breakdowns(scored_values)


def resolve_eval_decode_settings(config: AppConfig, args: argparse.Namespace) -> dict[str, float | int]:
    """Resolve eval decode settings, defaulting to the active training decode config."""

    return {
        "max_new_tokens": int(
            getattr(args, "eval_max_new_tokens", None)
            if getattr(args, "eval_max_new_tokens", None) is not None
            else config.train.max_new_tokens
        ),
        "temperature": float(
            getattr(args, "eval_temperature", None)
            if getattr(args, "eval_temperature", None) is not None
            else config.train.temperature
        ),
        "top_p": float(
            getattr(args, "eval_top_p", None) if getattr(args, "eval_top_p", None) is not None else config.train.top_p
        ),
        "query_batch_size": max(
            1,
            int(
                getattr(args, "eval_query_batch_size", None)
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
    """评估原始 query（不重写）基线。"""

    eval_queries = list(queries[:max_queries]) if max_queries is not None else list(queries)
    scored_values = [rewarder.score(q.qid, q.text, source_query=q.text) for q in eval_queries]
    return summarize_reward_breakdowns(scored_values)


def main() -> int:
    """训练主流程。"""

    # 1) 合并默认配置 + 低显存预设 + CLI 覆盖 + 运行时调整。
    args = parse_args()
    config = get_default_config()
    if args.low_mem_mode:
        config = apply_low_mem_mode(config)
    config = apply_overrides(config, args)
    config = apply_runtime_mode_adjustments(config, args)
    config = apply_reward_mode_prompt_defaults(config)

    # 2) 准备运行目录和随机种子。
    ensure_runtime_dirs(config)
    set_seed(config.data.seed)

    print("[config]")
    print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
    if args.low_mem_mode:
        print("[mode] low-mem preset enabled (intended for smoke tests on limited VRAM).")
        if not torch.cuda.is_available():
            print("[mode] CUDA is unavailable -> switched to CPU-compatible loading (much slower).")

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

    # 3) 加载数据并切分 train/val。
    queries, qrels = load_topics_qrels(config.data.topic_name)
    train_queries, val_queries = split_queries(queries, train_ratio=config.data.train_ratio, seed=config.data.seed)
    train_queries = maybe_limit(train_queries, config.data.max_train_queries)
    val_queries = maybe_limit(val_queries, config.data.max_val_queries)
    skipped_retrieval_ready = 0
    rewarder = Rewarder(
        qrels=qrels,
        prebuilt_index=config.data.prebuilt_index,
        reward_cfg=config.reward,
    )
    curriculum_bucket_counts: dict[str, int] | None = None
    if config.train.curriculum_enable:
        metadata_path = Path(
            config.train.curriculum_metadata_path
            or (Path(config.train.save_dir).resolve().parent / "curriculum_query_metadata.jsonl")
        )
        config.train.curriculum_metadata_path = str(metadata_path)
        metadata_by_qid = ensure_curriculum_metadata(
            train_queries,
            rewarder,
            metadata_path=metadata_path,
        )
        train_queries, curriculum_bucket_counts = filter_curriculum_train_queries(train_queries, metadata_by_qid)
        print(
            f"[curriculum] enabled phase={config.train.curriculum_phase} "
            f"metadata={metadata_path} bucket_counts={curriculum_bucket_counts}"
        )
    else:
        train_queries, skipped_retrieval_ready = filter_retrieval_ready_train_queries(train_queries)
        metadata_by_qid = {}

    print(
        f"[data] train_queries={len(train_queries)}, val_queries={len(val_queries)}, "
        f"qrels_qids={len(qrels)}, filtered_retrieval_ready_train_queries={skipped_retrieval_ready}, "
        f"curriculum_enabled={config.train.curriculum_enable}"
    )

    # 4) 初始化奖励器与模型组件。
    mrr_label = f"MRR@{config.reward.mrr_k}"

    # 5) 基线评估（原始 query）。
    base_original_val = evaluate_original(rewarder, val_queries, max_queries=config.data.max_val_queries)
    print(f"[baseline] original_val_{mrr_label}={base_original_val['mrr_mean']:.4f}")

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
        group_temperature_stride=config.train.group_temperature_stride,
        group_top_p_stride=config.train.group_top_p_stride,
        max_group_size=config.train.max_group_size,
        min_unique_final_queries=config.train.min_unique_final_queries,
        max_regen_rounds=config.train.max_regen_rounds,
        regen_temperature_delta=config.train.regen_temperature_delta,
        reward_gap_threshold=config.train.reward_gap_threshold,
        gap_sampling_temperature_delta=config.train.gap_sampling_temperature_delta,
        actor_chunk_size=config.train.actor_chunk_size,
        parallel_group_generate=args.parallel_group_generate,
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
    eval_history: deque[dict[str, float]] = deque(maxlen=max(1, int(config.train.early_stop_patience)))

    for epoch in range(1, config.train.num_epochs + 1):
        if config.train.curriculum_enable:
            epoch_queries = sample_curriculum_queries(
                train_queries,
                metadata_by_qid,
                phase=config.train.curriculum_phase,
                seed=config.data.seed,
                epoch=epoch,
            )
        else:
            epoch_queries = list(train_queries)
            random.Random(config.data.seed + epoch).shuffle(epoch_queries)
        for batch in iter_batches(epoch_queries, config.train.batch_size):
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
                f"pg={metrics['loss_pg']:.4f} pg_abs={metrics.get('loss_pg_abs_mean', 0.0):.4f} "
                f"kl={metrics['loss_kl']:.4f} "
                f"kl_dom={metrics.get('kl_dominance_ratio', 0.0):.3f} "
                f"reward={metrics['reward_mean']:.4f} mrr={metrics['mrr_mean']:.4f} "
                f"orig_mrr20={metrics.get('orig_mrr20_mean', 0.0):.4f} "
                f"rewrite_mrr20={metrics.get('rewrite_mrr20_mean', 0.0):.4f} "
                f"recall={metrics.get('recall_mean', 0.0):.4f} "
                f"recall_dense={metrics.get('recall_dense_mean', 0.0):.4f} "
                f"main_reward={metrics.get('main_reward_mean', 0.0):.4f} "
                f"nonzero_mrr20_ratio={metrics.get('nonzero_mrr20_ratio', 0.0):.4f} "
                f"delta_mrr20_pos_ratio={metrics.get('delta_mrr20_positive_ratio', 0.0):.4f} "
                f"term_preserve={metrics.get('term_preserve_mean', 0.0):.4f} "
                f"keyword_preserve={metrics.get('keyword_preserve_mean', 0.0):.4f} "
                f"locked_term_preserve={metrics.get('locked_term_preserve_mean', 0.0):.4f} "
                f"length_score={metrics.get('length_score_mean', 0.0):.4f} "
                f"clean_format={metrics.get('clean_format_mean', 0.0):.4f} "
                f"overedit_penalty={metrics.get('overedit_penalty_mean', 0.0):.4f} "
                f"bad_format_penalty={metrics.get('bad_format_penalty_mean', 0.0):.4f} "
                f"unsafe_copy_penalty={metrics.get('unsafe_copy_penalty_mean', 0.0):.4f} "
                f"unique_final_query_mean={metrics.get('unique_final_query_mean', 0.0):.4f} "
                f"generated_sample_count_mean={metrics.get('generated_sample_count_mean', 0.0):.4f} "
                f"generated_sample_count_max={metrics.get('generated_sample_count_max', 0.0):.0f} "
                f"extra_sample_ratio={metrics.get('extra_sample_ratio', 0.0):.4f} "
                f"reward_gap_raw_mean={metrics.get('reward_gap_raw_mean', 0.0):.4f} "
                f"reward_gap_met_ratio={metrics.get('reward_gap_met_ratio', 0.0):.4f} "
                f"max_group_size_hit_ratio={metrics.get('max_group_size_hit_ratio', 0.0):.4f} "
                f"collapsed_group_ratio={metrics.get('collapsed_group_ratio', 0.0):.4f} "
                f"all_same_final_query_ratio={metrics.get('all_same_final_query_ratio', 0.0):.4f} "
                f"flat_reward_group_ratio={metrics.get('flat_reward_group_ratio', 0.0):.4f} "
                f"flat_mrr20_group_ratio={metrics.get('flat_mrr20_group_ratio', 0.0):.4f} "
                f"flat_main_reward_group_ratio={metrics.get('flat_main_reward_group_ratio', 0.0):.4f} "
                f"best_reward_hit_best_mrr20_ratio={metrics.get('best_reward_hit_best_mrr20_ratio', 0.0):.4f}"
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
                    f"[eval] step={global_step} val_mrr={eval_metrics['mrr_mean']:.4f} "
                    f"val_orig_mrr20={eval_metrics.get('orig_mrr20_mean', 0.0):.4f} "
                    f"val_rewrite_mrr20={eval_metrics.get('rewrite_mrr20_mean', 0.0):.4f} "
                    f"val_reward={eval_metrics['reward_mean']:.4f} "
                    f"val_main_reward={eval_metrics.get('main_reward_mean', 0.0):.4f} "
                    f"val_nonzero_mrr20_ratio={eval_metrics.get('nonzero_mrr20_ratio', 0.0):.4f} "
                    f"val_delta_mrr20_pos_ratio={eval_metrics.get('delta_mrr20_positive_ratio', 0.0):.4f} "
                    f"val_recall_dense={eval_metrics.get('recall_dense_mean', 0.0):.4f} "
                    f"val_term_preserve={eval_metrics.get('term_preserve_mean', 0.0):.4f} "
                    f"val_keyword_preserve={eval_metrics.get('keyword_preserve_mean', 0.0):.4f} "
                    f"val_locked_term_preserve={eval_metrics.get('locked_term_preserve_mean', 0.0):.4f} "
                    f"val_length_score={eval_metrics.get('length_score_mean', 0.0):.4f} "
                    f"val_clean_format={eval_metrics.get('clean_format_mean', 0.0):.4f} "
                    f"val_overedit_penalty={eval_metrics.get('overedit_penalty_mean', 0.0):.4f} "
                    f"val_bad_format_penalty={eval_metrics.get('bad_format_penalty_mean', 0.0):.4f} "
                    f"val_unsafe_copy_penalty={eval_metrics.get('unsafe_copy_penalty_mean', 0.0):.4f}"
                )

                model.save_adapter(str(latest_path))
                current_eval_mrr = float(eval_metrics.get("rewrite_mrr20_mean", eval_metrics["mrr_mean"]))
                if current_eval_mrr > best_val_mrr:
                    best_val_mrr = current_eval_mrr
                    model.save_adapter(str(best_path))
                    print(f"[ckpt] best updated: mrr={best_val_mrr:.4f} -> {best_path}")

                eval_history.append(
                    {
                        "rewrite_mrr20_mean": current_eval_mrr,
                        "flat_main_reward_group_ratio": float(
                            metrics.get("flat_main_reward_group_ratio", metrics.get("flat_reward_group_ratio", 0.0))
                        ),
                        "kl_dominance_ratio": float(metrics.get("kl_dominance_ratio", 0.0)),
                        "delta_mrr20_positive_ratio": float(metrics.get("delta_mrr20_positive_ratio", 0.0)),
                    }
                )
                if len(eval_history) >= max(1, int(config.train.early_stop_patience)):
                    history_rows = list(eval_history)
                    mrr_stalled = all(
                        history_rows[idx]["rewrite_mrr20_mean"] <= history_rows[idx - 1]["rewrite_mrr20_mean"]
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
                    if mrr_stalled or flat_stalled or kl_bad:
                        reason = (
                            "rewrite_mrr20_stalled"
                            if mrr_stalled
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

    # 9) 训练结束后做最终评估与输出汇总。
    model.save_adapter(str(latest_path))
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
