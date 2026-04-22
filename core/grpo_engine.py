from __future__ import annotations

"""GRPO 训练主循环。

这里保留最核心的 top20_delta curriculum 训练流程：
1. 每个 query 采样一组 rewrite；
2. 清洗/兜底后用 BM25 奖励打分；
3. 组内归一化 reward 得到 advantage；
4. 用 PPO clipped objective + ref KL 做一次 LoRA 更新。
"""

from dataclasses import dataclass
from statistics import fmean
from typing import Sequence

import torch
from torch.nn.utils import clip_grad_norm_

from core.reward_func import RewardBreakdown, stabilize_generated_rewrite
from data.loader import QueryExample


@dataclass(slots=True)
class Sample:
    """单个采样候选，训练 loss、日志和诊断都会从这里取数。"""

    # 数据集 query id，用来把采样、奖励、trace 和指标行关联起来。
    qid: str

    # 送给 actor 的完整 prompt。
    prompt: str

    # 模型原始解码结果，尚未应用兜底逻辑。
    response_text: str

    # 从 response_text 中清洗出的候选检索 query。
    cleaned_query: str

    # 实际送入检索器的 query；不安全时可能回退到原 query。
    final_query: str

    # 生成回复的 token id，用来重算新策略/ref 策略 logprob。
    response_token_ids: list[int]

    # 采样时 actor 的旧策略 logprob，是 PPO ratio 的分母。
    logprob_old: torch.Tensor

    # 最终标量奖励，已经包含 top20 delta、bonus 和 penalty。
    reward: float

    # rewrite 的 MRR@k，默认是 MRR@20。
    mrr: float

    # rewrite 的 Recall@k，默认是 Recall@20。
    recall: float

    # rewrite 的辅助 Recall@k，默认是 Recall@50。
    recall_dense: float

    # 第一个相关文档命中位置带来的 rank bonus。
    rank_bonus: float

    # 主奖励项：delta MRR + Recall@20 + Recall@50 + rank bonus 的加权和。
    main_reward: float

    # 原 query 的 MRR@k 基线。
    orig_mrr: float

    # 原 query 的 Recall@k 基线。
    orig_recall: float

    # 原 query 的辅助 Recall@k 基线，默认 Recall@50。
    orig_recall_aux: float

    # 原 query 的 rank bonus 基线。
    orig_rank_bonus: float

    # rewrite MRR 减去原 query MRR。
    delta_mrr: float

    # rewrite Recall@20 减去原 query Recall@20。
    delta_recall: float

    # rewrite Recall@50 减去原 query Recall@50。
    delta_recall_aux: float

    # rewrite rank bonus 减去原 query rank bonus。
    delta_rank_bonus: float

    # 当 rewrite 同时不低于原 query 的 MRR 和 recall 时给的 anchor bonus。
    anchor_bonus: float

    # recall 低于原 query 时的惩罚。
    recall_drop_penalty: float

    # 源 query 中有意义关键词被保留的比例。
    keyword_preserve: float

    # keyword_preserve 低于 overedit_tau 时的过度改写惩罚。
    overedit_penalty: float

    # 格式惩罚：多行、解释文本、不可读字符、模板污染等。
    bad_format_penalty: float

    # 原 query 明显需要改写但模型直接复制时的惩罚。
    unsafe_copy_penalty: float

    # 是否触发兜底，把 final_query 替换回原 query。
    fallback_to_original: bool

    # 兜底原因，例如 empty_after_clean。
    fallback_reasons: tuple[str, ...]

    # 组内 reward 标准化后的 advantage，用于 PPO/GRPO policy loss。
    advantage: float = 0.0


def normalize_advantages(rewards: Sequence[float], eps: float = 1e-8) -> torch.Tensor:
    """Normalize rewards within each group to get relative advantages."""

    rewards_tensor = torch.tensor(list(rewards), dtype=torch.float32)
    if rewards_tensor.numel() == 0:
        return rewards_tensor

    mean = rewards_tensor.mean()
    std = rewards_tensor.std(unbiased=False)
    if std.item() < eps:
        return torch.zeros_like(rewards_tensor)
    return (rewards_tensor - mean) / (std + eps)


def ppo_clipped_objective(
    logprob_new: torch.Tensor,
    logprob_old: torch.Tensor,
    advantage: float,
    clip_range: float,
) -> torch.Tensor:
    """Token-level PPO clipped surrogate objective."""

    advantage_tensor = torch.full_like(logprob_new, float(advantage))
    ratios = torch.exp(logprob_new - logprob_old)
    unclipped = ratios * advantage_tensor
    clipped = torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range) * advantage_tensor
    return torch.min(unclipped, clipped)


class GRPOEngine:
    def __init__(
        self,
        model_wrapper,
        rewarder,
        optimizer: torch.optim.Optimizer,
        *,
        group_size: int,
        clip_range: float,
        kl_beta: float,
        grad_clip_norm: float,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        group_temperature_stride: float = 0.0,
        group_top_p_stride: float = 0.0,
        max_group_size: int = 14,
        min_unique_final_queries: int = 4,
        max_regen_rounds: int = 2,
        regen_temperature_delta: float = 0.15,
        reward_gap_threshold: float = 0.08,
        gap_sampling_temperature_delta: float = 0.15,
        actor_chunk_size: int = 8,
    ) -> None:
        self.model_wrapper = model_wrapper
        self.rewarder = rewarder
        self.optimizer = optimizer
        self.group_size = group_size
        self.max_group_size = max(int(group_size), int(max_group_size))
        self.clip_range = clip_range
        self.kl_beta = kl_beta
        self.grad_clip_norm = grad_clip_norm
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.group_temperature_stride = max(0.0, float(group_temperature_stride))
        self.group_top_p_stride = max(0.0, float(group_top_p_stride))
        self.min_unique_final_queries = max(1, int(min_unique_final_queries))
        self.max_regen_rounds = max(0, int(max_regen_rounds))
        self.regen_temperature_delta = max(0.0, float(regen_temperature_delta))
        self.reward_gap_threshold = max(0.0, float(reward_gap_threshold))
        self.gap_sampling_temperature_delta = max(0.0, float(gap_sampling_temperature_delta))
        self.actor_chunk_size = max(1, int(actor_chunk_size))

    def _stabilize_generated_sample(self, source_query: str, generated_sample) -> object:
        raw_text = generated_sample.raw_response_text or generated_sample.response_text
        return stabilize_generated_rewrite(
            raw_text,
            source_query=source_query,
            guardrail_cfg=self.model_wrapper.prompt_cfg,
            reward_cfg=self.rewarder.cfg,
        )

    def _generate_rollouts(
        self,
        prompt: str,
        *,
        num_samples: int,
        temperature: float,
    ) -> list[object]:
        # 同一个 query 会采样多个候选；temperature/top_p 逐个轻微拉开，
        # 让组内既有差异，又不至于完全发散。
        requested = max(1, int(num_samples))
        rollout_schedule = self._build_group_sampling_schedule(
            requested,
            base_temperature=temperature,
            base_top_p=self.top_p,
        )
        return [
            self.model_wrapper.generate_with_logprob(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=sample_temperature,
                top_p=sample_top_p,
            )
            for sample_temperature, sample_top_p in rollout_schedule
        ]

    def _build_group_sampling_schedule(
        self,
        num_samples: int,
        *,
        base_temperature: float,
        base_top_p: float,
    ) -> list[tuple[float, float]]:
        requested = max(1, int(num_samples))
        if requested == 1:
            return [(base_temperature, base_top_p)]

        schedule: list[tuple[float, float]] = []
        for idx in range(requested):
            sample_temperature = base_temperature
            sample_top_p = base_top_p
            if base_temperature > 0.0 and self.group_temperature_stride > 0.0:
                sample_temperature = min(1.35, base_temperature + self.group_temperature_stride * idx)
            if base_temperature > 0.0 and self.group_top_p_stride > 0.0:
                sample_top_p = min(0.995, base_top_p + self.group_top_p_stride * idx)
            schedule.append((sample_temperature, sample_top_p))
        return schedule

    def _generate_group_rollouts(self, prompt: str) -> list[object]:
        return self._generate_rollouts(
            prompt,
            num_samples=self.group_size,
            temperature=self.temperature,
        )

    def _generate_batch_group_rollouts(self, batch_queries: Sequence[QueryExample]) -> list[list[object]]:
        query_list = list(batch_queries)
        if not query_list:
            return []

        prompts = [self.model_wrapper.build_prompt(query.text) for query in query_list]
        rollout_groups: list[list[object]] = [[] for _ in query_list]
        rollout_schedule = self._build_group_sampling_schedule(
            self.group_size,
            base_temperature=self.temperature,
            base_top_p=self.top_p,
        )

        if not hasattr(self.model_wrapper, "generate_with_logprob_batch"):
            return [self._generate_group_rollouts(prompt) for prompt in prompts]

        for sample_temperature, sample_top_p in rollout_schedule:
            slot_samples = self.model_wrapper.generate_with_logprob_batch(
                prompts,
                max_new_tokens=self.max_new_tokens,
                temperature=sample_temperature,
                top_p=sample_top_p,
            )
            if len(slot_samples) != len(rollout_groups):
                raise RuntimeError(
                    "Batch rollout generation returned "
                    f"{len(slot_samples)} samples for {len(rollout_groups)} prompts."
                )
            for rollout_group, sample in zip(rollout_groups, slot_samples):
                rollout_group.append(sample)

        return rollout_groups

    def _maybe_regenerate_group(
        self,
        *,
        prompt: str,
        source_query: str,
        generated_group: list[object],
        stabilized_group: list[object],
    ) -> tuple[list[object], list[object]]:
        # 如果一组候选重复、污染或全是复制原 query，就追加几轮更高温采样，
        # 目标是给 GRPO 一个有 reward 差异的比较组。
        if self.max_regen_rounds <= 0 or self.temperature <= 0.0 or len(stabilized_group) <= 1:
            return generated_group, stabilized_group

        target_unique = min(len(stabilized_group), self.min_unique_final_queries)
        if target_unique <= 1:
            return generated_group, stabilized_group

        for round_idx in range(self.max_regen_rounds):
            unique_final_queries = list(dict.fromkeys(record.final_query for record in stabilized_group))
            if len(unique_final_queries) >= target_unique:
                break

            seen_final_queries: set[str] = set()
            candidate_indices: list[int] = []
            for idx, record in enumerate(stabilized_group):
                is_duplicate = record.final_query in seen_final_queries
                seen_final_queries.add(record.final_query)
                is_polluted = record.raw_contains_label or record.raw_multiline or record.raw_format_penalty > 0.0
                is_exact_copy = record.final_query.strip().lower() == source_query.strip().lower()
                if is_duplicate or is_polluted or is_exact_copy:
                    candidate_indices.append(idx)

            if not candidate_indices:
                break

            regen_temperature = min(1.35, self.temperature + self.regen_temperature_delta * (round_idx + 1))
            round_changed = False

            for idx in candidate_indices:
                current_record = stabilized_group[idx]
                current_is_polluted = (
                    current_record.raw_contains_label
                    or current_record.raw_multiline
                    or current_record.raw_format_penalty > 0.0
                )
                other_queries = {
                    record.final_query for pos, record in enumerate(stabilized_group) if pos != idx and record.final_query
                }

                regenerated = self.model_wrapper.generate_with_logprob(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=regen_temperature,
                    top_p=min(0.995, self.top_p + max(self.group_top_p_stride, 0.01) * (round_idx + 1)),
                )
                regenerated_record = self._stabilize_generated_sample(source_query, regenerated)
                regenerated_is_polluted = (
                    regenerated_record.raw_contains_label
                    or regenerated_record.raw_multiline
                    or regenerated_record.raw_format_penalty > 0.0
                )
                current_is_exact_copy = current_record.final_query.strip().lower() == source_query.strip().lower()
                regenerated_is_exact_copy = regenerated_record.final_query.strip().lower() == source_query.strip().lower()

                should_replace = False
                if regenerated_record.final_query and regenerated_record.final_query not in other_queries:
                    should_replace = True
                elif current_is_polluted and not regenerated_is_polluted:
                    should_replace = True
                elif current_is_exact_copy and not regenerated_is_exact_copy:
                    should_replace = True
                elif current_record.fallback_to_original and not regenerated_record.fallback_to_original:
                    should_replace = True

                if should_replace:
                    generated_group[idx] = regenerated
                    stabilized_group[idx] = regenerated_record
                    round_changed = True

            if not round_changed:
                break

        return generated_group, stabilized_group

    def _update_reward_cache(
        self,
        *,
        qid: str,
        source_query: str,
        final_queries: Sequence[str],
        reward_by_query: dict[str, RewardBreakdown],
    ) -> None:
        # 同一组里经常会出现相同 final_query；先去重再检索，避免重复打分。
        missing_queries = [
            rewritten_query
            for rewritten_query in dict.fromkeys(final_queries)
            if rewritten_query not in reward_by_query
        ]
        if not missing_queries:
            return

        if hasattr(self.rewarder, "score_batch"):
            rewards_group = self.rewarder.score_batch(
                qid,
                missing_queries,
                source_query=source_query,
            )
        else:
            rewards_group = [
                self.rewarder.score(qid, rewritten_query, source_query=source_query)
                for rewritten_query in missing_queries
            ]

        for rewritten_query, reward in zip(missing_queries, rewards_group):
            reward_by_query[rewritten_query] = reward

    @staticmethod
    def _compute_reward_gap_raw(
        final_queries: Sequence[str],
        reward_by_query: dict[str, RewardBreakdown],
    ) -> float:
        unique_queries = list(dict.fromkeys(final_queries))
        raw_rewards = [reward_by_query[rewritten_query].total for rewritten_query in unique_queries if rewritten_query in reward_by_query]
        if len(raw_rewards) < 2:
            return 0.0
        return float(max(raw_rewards) - min(raw_rewards))

    def train_step(
        self,
        batch_queries: Sequence[QueryExample],
        *,
        collect_best_queries: bool = False,
    ) -> dict[str, object]:
        """Run one GRPO update on a mini-batch."""

        previous_mode = self.model_wrapper.actor_model.training
        # 采样和 logprob 计算阶段关闭 dropout，避免同一个样本的 old/new logprob
        # 被训练随机性污染。
        self.model_wrapper.actor_model.eval()
        try:
            self.optimizer.zero_grad(set_to_none=True)  # 清空上一轮梯度；set_to_none=True 更省显存。

            loss_values: list[float] = []  # 每个有效样本的总 loss，用于计算 loss 均值。
            loss_pg_terms: list[float] = []  # policy gradient loss 部分，用于观察 PPO 主项大小。
            loss_kl_terms: list[float] = []  # KL loss 部分，用于观察 ref 约束强度。

            rewards: list[float] = []  # 每个采样候选的最终 reward，用于 reward_mean 和 advantage。
            mrr_scores: list[float] = []  # 每个候选 rewrite 的 MRR@20，用于 rewrite_mrr20_mean。
            recall_scores: list[float] = []  # 每个候选 rewrite 的 Recall@20，用于 rewrite_recall20_mean。
            recall_dense_scores: list[float] = []  # 每个候选 rewrite 的 Recall@50，用于 rewrite_recall50_mean。
            rank_bonus_scores: list[float] = []  # 每个候选的首个相关文档排名奖励，用于 rank_bonus_mean。
            main_reward_scores: list[float] = []  # 不含 penalty/anchor 的主奖励，用于诊断 reward 主体。
            orig_mrr_scores: list[float] = []  # 原 query 的 MRR@20 基线，用于和 rewrite 对比。
            orig_recall_scores: list[float] = []  # 原 query 的 Recall@20 基线，用于计算 recall drop。
            orig_recall_aux_scores: list[float] = []  # 原 query 的 Recall@50 基线，用于计算 delta_recall50。
            orig_rank_bonus_scores: list[float] = []  # 原 query 的 rank bonus 基线，用于计算排名奖励增量。
            delta_mrr_scores: list[float] = []  # rewrite MRR@20 - 原 query MRR@20，用于看是否真的提升。
            delta_recall_scores: list[float] = []  # rewrite Recall@20 - 原 query Recall@20，用于看召回变化。
            delta_recall_aux_scores: list[float] = []  # rewrite Recall@50 - 原 query Recall@50，用于看深召回变化。
            delta_rank_bonus_scores: list[float] = []  # rewrite rank bonus - 原 query rank bonus，用于看排名改善。
            anchor_bonus_scores: list[float] = []  # 达到“不低于原 query”条件时给的 anchor bonus。
            keyword_preserve_scores: list[float] = []  # 源 query 关键词保留比例，用于监控过度改写。
            recall_drop_penalties: list[float] = []  # Recall@20 低于原 query 时的惩罚，用于监控退化。
            overedit_penalties: list[float] = []  # 关键词保留过低时的惩罚，用于监控语义漂移。
            bad_format_penalties: list[float] = []  # 输出格式错误惩罚，用于监控多行/解释/污染输出。
            unsafe_copy_penalties: list[float] = []  # 不安全复制原 query 的惩罚，用于监控偷懒复制。
            all_advantages: list[float] = []  # 组内标准化后的 advantage，用于统计 adv_mean/adv_std。
            unique_final_query_counts: list[float] = []  # 每组不同 final_query 数量，用于判断采样多样性。
            generated_sample_counts: list[float] = []  # 每组最终生成候选数，包含补采样，用于看采样成本。
            reward_gap_raw_values: list[float] = []  # 每组最高/最低 reward 差，用于判断组内信号是否拉开。
            flat_reward_group_count = 0  # reward 全相同的组数；越高说明 reward 区分度越差。
            flat_mrr_group_count = 0  # MRR@20 全相同的组数；用于诊断检索指标是否太稀疏。
            flat_main_reward_group_count = 0  # main_reward 全相同的组数；用于诊断主奖励是否变平。
            collapsed_group_count = 0  # final_query 全坍缩成同一个文本的组数。
            all_same_final_query_group_count = 0  # 与 collapsed_group_count 同义保留项，用于日志兼容。
            reward_gap_met_count = 0  # reward gap 达到阈值的组数，用于观察补采样是否成功。
            max_group_size_hit_count = 0  # 补采样打到 max_group_size 仍没拉开 gap 的组数。
            extra_sample_count_total = 0  # 额外补采样总数，用于计算 extra_sample_ratio。
            best_reward_hit_best_mrr20_count = 0  # reward 最高样本是否也拿到组内最高 MRR@20 的计数。

            best_query_pairs: list[dict[str, object]] = []  # 可选输出：每个 query 的组内最佳 rewrite。
            group_query_summaries: list[dict[str, object]] = []  # 每个 query 组的详细 trace，写入 group_trace_log。

            valid_samples = 0  # 参与反传的有效样本数；空 token/logprob 的样本不会计入。
            sampled = 0  # 本 batch 总采样候选数，包含无效样本和补采样。
            num_groups = 0  # 本 batch 的 query 组数，一个 query 对应一个 group。

            batch_query_list = list(batch_queries)
            if hasattr(self.model_wrapper, "reset_rollout_batch_stats"):
                self.model_wrapper.reset_rollout_batch_stats()
            initial_rollout_groups = self._generate_batch_group_rollouts(batch_query_list)
            rollout_batch_prompt_counts = (
                self.model_wrapper.consume_rollout_batch_stats()
                if hasattr(self.model_wrapper, "consume_rollout_batch_stats")
                else []
            )
            if len(initial_rollout_groups) != len(batch_query_list):
                raise RuntimeError(
                    "Batch rollout grouping returned "
                    f"{len(initial_rollout_groups)} groups for {len(batch_query_list)} queries."
                )
            for query, generated_group in zip(batch_query_list, initial_rollout_groups):
                num_groups += 1
                prompt = self.model_wrapper.build_prompt(query.text)
                group_samples: list[Sample] = []

                # 1) 生成一组候选 rewrite，并先过 guardrail 得到 final_query。
                stabilized_group = [
                    self._stabilize_generated_sample(query.text, sample) for sample in generated_group
                ]
                generated_group, stabilized_group = self._maybe_regenerate_group(
                    prompt=prompt,
                    source_query=query.text,
                    generated_group=generated_group,
                    stabilized_group=stabilized_group,
                )
                initial_group_size = len(generated_group)
                reward_by_query: dict[str, RewardBreakdown] = {}
                final_query_group = [record.final_query for record in stabilized_group]
                # 2) 对组内唯一 final_query 检索打分，复用 reward cache。
                self._update_reward_cache(
                    qid=query.qid,
                    source_query=query.text,
                    final_queries=final_query_group,
                    reward_by_query=reward_by_query,
                )
                reward_gap_raw = self._compute_reward_gap_raw(final_query_group, reward_by_query)
                gap_sampling_temperatures: list[float] = []

                # 3) 如果组内 reward gap 太小，继续补采样；否则 advantage 近似全 0。
                while reward_gap_raw < self.reward_gap_threshold and len(generated_group) < self.max_group_size:
                    extra_round_idx = len(gap_sampling_temperatures) + 1
                    extra_temperature = round(
                        min(
                            1.35,
                            self.temperature + self.gap_sampling_temperature_delta * extra_round_idx,
                        ),
                        6,
                    )
                    gap_sampling_temperatures.append(extra_temperature)
                    extra_generated_group = self._generate_rollouts(
                        prompt,
                        num_samples=1,
                        temperature=extra_temperature,
                    )
                    extra_stabilized_group = [
                        self._stabilize_generated_sample(query.text, sample) for sample in extra_generated_group
                    ]
                    generated_group.extend(extra_generated_group)
                    stabilized_group.extend(extra_stabilized_group)
                    final_query_group = [record.final_query for record in stabilized_group]
                    self._update_reward_cache(
                        qid=query.qid,
                        source_query=query.text,
                        final_queries=final_query_group,
                        reward_by_query=reward_by_query,
                    )
                    reward_gap_raw = self._compute_reward_gap_raw(final_query_group, reward_by_query)

                reward_gap_met = reward_gap_raw >= self.reward_gap_threshold
                reward_gap_stop_reason = "threshold_reached" if reward_gap_met else "max_group_size_reached"
                extra_sample_count = len(generated_group) - initial_group_size
                extra_sample_count_total += extra_sample_count
                generated_sample_counts.append(float(len(generated_group)))
                reward_gap_raw_values.append(float(reward_gap_raw))
                if reward_gap_met:
                    reward_gap_met_count += 1
                if reward_gap_stop_reason == "max_group_size_reached":
                    max_group_size_hit_count += 1

                unique_final_queries = list(dict.fromkeys(final_query_group))
                group_collapsed = len(unique_final_queries) == 1

                # 4) 把 reward breakdown 固化成 Sample，后面 loss 和日志都只读 Sample。
                for generated, stabilized in zip(generated_group, stabilized_group):
                    reward = reward_by_query[stabilized.final_query]
                    keyword_preserve = getattr(reward, "keyword_preserve", getattr(reward, "term_preserve", 1.0))
                    group_samples.append(
                        Sample(
                            qid=query.qid,
                            prompt=prompt,
                            response_text=generated.response_text,
                            cleaned_query=stabilized.cleaned_query,
                            final_query=stabilized.final_query,
                            response_token_ids=generated.response_token_ids,
                            logprob_old=generated.logprob_old,
                            reward=reward.total,
                            mrr=reward.mrr,
                            recall=reward.recall,
                            recall_dense=reward.recall_dense,
                            rank_bonus=getattr(reward, "rank_bonus", 0.0),
                            main_reward=getattr(reward, "main_reward", reward.total),
                            orig_mrr=getattr(reward, "orig_mrr", 0.0),
                            orig_recall=getattr(reward, "orig_recall", 0.0),
                            orig_recall_aux=getattr(reward, "orig_recall_aux", 0.0),
                            orig_rank_bonus=getattr(reward, "orig_rank_bonus", 0.0),
                            delta_mrr=getattr(reward, "delta_mrr", 0.0),
                            delta_recall=getattr(reward, "delta_recall", 0.0),
                            delta_recall_aux=getattr(reward, "delta_recall_aux", 0.0),
                            delta_rank_bonus=getattr(reward, "delta_rank_bonus", 0.0),
                            anchor_bonus=getattr(reward, "anchor_bonus", 0.0),
                            recall_drop_penalty=getattr(reward, "recall_drop_penalty", 0.0),
                            keyword_preserve=keyword_preserve,
                            overedit_penalty=getattr(reward, "overedit_penalty", 0.0),
                            bad_format_penalty=reward.bad_format_penalty,
                            unsafe_copy_penalty=reward.unsafe_copy_penalty,
                            fallback_to_original=stabilized.fallback_to_original,
                            fallback_reasons=stabilized.fallback_reasons,
                        )
                    )
                sampled += len(group_samples)
                unique_final_query_counts.append(float(len(unique_final_queries)))
                if group_collapsed:
                    collapsed_group_count += 1
                    all_same_final_query_group_count += 1
                if len({sample.reward for sample in group_samples}) == 1:
                    flat_reward_group_count += 1
                if len({sample.mrr for sample in group_samples}) == 1:
                    flat_mrr_group_count += 1
                if len({sample.main_reward for sample in group_samples}) == 1:
                    flat_main_reward_group_count += 1
                if group_samples:
                    best_reward_sample = max(group_samples, key=lambda sample: sample.reward)
                    best_group_mrr = max(sample.mrr for sample in group_samples)
                    if best_reward_sample.mrr == best_group_mrr:
                        best_reward_hit_best_mrr20_count += 1

                group_query_summaries.append(
                    {
                        "qid": str(query.qid),
                        "input_query": query.text,
                        "initial_group_size": initial_group_size,
                        "generated_sample_count": len(group_samples),
                        "extra_sample_count": extra_sample_count,
                        "reward_gap_raw": float(reward_gap_raw),
                        "reward_gap_threshold": float(self.reward_gap_threshold),
                        "reward_gap_met": reward_gap_met,
                        "reward_gap_stop_reason": reward_gap_stop_reason,
                        "collapsed_group": group_collapsed,
                        "gap_sampling_rounds": len(gap_sampling_temperatures),
                        "gap_sampling_temperatures": gap_sampling_temperatures,
                        "group_raw_responses": [
                            generated.raw_response_text or generated.response_text for generated in generated_group
                        ],
                        "group_rollout_responses": [sample.response_text for sample in group_samples],
                        "group_cleaned_queries": [sample.cleaned_query for sample in group_samples],
                        "group_final_queries": [sample.final_query for sample in group_samples],
                        "group_fallback_to_original": [sample.fallback_to_original for sample in group_samples],
                        "group_fallback_reasons": [list(sample.fallback_reasons) for sample in group_samples],
                        "group_rewards": [sample.reward for sample in group_samples],
                        "group_mrr": [sample.mrr for sample in group_samples],
                        "group_recall": [sample.recall for sample in group_samples],
                        "group_recall_dense": [sample.recall_dense for sample in group_samples],
                        "group_rank_bonus": [sample.rank_bonus for sample in group_samples],
                        "group_orig_mrr": [sample.orig_mrr for sample in group_samples],
                        "group_orig_recall": [sample.orig_recall for sample in group_samples],
                        "group_orig_recall_aux": [sample.orig_recall_aux for sample in group_samples],
                        "group_orig_rank_bonus": [sample.orig_rank_bonus for sample in group_samples],
                        "group_delta_mrr": [sample.delta_mrr for sample in group_samples],
                        "group_delta_recall": [sample.delta_recall for sample in group_samples],
                        "group_delta_recall_aux": [sample.delta_recall_aux for sample in group_samples],
                        "group_delta_rank_bonus": [sample.delta_rank_bonus for sample in group_samples],
                        "group_main_rewards": [sample.main_reward for sample in group_samples],
                        "group_anchor_bonus": [sample.anchor_bonus for sample in group_samples],
                        "group_keyword_preserve": [sample.keyword_preserve for sample in group_samples],
                        "group_recall_drop_penalties": [sample.recall_drop_penalty for sample in group_samples],
                        "group_overedit_penalties": [sample.overedit_penalty for sample in group_samples],
                        "group_bad_format_penalties": [sample.bad_format_penalty for sample in group_samples],
                        "group_unsafe_copy_penalties": [sample.unsafe_copy_penalty for sample in group_samples],
                    }
                )

                if collect_best_queries and group_samples:
                    best_sample = max(group_samples, key=lambda sample: sample.reward)
                    best_query_pairs.append(
                        {
                            "qid": str(query.qid),
                            "input_query": query.text,
                            "best_rewritten_query": best_sample.final_query,
                            "best_reward": best_sample.reward,
                        }
                    )

                # 5) GRPO 的关键：只比较同一个 query 下的一组候选。
                advantages = normalize_advantages([sample.reward for sample in group_samples]).tolist()
                for sample, advantage in zip(group_samples, advantages):
                    sample.advantage = float(advantage)
                    all_advantages.append(sample.advantage)
                    rewards.append(sample.reward)
                    mrr_scores.append(sample.mrr)
                    recall_scores.append(sample.recall)
                    recall_dense_scores.append(sample.recall_dense)
                    rank_bonus_scores.append(sample.rank_bonus)
                    main_reward_scores.append(sample.main_reward)
                    orig_mrr_scores.append(sample.orig_mrr)
                    orig_recall_scores.append(sample.orig_recall)
                    orig_recall_aux_scores.append(sample.orig_recall_aux)
                    orig_rank_bonus_scores.append(sample.orig_rank_bonus)
                    delta_mrr_scores.append(sample.delta_mrr)
                    delta_recall_scores.append(sample.delta_recall)
                    delta_recall_aux_scores.append(sample.delta_recall_aux)
                    delta_rank_bonus_scores.append(sample.delta_rank_bonus)
                    anchor_bonus_scores.append(sample.anchor_bonus)
                    keyword_preserve_scores.append(sample.keyword_preserve)
                    recall_drop_penalties.append(sample.recall_drop_penalty)
                    overedit_penalties.append(sample.overedit_penalty)
                    bad_format_penalties.append(sample.bad_format_penalty)
                    unsafe_copy_penalties.append(sample.unsafe_copy_penalty)

                if group_collapsed:
                    continue

                valid_group_samples = [
                    sample for sample in group_samples if sample.response_token_ids and sample.logprob_old.numel() > 0
                ]
                if not valid_group_samples:
                    continue

                if hasattr(self.model_wrapper, "compute_logprob_batch"):
                    prompts = [sample.prompt for sample in valid_group_samples]
                    response_token_ids_batch = [sample.response_token_ids for sample in valid_group_samples]
                    logprob_ref_batch = self.model_wrapper.compute_logprob_batch(
                        prompts,
                        response_token_ids_batch,
                        policy="ref",
                        no_grad=True,
                    )

                    # 6) ref logprob 不需要梯度；actor logprob 分 chunk 重算并反传。
                    actor_chunk_size = min(self.actor_chunk_size, len(valid_group_samples))
                    for chunk_start in range(0, len(valid_group_samples), actor_chunk_size):
                        chunk_samples = valid_group_samples[chunk_start : chunk_start + actor_chunk_size]
                        chunk_prompts = prompts[chunk_start : chunk_start + actor_chunk_size]
                        chunk_response_token_ids = response_token_ids_batch[
                            chunk_start : chunk_start + actor_chunk_size
                        ]
                        chunk_logprob_ref = logprob_ref_batch[chunk_start : chunk_start + actor_chunk_size]
                        chunk_logprob_new = self.model_wrapper.compute_logprob_batch(
                            chunk_prompts,
                            chunk_response_token_ids,
                            policy="actor",
                            no_grad=False,
                        )

                        chunk_loss: torch.Tensor | None = None
                        for sample, logprob_new, logprob_ref in zip(
                            chunk_samples,
                            chunk_logprob_new,
                            chunk_logprob_ref,
                        ):
                            t = min(logprob_new.numel(), sample.logprob_old.numel(), logprob_ref.numel())
                            if t == 0:
                                continue

                            logprob_new = logprob_new[:t]
                            logprob_old = sample.logprob_old[:t].to(logprob_new.device)
                            logprob_ref = logprob_ref[:t].to(logprob_new.device)

                            clipped_obj = ppo_clipped_objective(
                                logprob_new=logprob_new,
                                logprob_old=logprob_old,
                                advantage=sample.advantage,
                                clip_range=self.clip_range,
                            )
                            loss_pg = -clipped_obj.mean()
                            loss_kl = self.kl_beta * (logprob_new - logprob_ref).mean()
                            loss = loss_pg + loss_kl
                            if not torch.isfinite(loss):
                                continue

                            chunk_loss = loss if chunk_loss is None else chunk_loss + loss
                            loss_values.append(float(loss.detach().cpu()))
                            loss_pg_terms.append(float(loss_pg.detach().cpu()))
                            loss_kl_terms.append(float(loss_kl.detach().cpu()))
                            valid_samples += 1

                        if chunk_loss is not None:
                            chunk_loss.backward()
                else:
                    for sample in valid_group_samples:
                        logprob_new = self.model_wrapper.compute_logprob(
                            sample.prompt,
                            sample.response_token_ids,
                            policy="actor",
                            no_grad=False,
                        )
                        logprob_ref = self.model_wrapper.compute_logprob(
                            sample.prompt,
                            sample.response_token_ids,
                            policy="ref",
                            no_grad=True,
                        )

                        t = min(logprob_new.numel(), sample.logprob_old.numel(), logprob_ref.numel())
                        if t == 0:
                            continue

                        logprob_new = logprob_new[:t]
                        logprob_old = sample.logprob_old[:t].to(logprob_new.device)
                        logprob_ref = logprob_ref[:t].to(logprob_new.device)

                        clipped_obj = ppo_clipped_objective(
                            logprob_new=logprob_new,
                            logprob_old=logprob_old,
                            advantage=sample.advantage,
                            clip_range=self.clip_range,
                        )
                        loss_pg = -clipped_obj.mean()
                        loss_kl = self.kl_beta * (logprob_new - logprob_ref).mean()
                        loss = loss_pg + loss_kl
                        if not torch.isfinite(loss):
                            continue

                        # Immediate backward prevents retaining many graphs at once.
                        loss.backward()
                        loss_values.append(float(loss.detach().cpu()))
                        loss_pg_terms.append(float(loss_pg.detach().cpu()))
                        loss_kl_terms.append(float(loss_kl.detach().cpu()))
                        valid_samples += 1

            nonzero_reward_ratio = (sum(1 for value in rewards if value > 0.0) / len(rewards)) if rewards else 0.0

            # 汇总训练日志：一部分看检索效果，一部分看采样是否坍缩、reward 是否拉开。
            metrics: dict[str, object] = {
                "reward_mean": fmean(rewards) if rewards else 0.0,  # 平均最终 reward，直接反映本 batch 训练信号强弱。
                "mrr_mean": fmean(mrr_scores) if mrr_scores else 0.0,  # rewrite 的平均 MRR@20，兼容通用 mrr_mean 字段。
                "recall_mean": fmean(recall_scores) if recall_scores else 0.0,  # rewrite 的平均 Recall@20，兼容通用 recall_mean 字段。
                "recall_dense_mean": fmean(recall_dense_scores) if recall_dense_scores else 0.0,  # rewrite 的平均 Recall@50。
                "rank_bonus_mean": fmean(rank_bonus_scores) if rank_bonus_scores else 0.0,  # 平均首个相关文档排名 bonus。
                "main_reward_mean": fmean(main_reward_scores) if main_reward_scores else 0.0,  # 不含惩罚/anchor 的主奖励均值。
                "orig_mrr20_mean": fmean(orig_mrr_scores) if orig_mrr_scores else 0.0,  # 原 query 的平均 MRR@20 基线。
                "rewrite_mrr20_mean": fmean(mrr_scores) if mrr_scores else 0.0,  # rewrite 的平均 MRR@20，训练主指标。
                "orig_recall20_mean": fmean(orig_recall_scores) if orig_recall_scores else 0.0,  # 原 query 的平均 Recall@20。
                "rewrite_recall20_mean": fmean(recall_scores) if recall_scores else 0.0,  # rewrite 的平均 Recall@20。
                "rewrite_recall50_mean": fmean(recall_dense_scores) if recall_dense_scores else 0.0,  # rewrite 的平均 Recall@50。
                "delta_mrr20_mean": fmean(delta_mrr_scores) if delta_mrr_scores else 0.0,  # rewrite 相对原 query 的 MRR@20 平均提升。
                "delta_recall20_mean": fmean(delta_recall_scores) if delta_recall_scores else 0.0,  # rewrite 相对原 query 的 Recall@20 平均变化。
                "delta_recall50_mean": fmean(delta_recall_aux_scores) if delta_recall_aux_scores else 0.0,  # rewrite 相对原 query 的 Recall@50 平均变化。
                "orig_rank_bonus_mean": fmean(orig_rank_bonus_scores) if orig_rank_bonus_scores else 0.0,  # 原 query 的平均 rank bonus。
                "delta_rank_bonus_mean": fmean(delta_rank_bonus_scores) if delta_rank_bonus_scores else 0.0,  # rewrite 相对原 query 的 rank bonus 平均变化。
                "anchor_bonus_mean": fmean(anchor_bonus_scores) if anchor_bonus_scores else 0.0,  # 平均 anchor bonus，越高表示 rewrite 更常不低于原 query。
                "keyword_preserve_mean": fmean(keyword_preserve_scores) if keyword_preserve_scores else 0.0,  # 关键词保留均值，用于观察是否过度改写。
                "recall_drop_penalty_mean": (  # 平均 recall 下降惩罚，用于观察 rewrite 是否牺牲召回。
                    fmean(recall_drop_penalties) if recall_drop_penalties else 0.0
                ),
                "overedit_penalty_mean": fmean(overedit_penalties) if overedit_penalties else 0.0,  # 平均过度改写惩罚。
                "bad_format_penalty_mean": fmean(bad_format_penalties) if bad_format_penalties else 0.0,  # 平均格式惩罚。
                "unsafe_copy_penalty_mean": fmean(unsafe_copy_penalties) if unsafe_copy_penalties else 0.0,  # 平均不安全复制惩罚。
                "nonzero_reward_ratio": nonzero_reward_ratio,  # reward 大于 0 的候选比例，用于看有效正信号多少。
                "nonzero_mrr20_ratio": (  # MRR@20 大于 0 的候选比例，用于看命中 top20 的覆盖率。
                    sum(1 for value in mrr_scores if value > 0.0) / len(mrr_scores)
                ) if mrr_scores else 0.0,
                "nonzero_recall20_ratio": (  # Recall@20 大于 0 的候选比例，用于看 top20 是否召回到相关文档。
                    sum(1 for value in recall_scores if value > 0.0) / len(recall_scores)
                ) if recall_scores else 0.0,
                "delta_mrr20_positive_ratio": (  # delta MRR@20 为正的候选比例，越高表示 rewrite 更常提升排序。
                    sum(1 for value in delta_mrr_scores if value > 0.0) / len(delta_mrr_scores)
                ) if delta_mrr_scores else 0.0,
                "delta_recall20_positive_ratio": (  # delta Recall@20 为正的候选比例，越高表示 rewrite 更常提升召回。
                    sum(1 for value in delta_recall_scores if value > 0.0) / len(delta_recall_scores)
                ) if delta_recall_scores else 0.0,
                "anchor_hit_ratio": (  # 拿到 anchor bonus 的候选比例，即同时不低于原 query 的比例。
                    sum(1 for value in anchor_bonus_scores if value > 0.0) / len(anchor_bonus_scores)
                ) if anchor_bonus_scores else 0.0,
                "recall_drop_ratio": (  # 出现 recall 下降惩罚的候选比例，用于监控退化风险。
                    sum(1 for value in recall_drop_penalties if value > 0.0) / len(recall_drop_penalties)
                ) if recall_drop_penalties else 0.0,
                "adv_mean": fmean(all_advantages) if all_advantages else 0.0,  # 组内标准化 advantage 的均值，正常应接近 0。
                "adv_std": float(torch.tensor(all_advantages).std(unbiased=False)) if all_advantages else 0.0,  # advantage 标准差，用于看组内 reward 是否有区分度。
                "unique_final_query_mean": fmean(unique_final_query_counts) if unique_final_query_counts else 0.0,  # 每组不同 final_query 的平均数量。
                "generated_sample_count_mean": fmean(generated_sample_counts) if generated_sample_counts else 0.0,  # 每组平均生成候选数，包含补采样。
                "generated_sample_count_max": float(max(generated_sample_counts)) if generated_sample_counts else 0.0,  # 单组最大生成候选数。
                "extra_sample_ratio": (extra_sample_count_total / sampled) if sampled else 0.0,  # 补采样占总采样的比例。
                "reward_gap_raw_mean": fmean(reward_gap_raw_values) if reward_gap_raw_values else 0.0,  # 每组最高/最低 reward 差的均值。
                "reward_gap_met_ratio": (reward_gap_met_count / num_groups) if num_groups else 0.0,  # reward gap 达到阈值的组比例。
                "max_group_size_hit_ratio": (max_group_size_hit_count / num_groups) if num_groups else 0.0,  # 补到 max_group_size 仍没达标的组比例。
                "collapsed_group_ratio": (collapsed_group_count / num_groups) if num_groups else 0.0,  # 组内 final_query 坍缩成一个文本的比例。
                "all_same_final_query_ratio": (all_same_final_query_group_count / num_groups) if num_groups else 0.0,  # 与 collapsed_group_ratio 同义的兼容指标。
                "flat_reward_group_ratio": (flat_reward_group_count / num_groups) if num_groups else 0.0,  # 组内 reward 全相同的比例。
                "flat_mrr20_group_ratio": (flat_mrr_group_count / num_groups) if num_groups else 0.0,  # 组内 MRR@20 全相同的比例。
                "flat_main_reward_group_ratio": (  # 组内 main_reward 全相同的比例，用于发现主奖励变平。
                    flat_main_reward_group_count / num_groups
                ) if num_groups else 0.0,
                "best_reward_hit_best_mrr20_ratio": (  # reward 最高样本同时也是组内最高 MRR@20 的比例。
                    best_reward_hit_best_mrr20_count / num_groups
                ) if num_groups else 0.0,
                "sampled": float(sampled),  # 本 batch 总采样候选数。
                "valid_samples": float(valid_samples),  # 真正参与反传更新的有效样本数。
                "group_query_summaries": group_query_summaries,  # 每个 query 组的详细 trace，调用方会写入 group_trace_log。
            }
            metrics.update(
                {
                    "rollout_batch_prompt_count_mean": (
                        fmean(rollout_batch_prompt_counts) if rollout_batch_prompt_counts else 0.0
                    ),
                    "rollout_batch_prompt_count_min": (
                        float(min(rollout_batch_prompt_counts)) if rollout_batch_prompt_counts else 0.0
                    ),
                    "rollout_batch_prompt_count_max": (
                        float(max(rollout_batch_prompt_counts)) if rollout_batch_prompt_counts else 0.0
                    ),
                    "rollout_batch_call_count": float(len(rollout_batch_prompt_counts)),
                    "rollout_batch_total_prompts": float(sum(rollout_batch_prompt_counts)),
                    "rollout_batch_expected_prompts": float(len(batch_query_list) * self.group_size),
                    "rollout_batch_fallback_split_count": float(
                        max(0, len(rollout_batch_prompt_counts) - self.group_size)
                    ),
                }
            )

            if valid_samples == 0:
                metrics.update(
                    {
                        "loss": 0.0,  # 没有有效样本时总 loss 记 0，表示本步没有可反传信号。
                        "loss_pg": 0.0,  # 没有有效样本时 policy gradient loss 记 0。
                        "loss_pg_abs_mean": 0.0,  # 没有有效样本时 policy loss 绝对值均值记 0。
                        "loss_kl": 0.0,  # 没有有效样本时 KL loss 记 0。
                        "kl_dominance_ratio": 0.0,  # 没有有效样本时 KL 占比记 0，避免误判 KL 主导。
                        "updated": 0.0,  # 标记本 train_step 没有执行 optimizer.step()。
                    }
                )
                if collect_best_queries:
                    metrics["best_query_pairs"] = best_query_pairs  # 可选调试字段：每组 reward 最高的 rewrite。
                return metrics

            # 前面是“逐样本累计梯度”，这里除以有效样本数，变回 mean loss 的尺度。
            grad_scale = 1.0 / float(valid_samples)
            for param in self.model_wrapper.trainable_parameters():
                if param.grad is not None:
                    param.grad.mul_(grad_scale)

            clip_grad_norm_(self.model_wrapper.trainable_parameters(), self.grad_clip_norm)
            self.optimizer.step()

            loss_pg_mean = fmean(loss_pg_terms) if loss_pg_terms else 0.0
            loss_kl_mean = fmean(loss_kl_terms) if loss_kl_terms else 0.0
            loss_pg_abs_mean = fmean(abs(value) for value in loss_pg_terms) if loss_pg_terms else 0.0
            loss_kl_abs_mean = fmean(abs(value) for value in loss_kl_terms) if loss_kl_terms else 0.0
            kl_dominance_ratio = loss_kl_abs_mean / (loss_pg_abs_mean + loss_kl_abs_mean + 1e-12)

            metrics.update(
                {
                    "loss": fmean(loss_values) if loss_values else 0.0,  # 有效样本上的平均总 loss。
                    "loss_pg": loss_pg_mean,  # PPO clipped policy loss 均值，主要反映 advantage 驱动项。
                    "loss_pg_abs_mean": loss_pg_abs_mean,  # policy loss 绝对值均值，用于和 KL 项比较量级。
                    "loss_kl": loss_kl_mean,  # ref KL loss 均值，约束 actor 不要偏离基座太远。
                    "kl_dominance_ratio": kl_dominance_ratio,  # KL 绝对量级占 loss_pg+loss_kl 的比例。
                    "updated": 1.0,  # 标记本 train_step 已执行 optimizer.step()。
                }
            )
            if collect_best_queries:
                metrics["best_query_pairs"] = best_query_pairs  # 可选调试字段：每组 reward 最高的 rewrite。
            return metrics
        finally:
            self.model_wrapper.actor_model.train(previous_mode)
