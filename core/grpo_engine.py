from __future__ import annotations

"""Core GRPO training loop implemented with plain PyTorch."""

from dataclasses import dataclass
from statistics import fmean
from typing import Sequence

import torch
from torch.nn.utils import clip_grad_norm_

from core.reward_func import RewardBreakdown, stabilize_generated_rewrite
from data.loader import QueryExample

@dataclass(slots=True)
class Sample:
    """One sampled candidate used for PPO/GRPO loss computation."""

    qid: str
    prompt: str
    response_text: str
    cleaned_query: str
    final_query: str
    response_token_ids: list[int]
    logprob_old: torch.Tensor
    reward: float
    mrr: float
    recall: float
    copy_penalty: float
    format_penalty: float
    duplicate_penalty: float
    fallback_to_original: bool
    fallback_reasons: tuple[str, ...]
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


def compute_group_duplicate_penalties(queries: Sequence[str], penalty_step: float) -> list[float]:
    """Apply a small deterministic penalty to repeated final queries within one GRPO group."""

    occurrence_by_query: dict[str, int] = {}
    penalties: list[float] = []
    for query in queries:
        occurrence = occurrence_by_query.get(query, 0)
        penalties.append(float(occurrence) * float(penalty_step))
        occurrence_by_query[query] = occurrence + 1
    return penalties


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
        max_group_size: int = 24,
        min_unique_final_queries: int = 3,
        max_regen_rounds: int = 2,
        regen_temperature_delta: float = 0.15,
        reward_gap_threshold: float = 0.10,
        gap_sampling_temperature_delta: float = 0.15,
        parallel_group_generate: bool = False,
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
        self.min_unique_final_queries = max(1, int(min_unique_final_queries))
        self.max_regen_rounds = max(0, int(max_regen_rounds))
        self.regen_temperature_delta = max(0.0, float(regen_temperature_delta))
        self.reward_gap_threshold = max(0.0, float(reward_gap_threshold))
        self.gap_sampling_temperature_delta = max(0.0, float(gap_sampling_temperature_delta))
        self.parallel_group_generate = parallel_group_generate

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
        allow_parallel: bool,
    ) -> list[object]:
        requested = max(1, int(num_samples))
        if (
            allow_parallel
            and requested > 1
            and self.parallel_group_generate
            and hasattr(self.model_wrapper, "generate_group_with_logprob")
        ):
            return self.model_wrapper.generate_group_with_logprob(
                prompt,
                num_return_sequences=requested,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=self.top_p,
            )
        return [
            self.model_wrapper.generate_with_logprob(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=self.top_p,
            )
            for _ in range(requested)
        ]

    def _generate_group_rollouts(self, prompt: str) -> list[object]:
        return self._generate_rollouts(
            prompt,
            num_samples=self.group_size,
            temperature=self.temperature,
            allow_parallel=True,
        )

    def _maybe_regenerate_group(
        self,
        *,
        prompt: str,
        source_query: str,
        generated_group: list[object],
        stabilized_group: list[object],
    ) -> tuple[list[object], list[object]]:
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
                if is_duplicate or is_polluted:
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
                    top_p=self.top_p,
                )
                regenerated_record = self._stabilize_generated_sample(source_query, regenerated)
                regenerated_is_polluted = (
                    regenerated_record.raw_contains_label
                    or regenerated_record.raw_multiline
                    or regenerated_record.raw_format_penalty > 0.0
                )

                should_replace = False
                if regenerated_record.final_query and regenerated_record.final_query not in other_queries:
                    should_replace = True
                elif current_is_polluted and not regenerated_is_polluted:
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
        # Keep rollout/logprob passes deterministic by disabling dropout.
        self.model_wrapper.actor_model.eval()
        try:
            self.optimizer.zero_grad(set_to_none=True)

            # We accumulate gradients per sample immediately to avoid keeping
            # all computation graphs in memory until the end of the batch.
            loss_values: list[float] = []
            loss_pg_terms: list[float] = []
            loss_kl_terms: list[float] = []

            rewards: list[float] = []
            mrr_scores: list[float] = []
            recall_scores: list[float] = []
            copy_penalties: list[float] = []
            format_penalties: list[float] = []
            duplicate_penalties: list[float] = []
            all_advantages: list[float] = []
            unique_final_query_counts: list[float] = []
            generated_sample_counts: list[float] = []
            reward_gap_raw_values: list[float] = []
            flat_reward_group_count = 0
            all_same_final_query_group_count = 0
            reward_gap_met_count = 0
            max_group_size_hit_count = 0
            extra_sample_count_total = 0

            best_query_pairs: list[dict[str, object]] = []
            group_query_summaries: list[dict[str, object]] = []

            valid_samples = 0
            sampled = 0
            num_groups = 0

            for query in batch_queries:
                num_groups += 1
                prompt = self.model_wrapper.build_prompt(query.text)
                group_samples: list[Sample] = []

                generated_group = self._generate_group_rollouts(prompt)
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
                self._update_reward_cache(
                    qid=query.qid,
                    source_query=query.text,
                    final_queries=final_query_group,
                    reward_by_query=reward_by_query,
                )
                reward_gap_raw = self._compute_reward_gap_raw(final_query_group, reward_by_query)
                gap_sampling_temperatures: list[float] = []

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
                        allow_parallel=False,
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

                duplicate_penalty_group = compute_group_duplicate_penalties(
                    final_query_group,
                    getattr(self.rewarder.cfg, "group_duplicate_penalty", 0.0),
                )
                unique_final_queries = list(dict.fromkeys(final_query_group))

                for generated, stabilized, duplicate_penalty in zip(
                    generated_group,
                    stabilized_group,
                    duplicate_penalty_group,
                ):
                    reward = reward_by_query[stabilized.final_query]
                    group_samples.append(
                        Sample(
                            qid=query.qid,
                            prompt=prompt,
                            response_text=generated.response_text,
                            cleaned_query=stabilized.cleaned_query,
                            final_query=stabilized.final_query,
                            response_token_ids=generated.response_token_ids,
                            logprob_old=generated.logprob_old,
                            reward=reward.total - duplicate_penalty,
                            mrr=reward.mrr,
                            recall=reward.recall,
                            copy_penalty=reward.copy_penalty,
                            format_penalty=reward.format_penalty,
                            duplicate_penalty=duplicate_penalty,
                            fallback_to_original=stabilized.fallback_to_original,
                            fallback_reasons=stabilized.fallback_reasons,
                        )
                    )
                sampled += len(group_samples)
                unique_final_query_counts.append(float(len(unique_final_queries)))
                if len(unique_final_queries) == 1:
                    all_same_final_query_group_count += 1
                if len({sample.reward for sample in group_samples}) == 1:
                    flat_reward_group_count += 1

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
                        "group_duplicate_penalties": [sample.duplicate_penalty for sample in group_samples],
                        "group_rewards": [sample.reward for sample in group_samples],
                        "group_mrr": [sample.mrr for sample in group_samples],
                        "group_recall": [sample.recall for sample in group_samples],
                        "group_copy_penalties": [sample.copy_penalty for sample in group_samples],
                        "group_format_penalties": [sample.format_penalty for sample in group_samples],
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

                advantages = normalize_advantages([sample.reward for sample in group_samples]).tolist()
                for sample, advantage in zip(group_samples, advantages):
                    sample.advantage = float(advantage)
                    all_advantages.append(sample.advantage)
                    rewards.append(sample.reward)
                    mrr_scores.append(sample.mrr)
                    recall_scores.append(sample.recall)
                    copy_penalties.append(sample.copy_penalty)
                    format_penalties.append(sample.format_penalty)
                    duplicate_penalties.append(sample.duplicate_penalty)

                for sample in group_samples:
                    if not sample.response_token_ids or sample.logprob_old.numel() == 0:
                        continue

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

            metrics: dict[str, object] = {
                "reward_mean": fmean(rewards) if rewards else 0.0,
                "mrr_mean": fmean(mrr_scores) if mrr_scores else 0.0,
                "recall_mean": fmean(recall_scores) if recall_scores else 0.0,
                "copy_penalty_mean": fmean(copy_penalties) if copy_penalties else 0.0,
                "format_penalty_mean": fmean(format_penalties) if format_penalties else 0.0,
                "duplicate_penalty_mean": fmean(duplicate_penalties) if duplicate_penalties else 0.0,
                "nonzero_reward_ratio": nonzero_reward_ratio,
                "adv_mean": fmean(all_advantages) if all_advantages else 0.0,
                "adv_std": float(torch.tensor(all_advantages).std(unbiased=False)) if all_advantages else 0.0,
                "unique_final_query_mean": fmean(unique_final_query_counts) if unique_final_query_counts else 0.0,
                "generated_sample_count_mean": fmean(generated_sample_counts) if generated_sample_counts else 0.0,
                "generated_sample_count_max": float(max(generated_sample_counts)) if generated_sample_counts else 0.0,
                "extra_sample_ratio": (extra_sample_count_total / sampled) if sampled else 0.0,
                "reward_gap_raw_mean": fmean(reward_gap_raw_values) if reward_gap_raw_values else 0.0,
                "reward_gap_met_ratio": (reward_gap_met_count / num_groups) if num_groups else 0.0,
                "max_group_size_hit_ratio": (max_group_size_hit_count / num_groups) if num_groups else 0.0,
                "all_same_final_query_ratio": (all_same_final_query_group_count / num_groups) if num_groups else 0.0,
                "flat_reward_group_ratio": (flat_reward_group_count / num_groups) if num_groups else 0.0,
                "sampled": float(sampled),
                "valid_samples": float(valid_samples),
                "group_query_summaries": group_query_summaries,
            }

            if valid_samples == 0:
                metrics.update(
                    {
                        "loss": 0.0,
                        "loss_pg": 0.0,
                        "loss_pg_abs_mean": 0.0,
                        "loss_kl": 0.0,
                        "kl_dominance_ratio": 0.0,
                        "updated": 0.0,
                    }
                )
                if collect_best_queries:
                    metrics["best_query_pairs"] = best_query_pairs
                return metrics

            # Convert accumulated sum-gradients into mean-gradients.
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
                    "loss": fmean(loss_values) if loss_values else 0.0,
                    "loss_pg": loss_pg_mean,
                    "loss_pg_abs_mean": loss_pg_abs_mean,
                    "loss_kl": loss_kl_mean,
                    "kl_dominance_ratio": kl_dominance_ratio,
                    "updated": 1.0,
                }
            )
            if collect_best_queries:
                metrics["best_query_pairs"] = best_query_pairs
            return metrics
        finally:
            self.model_wrapper.actor_model.train(previous_mode)
