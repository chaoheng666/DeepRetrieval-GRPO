from __future__ import annotations

"""GRPO 训练核心模块（纯 PyTorch 实现）。"""

from dataclasses import dataclass
from statistics import fmean
from typing import Sequence

import torch
from torch.nn.utils import clip_grad_norm_

from data.loader import QueryExample

from .reward_func import compute_unreadable_ratio


@dataclass(slots=True)
class Sample:
    """单条采样样本，承载 loss 计算与日志所需字段。"""

    qid: str
    prompt: str
    response_text: str
    rewritten_query: str
    response_token_ids: list[int]
    logprob_old: torch.Tensor
    reward: float
    mrr: float
    overlap: float
    penalty: float
    unreadable_penalty: float
    unreadable_ratio: float
    advantage: float = 0.0


def normalize_advantages(rewards: Sequence[float], eps: float = 1e-8) -> torch.Tensor:
    """组内奖励标准化，得到相对优势。"""

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
    """计算 token 级 PPO clipped surrogate objective。"""

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
    ) -> None:
        """初始化训练引擎。

        这里不负责解析配置，仅接收 train.py 注入的超参数。
        """

        self.model_wrapper = model_wrapper
        self.rewarder = rewarder
        self.optimizer = optimizer
        self.group_size = group_size
        self.clip_range = clip_range
        self.kl_beta = kl_beta
        self.grad_clip_norm = grad_clip_norm
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def train_step(
        self,
        batch_queries: Sequence[QueryExample],
        *,
        collect_best_queries: bool = False,
    ) -> dict[str, object]:
        """执行一个 batch 的 GRPO 更新。"""

        previous_mode = self.model_wrapper.actor_model.training
        # rollout 与重算 logprob 阶段关闭 dropout，降低采样噪声。
        self.model_wrapper.actor_model.eval()
        try:
            self.optimizer.zero_grad(set_to_none=True)

            loss_terms: list[torch.Tensor] = []
            loss_pg_terms: list[float] = []
            loss_kl_terms: list[float] = []

            rewards: list[float] = []
            mrr_scores: list[float] = []
            penalties: list[float] = []
            overlaps: list[float] = []
            unreadable_ratios: list[float] = []
            all_advantages: list[float] = []

            best_query_pairs: list[dict[str, object]] = []
            group_query_summaries: list[dict[str, object]] = []

            valid_samples = 0
            sampled = 0

            for query in batch_queries:
                prompt = self.model_wrapper.build_prompt(query.text)
                group_samples: list[Sample] = []

                for _ in range(self.group_size):
                    # Step 1) 组内采样，记录 old-policy logprob。
                    generated = self.model_wrapper.generate_with_logprob(
                        prompt,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                    reward = self.rewarder.score(query.qid, generated.response_text, source_query=query.text)
                    group_samples.append(
                        Sample(
                            qid=query.qid,
                            prompt=prompt,
                            response_text=generated.response_text,
                            rewritten_query=reward.rewritten_query,
                            response_token_ids=generated.response_token_ids,
                            logprob_old=generated.logprob_old,
                            reward=reward.total,
                            mrr=reward.mrr,
                            overlap=reward.overlap,
                            penalty=reward.penalty,
                            unreadable_penalty=reward.unreadable_penalty,
                            unreadable_ratio=compute_unreadable_ratio(generated.response_text),
                        )
                    )
                    sampled += 1

                group_query_summaries.append(
                    {
                        # 每个 query 一条汇总日志，便于排查 4bit 乱码样本。
                        "qid": str(query.qid),
                        "input_query": query.text,
                        "group_raw_responses": [sample.response_text for sample in group_samples],
                        "group_cleaned_queries": [sample.rewritten_query for sample in group_samples],
                        "group_rewards": [sample.reward for sample in group_samples],
                        "group_mrr": [sample.mrr for sample in group_samples],
                        "group_penalties": [sample.penalty for sample in group_samples],
                        "group_unreadable_penalties": [sample.unreadable_penalty for sample in group_samples],
                    }
                )

                if collect_best_queries and group_samples:
                    # ties 时 max 保留先出现的候选。
                    best_sample = max(group_samples, key=lambda sample: sample.reward)
                    best_query_pairs.append(
                        {
                            "qid": str(query.qid),
                            "input_query": query.text,
                            "best_rewritten_query": best_sample.rewritten_query,
                            "best_reward": best_sample.reward,
                        }
                    )

                # Step 2) 组内优势标准化（GRPO 核心）。
                advantages = normalize_advantages([sample.reward for sample in group_samples]).tolist()
                for sample, advantage in zip(group_samples, advantages):
                    sample.advantage = float(advantage)
                    all_advantages.append(sample.advantage)
                    rewards.append(sample.reward)
                    mrr_scores.append(sample.mrr)
                    penalties.append(sample.penalty)
                    overlaps.append(sample.overlap)
                    unreadable_ratios.append(sample.unreadable_ratio)

                for sample in group_samples:
                    if not sample.response_token_ids or sample.logprob_old.numel() == 0:
                        continue

                    # Step 3) 新策略与参考策略的 token 级 logprob。
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

                    # Step 4) PPO clipped loss + KL 正则。
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

                    loss_terms.append(loss)
                    loss_pg_terms.append(float(loss_pg.detach().cpu()))
                    loss_kl_terms.append(float(loss_kl.detach().cpu()))
                    valid_samples += 1

            nonzero_reward_ratio = (sum(1 for value in rewards if value > 0.0) / len(rewards)) if rewards else 0.0

            metrics: dict[str, object] = {
                "reward_mean": fmean(rewards) if rewards else 0.0,
                "mrr_mean": fmean(mrr_scores) if mrr_scores else 0.0,
                "penalty_mean": fmean(penalties) if penalties else 0.0,
                "overlap_mean": fmean(overlaps) if overlaps else 0.0,
                "unreadable_ratio_mean": fmean(unreadable_ratios) if unreadable_ratios else 0.0,
                "nonzero_reward_ratio": nonzero_reward_ratio,
                "adv_mean": fmean(all_advantages) if all_advantages else 0.0,
                "adv_std": float(torch.tensor(all_advantages).std(unbiased=False)) if all_advantages else 0.0,
                "sampled": float(sampled),
                "valid_samples": float(valid_samples),
                "group_query_summaries": group_query_summaries,
            }

            if not loss_terms:
                # 没有可训练 token 样本时返回指标但不更新参数。
                metrics.update(
                    {
                        "loss": 0.0,
                        "loss_pg": 0.0,
                        "loss_kl": 0.0,
                        "updated": 0.0,
                    }
                )
                if collect_best_queries:
                    metrics["best_query_pairs"] = best_query_pairs
                return metrics

            loss_batch = torch.stack(loss_terms).mean()
            if not torch.isfinite(loss_batch):
                # 聚合后再次做 NaN/Inf 保护。
                metrics.update(
                    {
                        "loss": float("nan"),
                        "loss_pg": float("nan"),
                        "loss_kl": float("nan"),
                        "updated": 0.0,
                    }
                )
                if collect_best_queries:
                    metrics["best_query_pairs"] = best_query_pairs
                return metrics

            # Step 5) 反传 + 梯度裁剪 + 更新参数。
            loss_batch.backward()
            clip_grad_norm_(self.model_wrapper.trainable_parameters(), self.grad_clip_norm)
            self.optimizer.step()

            metrics.update(
                {
                    "loss": float(loss_batch.detach().cpu()),
                    "loss_pg": fmean(loss_pg_terms) if loss_pg_terms else 0.0,
                    "loss_kl": fmean(loss_kl_terms) if loss_kl_terms else 0.0,
                    "updated": 1.0,
                }
            )
            if collect_best_queries:
                metrics["best_query_pairs"] = best_query_pairs
            return metrics
        finally:
            self.model_wrapper.actor_model.train(previous_mode)
