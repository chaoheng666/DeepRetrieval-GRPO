from __future__ import annotations

"""GRPO 训练核心模块。

本模块实现了你要求的“纯 PyTorch 自写 GRPO”关键流程，不依赖第三方 RL 框架：

1. 组采样（每个 query 采样 K 个 response）
2. 组内优势归一化（relative advantage）
3. PPO clipped policy objective
4. 与参考策略的 KL 正则项
5. 反向传播、梯度裁剪与优化器更新
"""

from dataclasses import dataclass
from statistics import fmean
from typing import Sequence

import torch
from torch.nn.utils import clip_grad_norm_

from data.loader import QueryExample


@dataclass(slots=True)
class Sample:
    """单条采样结果。

    该结构把训练所需字段放在一起，方便从采样阶段流向 loss 计算阶段：
    - response_token_ids / logprob_old：用于 PPO ratio
    - reward / mrr / penalty：用于训练监控和优势归一化
    - advantage：组内归一化后的相对优势
    """

    qid: str
    prompt: str
    response_text: str
    response_token_ids: list[int]
    logprob_old: torch.Tensor
    reward: float
    mrr: float
    penalty: float
    advantage: float = 0.0


def normalize_advantages(rewards: Sequence[float], eps: float = 1e-8) -> torch.Tensor:
    """组内奖励标准化，得到相对优势。

    公式：
      adv_i = (r_i - mean(r)) / (std(r) + eps)

    数值稳定性：
    - 若 std 非常小（接近 0），返回全 0，避免除零和极端梯度。
    """

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
    """计算 token 级 PPO clipped surrogate objective。

    ratio = exp(logprob_new - logprob_old)
    objective = min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv)

    返回逐 token 的 objective，调用方再做 mean 和负号得到 loss_pg。
    """

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

        参数由 train.py 统一注入，避免在此模块里关心 CLI 与配置读取细节。
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

    def train_step(self, batch_queries: Sequence[QueryExample]) -> dict[str, float]:
        """执行一个 batch 的 GRPO 更新步骤。

        每条 query 的流程：
        1. 从当前策略采样 K 个重写结果（并记录采样时 old logprob）
        2. 计算每条样本的序列级奖励（MRR - penalty）
        3. 在组内做奖励标准化，得到 relative advantage
        4. 重新前向计算：
           - logprob_new（当前 actor）
           - logprob_ref（冻结参考策略）
           并构造 loss_pg + loss_kl
        5. 聚合整个 batch 的损失，反传并更新参数
        """

        self.model_wrapper.actor_model.train(True)
        self.optimizer.zero_grad(set_to_none=True)

        loss_terms: list[torch.Tensor] = []
        loss_pg_terms: list[float] = []
        loss_kl_terms: list[float] = []
        rewards: list[float] = []
        mrr_scores: list[float] = []
        penalties: list[float] = []
        all_advantages: list[float] = []
        valid_samples = 0
        sampled = 0

        for query in batch_queries:
            prompt = self.model_wrapper.build_prompt(query.text)
            group_samples: list[Sample] = []

            for _ in range(self.group_size):
                # Step 1: 组内采样。这里得到的是“旧策略概率”（采样时记录）。
                generated = self.model_wrapper.generate_with_logprob(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                # Step 2: 序列级奖励（检索质量 + 文本质量）。
                reward = self.rewarder.score(query.qid, generated.response_text)
                sample = Sample(
                    qid=query.qid,
                    prompt=prompt,
                    response_text=generated.response_text,
                    response_token_ids=generated.response_token_ids,
                    logprob_old=generated.logprob_old,
                    reward=reward.total,
                    mrr=reward.mrr,
                    penalty=reward.penalty,
                )
                group_samples.append(sample)
                sampled += 1

            # Step 3: GRPO 核心——组内相对优势，而不是显式 value function。
            advantages = normalize_advantages([s.reward for s in group_samples]).tolist()
            for sample, advantage in zip(group_samples, advantages):
                sample.advantage = float(advantage)
                all_advantages.append(sample.advantage)
                rewards.append(sample.reward)
                mrr_scores.append(sample.mrr)
                penalties.append(sample.penalty)

            for sample in group_samples:
                # 空生成无法做 token-level 更新，直接跳过。
                if not sample.response_token_ids or sample.logprob_old.numel() == 0:
                    continue

                # Step 4a: 新策略概率（保留梯度，用于更新 actor）。
                logprob_new = self.model_wrapper.compute_logprob(
                    sample.prompt,
                    sample.response_token_ids,
                    policy="actor",
                    no_grad=False,
                )
                # Step 4b: 参考策略概率（冻结，不回传梯度）。
                logprob_ref = self.model_wrapper.compute_logprob(
                    sample.prompt,
                    sample.response_token_ids,
                    policy="ref",
                    no_grad=True,
                )
                # 保护措施：采样阶段和重算阶段 token 数可能存在轻微不一致，
                # 统一截断到最短长度，确保张量对齐可计算。
                t = min(logprob_new.numel(), sample.logprob_old.numel(), logprob_ref.numel())
                if t == 0:
                    continue

                logprob_new = logprob_new[:t]
                logprob_old = sample.logprob_old[:t].to(logprob_new.device)
                logprob_ref = logprob_ref[:t].to(logprob_new.device)

                # Step 4c: PPO clipped policy loss。
                clipped_obj = ppo_clipped_objective(
                    logprob_new=logprob_new,
                    logprob_old=logprob_old,
                    advantage=sample.advantage,
                    clip_range=self.clip_range,
                )
                loss_pg = -clipped_obj.mean()
                # Step 4d: KL 正则项，约束新策略不要偏离 ref 过快。
                loss_kl = self.kl_beta * (logprob_new - logprob_ref).mean()
                loss = loss_pg + loss_kl

                if not torch.isfinite(loss):
                    # 出现 NaN/Inf 时跳过该样本，避免污染优化器状态。
                    continue

                loss_terms.append(loss)
                loss_pg_terms.append(float(loss_pg.detach().cpu()))
                loss_kl_terms.append(float(loss_kl.detach().cpu()))
                valid_samples += 1

        if not loss_terms:
            # 没有可用的 token-level 样本，返回指标但不更新参数。
            return {
                "loss": 0.0,
                "loss_pg": 0.0,
                "loss_kl": 0.0,
                "reward_mean": fmean(rewards) if rewards else 0.0,
                "mrr_mean": fmean(mrr_scores) if mrr_scores else 0.0,
                "penalty_mean": fmean(penalties) if penalties else 0.0,
                "adv_mean": fmean(all_advantages) if all_advantages else 0.0,
                "adv_std": float(torch.tensor(all_advantages).std(unbiased=False)) if all_advantages else 0.0,
                "sampled": float(sampled),
                "valid_samples": 0.0,
                "updated": 0.0,
            }

        loss_batch = torch.stack(loss_terms).mean()
        if not torch.isfinite(loss_batch):
            # 聚合后再次做数值检查，双重保险。
            return {
                "loss": float("nan"),
                "loss_pg": float("nan"),
                "loss_kl": float("nan"),
                "reward_mean": fmean(rewards) if rewards else 0.0,
                "mrr_mean": fmean(mrr_scores) if mrr_scores else 0.0,
                "penalty_mean": fmean(penalties) if penalties else 0.0,
                "adv_mean": fmean(all_advantages) if all_advantages else 0.0,
                "adv_std": float(torch.tensor(all_advantages).std(unbiased=False)) if all_advantages else 0.0,
                "sampled": float(sampled),
                "valid_samples": float(valid_samples),
                "updated": 0.0,
            }

        # Step 5: 反向传播 + 梯度裁剪 + 参数更新。
        loss_batch.backward()
        # 只裁剪可训练参数（通常是 LoRA 参数）。
        clip_grad_norm_(self.model_wrapper.trainable_parameters(), self.grad_clip_norm)
        self.optimizer.step()

        return {
            "loss": float(loss_batch.detach().cpu()),
            "loss_pg": fmean(loss_pg_terms) if loss_pg_terms else 0.0,
            "loss_kl": fmean(loss_kl_terms) if loss_kl_terms else 0.0,
            "reward_mean": fmean(rewards) if rewards else 0.0,
            "mrr_mean": fmean(mrr_scores) if mrr_scores else 0.0,
            "penalty_mean": fmean(penalties) if penalties else 0.0,
            "adv_mean": fmean(all_advantages) if all_advantages else 0.0,
            "adv_std": float(torch.tensor(all_advantages).std(unbiased=False)) if all_advantages else 0.0,
            "sampled": float(sampled),
            "valid_samples": float(valid_samples),
            "updated": 1.0,
        }
