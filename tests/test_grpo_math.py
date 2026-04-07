import unittest

import torch

from app_config import RewardConfig
from core.grpo_engine import normalize_advantages, ppo_clipped_objective
from core.reward_func import clean_rewritten_query, compute_mrr_at_k, compute_text_penalty


class AdvantageTests(unittest.TestCase):
    """优势归一化单元测试。"""

    def test_normalize_advantages_zero_std(self):
        # 组内奖励全部相同，标准差为 0，优势应全部回落为 0。
        adv = normalize_advantages([1.0, 1.0, 1.0])
        self.assertTrue(torch.allclose(adv, torch.zeros_like(adv)))

    def test_normalize_advantages_centered(self):
        # 标准化后，均值应接近 0，方差应接近 1。
        adv = normalize_advantages([0.0, 1.0, 2.0])
        self.assertAlmostEqual(float(adv.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(adv.std(unbiased=False)), 1.0, places=5)


class PpoClipTests(unittest.TestCase):
    """PPO clip 公式单元测试。"""

    def test_clipped_objective_positive_advantage(self):
        logprob_old = torch.log(torch.tensor([0.5, 0.5], dtype=torch.float32))
        logprob_new = torch.log(torch.tensor([0.75, 0.25], dtype=torch.float32))
        objective = ppo_clipped_objective(logprob_new, logprob_old, advantage=1.0, clip_range=0.2)
        self.assertEqual(objective.shape[0], 2)
        # ratio = [1.5, 0.5], clipped to [1.2, 0.8], min picks [1.2, 0.5]
        self.assertTrue(torch.allclose(objective, torch.tensor([1.2, 0.5]), atol=1e-5))


class RewardMathTests(unittest.TestCase):
    """MRR 与文本惩罚单元测试。"""

    def test_compute_mrr_at_k_hit(self):
        mrr, rank = compute_mrr_at_k(["D1", "D2", "D3"], {"D3"}, topk=10)
        self.assertAlmostEqual(mrr, 1.0 / 3.0)
        self.assertEqual(rank, 3)

    def test_compute_mrr_at_k_miss(self):
        mrr, rank = compute_mrr_at_k(["D1", "D2"], {"D3"}, topk=10)
        self.assertEqual(mrr, 0.0)
        self.assertIsNone(rank)

    def test_text_penalty_short_and_repeat(self):
        cfg = RewardConfig(min_query_chars=3, max_repeat_ratio=0.2, penalty_short=0.5, penalty_repeat=0.3)
        p = compute_text_penalty("aa aa aa", cfg)
        self.assertGreaterEqual(p.repeat, 0.3)
        self.assertEqual(p.short, 0.0)


class QueryCleaningTests(unittest.TestCase):
    def test_clean_rewritten_query_prefers_high_overlap_complete_candidate(self):
        raw = (
            "Androgen receptor definition\n\n"
            "Rewritten Query: Definition of the androgen receptor protein\n\n"
            "Rewritten Query: Information on the"
        )
        cleaned = clean_rewritten_query(raw, source_query="Androgen receptor define")
        self.assertEqual(cleaned, "Androgen receptor definition")

    def test_clean_rewritten_query_handles_single_line_quotes(self):
        raw = '   "causes   of climate change 2024 report"   '
        cleaned = clean_rewritten_query(raw)
        self.assertEqual(cleaned, "causes of climate change 2024 report")

    def test_clean_rewritten_query_uses_marker_next_line(self):
        raw = (
            "Search Query:\n"
            "  best budget gaming laptop 2024\n"
            "\n"
            "Explanation: keep under $1000"
        )
        cleaned = clean_rewritten_query(raw, source_query="best budget gaming laptop 2024")
        self.assertEqual(cleaned, "best budget gaming laptop 2024")

    def test_clean_rewritten_query_without_source_is_deterministic(self):
        raw = (
            "Rewritten Query: travel insurance for japan\n"
            "Search Query: travel insurance japan coverage"
        )
        cleaned = clean_rewritten_query(raw)
        self.assertEqual(cleaned, "travel insurance for japan")


if __name__ == "__main__":
    unittest.main()
