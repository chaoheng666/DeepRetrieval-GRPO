import unittest

import torch

from app_config import PromptConfig, RewardConfig
from core.grpo_engine import normalize_advantages, ppo_clipped_objective
from core.reward_func import (
    Rewarder,
    clean_rewritten_query,
    compose_reward,
    compute_copy_penalty,
    compute_format_penalty,
    compute_mrr_at_k,
    compute_recall_at_k,
    stabilize_generated_rewrite,
)


class AdvantageTests(unittest.TestCase):
    def test_normalize_advantages_zero_std(self):
        adv = normalize_advantages([1.0, 1.0, 1.0])
        self.assertTrue(torch.allclose(adv, torch.zeros_like(adv)))

    def test_normalize_advantages_centered(self):
        adv = normalize_advantages([0.0, 1.0, 2.0])
        self.assertAlmostEqual(float(adv.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(adv.std(unbiased=False)), 1.0, places=5)


class PpoClipTests(unittest.TestCase):
    def test_clipped_objective_positive_advantage(self):
        logprob_old = torch.log(torch.tensor([0.5, 0.5], dtype=torch.float32))
        logprob_new = torch.log(torch.tensor([0.75, 0.25], dtype=torch.float32))
        objective = ppo_clipped_objective(logprob_new, logprob_old, advantage=1.0, clip_range=0.2)
        self.assertEqual(objective.shape[0], 2)
        self.assertTrue(torch.allclose(objective, torch.tensor([1.2, 0.5]), atol=1e-5))


class RewardMathTests(unittest.TestCase):
    def test_compute_mrr_at_k_hit(self):
        mrr, rank = compute_mrr_at_k(["D1", "D2", "D3"], {"D3"}, topk=10)
        self.assertAlmostEqual(mrr, 1.0 / 3.0)
        self.assertEqual(rank, 3)

    def test_compute_mrr_at_k_miss(self):
        mrr, rank = compute_mrr_at_k(["D1", "D2"], {"D3"}, topk=10)
        self.assertEqual(mrr, 0.0)
        self.assertIsNone(rank)

    def test_compute_recall_at_k_hit(self):
        recall, hit_count, total = compute_recall_at_k(["D1", "D2", "D3"], {"D2", "D9"}, topk=3)
        self.assertAlmostEqual(recall, 0.5)
        self.assertEqual(hit_count, 1)
        self.assertEqual(total, 2)

    def test_compute_recall_at_k_miss(self):
        recall, hit_count, total = compute_recall_at_k(["D1", "D2"], {"D9"}, topk=2)
        self.assertEqual(recall, 0.0)
        self.assertEqual(hit_count, 0)
        self.assertEqual(total, 1)

    def test_compute_recall_at_k_multiple_hits(self):
        recall, hit_count, total = compute_recall_at_k(["D1", "D2", "D3", "D4"], {"D2", "D4"}, topk=4)
        self.assertAlmostEqual(recall, 1.0)
        self.assertEqual(hit_count, 2)
        self.assertEqual(total, 2)

    def test_copy_penalty_piecewise(self):
        self.assertEqual(compute_copy_penalty(0.55, 0.6), 0.0)
        self.assertAlmostEqual(compute_copy_penalty(0.8, 0.6), 0.2)

    def test_format_penalty_empty(self):
        cfg = RewardConfig()
        self.assertEqual(compute_format_penalty("", cfg), 1.0)

    def test_format_penalty_multiline(self):
        cfg = RewardConfig()
        self.assertEqual(compute_format_penalty("first line\nsecond line", cfg), 1.0)

    def test_format_penalty_explanation(self):
        cfg = RewardConfig()
        self.assertEqual(compute_format_penalty("because this query is better", cfg), 1.0)

    def test_format_penalty_non_english(self):
        cfg = RewardConfig()
        self.assertEqual(compute_format_penalty("天气 预报 北京 明天", cfg), 1.0)

    def test_format_penalty_overlength(self):
        cfg = RewardConfig(format_max_tokens=3)
        self.assertEqual(compute_format_penalty("best budget gaming laptop 2024", cfg), 1.0)

    def test_format_penalty_unreadable(self):
        cfg = RewardConfig(format_max_unreadable_ratio=0.0)
        self.assertEqual(compute_format_penalty("normal § query", cfg), 1.0)

    def test_format_penalty_valid_query(self):
        cfg = RewardConfig()
        self.assertEqual(compute_format_penalty("best budget gaming laptop 2024", cfg), 0.0)

    def test_total_reward_formula(self):
        cfg = RewardConfig(w_mrr=1.0, w_recall=0.3, w_copy=0.15, w_format=0.2)
        total = compose_reward(mrr=0.5, recall=0.4, copy_penalty=0.1, format_penalty=1.0, cfg=cfg)
        expected = 1.0 * 0.5 + 0.3 * 0.4 - 0.15 * 0.1 - 0.2 * 1.0
        self.assertAlmostEqual(total, expected)


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

    def test_clean_rewritten_query_discards_prompt_template_leakage(self):
        raw = (
            "finderscope\n\n"
            "Example\n"
            "User query: what is a finderscope\n"
            "Better BM25 query:"
        )
        cleaned = clean_rewritten_query(raw, source_query="what is a finderscope")
        self.assertEqual(cleaned, "finderscope")

    def test_stabilize_generated_rewrite_falls_back_when_numeric_is_lost(self):
        cfg = RewardConfig()
        prompt_cfg = PromptConfig(min_terms=3, max_terms=12, fallback_mode="conservative")
        record = stabilize_generated_rewrite(
            "best laptop under 1000",
            source_query="best laptop under 1000 2024",
            guardrail_cfg=prompt_cfg,
            reward_cfg=cfg,
        )
        self.assertTrue(record.fallback_to_original)
        self.assertIn("lost_numeric", record.fallback_reasons)
        self.assertEqual(record.final_query, "best laptop under 1000 2024")

    def test_stabilize_generated_rewrite_falls_back_when_acronym_or_negation_is_lost(self):
        cfg = RewardConfig()
        prompt_cfg = PromptConfig(min_terms=3, max_terms=12, fallback_mode="balanced")

        acronym_record = stabilize_generated_rewrite(
            "chronic obstructive pulmonary disease treatment options",
            source_query="COPD treatment options",
            guardrail_cfg=prompt_cfg,
            reward_cfg=cfg,
        )
        self.assertTrue(acronym_record.fallback_to_original)
        self.assertIn("lost_acronym", acronym_record.fallback_reasons)
        self.assertEqual(acronym_record.final_query, "COPD treatment options")

        negation_record = stabilize_generated_rewrite(
            "foods gluten",
            source_query="foods without gluten",
            guardrail_cfg=prompt_cfg,
            reward_cfg=cfg,
        )
        self.assertTrue(negation_record.fallback_to_original)
        self.assertIn("lost_negation", negation_record.fallback_reasons)
        self.assertEqual(negation_record.final_query, "foods without gluten")

    def test_stabilize_generated_rewrite_falls_back_on_format_fail(self):
        cfg = RewardConfig()
        prompt_cfg = PromptConfig(min_terms=3, max_terms=11, fallback_mode="balanced")
        record = stabilize_generated_rewrite(
            "because this query is better",
            source_query="best budget gaming laptop 2024",
            guardrail_cfg=prompt_cfg,
            reward_cfg=cfg,
        )
        self.assertTrue(record.fallback_to_original)
        self.assertIn("format_fail", record.fallback_reasons)
        self.assertEqual(record.final_query, "best budget gaming laptop 2024")

    def test_stabilize_generated_rewrite_falls_back_on_conservative_divergence(self):
        cfg = RewardConfig()
        prompt_cfg = PromptConfig(min_terms=2, max_terms=11, fallback_mode="conservative")
        record = stabilize_generated_rewrite(
            "south america regional dispute",
            source_query="guayana venezuela",
            guardrail_cfg=prompt_cfg,
            reward_cfg=cfg,
        )
        self.assertTrue(record.fallback_to_original)
        self.assertIn("diverged_from_lexical_source", record.fallback_reasons)
        self.assertEqual(record.final_query, "guayana venezuela")


class RewarderConsistencyTests(unittest.TestCase):
    def test_score_cleans_query_before_search(self):
        rewarder = Rewarder.__new__(Rewarder)
        seen: dict[str, object] = {}

        def _search_docids(query: str):
            seen["search_query"] = query
            return ["D1"]

        def _score_one(qid: str, cleaned_query: str, hits_docids: list[str], source_query: str | None):
            seen["score_query"] = cleaned_query
            seen["score_hits"] = list(hits_docids)
            seen["score_qid"] = qid
            seen["score_source"] = source_query
            return cleaned_query

        rewarder._search_docids = _search_docids  # type: ignore[attr-defined]
        rewarder._score_one = _score_one  # type: ignore[attr-defined]

        raw = "Search Query:\n  best budget gaming laptop 2024\nExplanation: keep concise"
        scored = rewarder.score("q1", raw, source_query="best budget gaming laptop 2024")

        self.assertEqual(scored, "best budget gaming laptop 2024")
        self.assertEqual(seen["search_query"], "best budget gaming laptop 2024")
        self.assertEqual(seen["score_query"], "best budget gaming laptop 2024")
        self.assertEqual(seen["score_hits"], ["D1"])
        self.assertEqual(seen["score_qid"], "q1")
        self.assertEqual(seen["score_source"], "best budget gaming laptop 2024")

    def test_score_batch_cleans_queries_before_batch_search(self):
        rewarder = Rewarder.__new__(Rewarder)
        seen: dict[str, object] = {}

        def _search_docids_batch(queries: list[str]):
            seen["search_queries"] = list(queries)
            return [["D1"], ["D2"]]

        def _score_one(qid: str, cleaned_query: str, hits_docids: list[str], source_query: str | None):
            return {
                "qid": qid,
                "query": cleaned_query,
                "hits": list(hits_docids),
                "source": source_query,
            }

        rewarder._search_docids_batch = _search_docids_batch  # type: ignore[attr-defined]
        rewarder._score_one = _score_one  # type: ignore[attr-defined]

        raw_queries = [
            "Search Query:\n  best budget gaming laptop 2024\nExplanation: keep concise",
            "mastoidectomy\n\nUser query: what is the capital of the united states\nBetter BM25",
        ]
        scored = rewarder.score_batch("q1", raw_queries, source_query="best budget gaming laptop 2024")

        self.assertEqual(
            seen["search_queries"],
            ["best budget gaming laptop 2024", "mastoidectomy"],
        )
        self.assertEqual(scored[0]["query"], "best budget gaming laptop 2024")
        self.assertEqual(scored[1]["query"], "mastoidectomy")


if __name__ == "__main__":
    unittest.main()
