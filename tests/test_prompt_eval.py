import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from app_config import RewardConfig
from prompt_eval.run_prompt_eval import (
    compute_metric_deltas,
    compute_win_tie_loss,
    filter_prompt_specs,
    rank_prompt_results,
    sample_eval_queries,
    select_best_prompt,
    stabilize_generated_rewrite,
)
from prompt_eval.prompt_bank import build_prompt_specs


@dataclass(frozen=True, slots=True)
class _Query:
    qid: str
    text: str


class PromptEvalHelperTests(unittest.TestCase):
    def test_prompt_bank_has_50_candidates_and_core_ids(self):
        prompts = build_prompt_specs("sys", "{query}")
        ids = {item.id for item in prompts}
        self.assertEqual(len(prompts), 50)
        self.assertIn("p01_det", ids)
        self.assertIn("p06_demo", ids)
        self.assertIn("p13_demo", ids)
        self.assertIn("p25_demo", ids)

    def test_prompt_bank_carries_guardrail_metadata(self):
        prompts = build_prompt_specs("sys", "{query}")
        spec = next(item for item in prompts if item.id == "p24_det")
        self.assertEqual(spec.min_terms, 2)
        self.assertEqual(spec.max_terms, 10)
        self.assertEqual(spec.fallback_mode, "conservative")

    def test_filter_prompt_specs_by_ids_and_limit(self):
        prompts = build_prompt_specs("sys", "{query}")
        selected = filter_prompt_specs(
            prompts,
            prompt_ids_csv="p01_det,p02_det,p03_det",
            max_prompts=2,
        )
        self.assertEqual([item.id for item in selected], ["p01_det", "p02_det"])

    def test_sample_eval_queries_reproducible_with_same_seed(self):
        queries = [_Query(qid=f"q{i}", text=f"query {i}") for i in range(20)]
        sample_a = sample_eval_queries(queries, sample_size=7, seed=42)
        sample_b = sample_eval_queries(queries, sample_size=7, seed=42)

        self.assertEqual([q.qid for q in sample_a], [q.qid for q in sample_b])

    def test_sample_eval_queries_changes_with_different_seed(self):
        queries = [_Query(qid=f"q{i}", text=f"query {i}") for i in range(20)]
        sample_a = sample_eval_queries(queries, sample_size=7, seed=11)
        sample_b = sample_eval_queries(queries, sample_size=7, seed=12)

        self.assertNotEqual([q.qid for q in sample_a], [q.qid for q in sample_b])

    def test_compute_metric_deltas(self):
        baseline = {"mrr": 0.2, "recall": 0.5, "reward_mean": 0.3}
        prompt = {"mrr": 0.25, "recall": 0.45, "reward_mean": 0.4}
        deltas = compute_metric_deltas(baseline, prompt)

        self.assertAlmostEqual(deltas["delta_mrr"], 0.05)
        self.assertAlmostEqual(deltas["delta_recall"], -0.05)
        self.assertAlmostEqual(deltas["delta_reward_mean"], 0.1)

    def test_compute_win_tie_loss_by_mrr(self):
        baseline_by_qid = {
            "q1": SimpleNamespace(mrr=0.1),
            "q2": SimpleNamespace(mrr=0.2),
            "q3": SimpleNamespace(mrr=0.3),
        }
        prompt_by_qid = {
            "q1": ("r1", SimpleNamespace(mrr=0.3)),
            "q2": ("r2", SimpleNamespace(mrr=0.2)),
            "q3": ("r3", SimpleNamespace(mrr=0.1)),
        }
        wtl = compute_win_tie_loss(baseline_by_qid, prompt_by_qid)

        self.assertEqual(wtl["win"], 1)
        self.assertEqual(wtl["tie"], 1)
        self.assertEqual(wtl["loss"], 1)
        self.assertEqual(wtl["total"], 3)
        self.assertAlmostEqual(wtl["win_rate"], 1.0 / 3.0)

    def test_best_prompt_selection_prefers_mrr_then_recall_then_reward(self):
        prompt_results = [
            {
                "prompt_id": "p1",
                "prompt_name": "P1",
                "metrics": {"mrr": 0.28, "recall": 0.60, "reward_mean": 0.60},
                "deltas": {"delta_mrr": 0.01, "delta_recall": 0.0, "delta_reward_mean": 0.0},
                "win_tie_loss": {"win": 1, "tie": 0, "loss": 0},
            },
            {
                "prompt_id": "p2",
                "prompt_name": "P2",
                "metrics": {"mrr": 0.30, "recall": 0.40, "reward_mean": 0.70},
                "deltas": {"delta_mrr": 0.03, "delta_recall": 0.0, "delta_reward_mean": 0.0},
                "win_tie_loss": {"win": 1, "tie": 0, "loss": 0},
            },
            {
                "prompt_id": "p3",
                "prompt_name": "P3",
                "metrics": {"mrr": 0.30, "recall": 0.50, "reward_mean": 0.40},
                "deltas": {"delta_mrr": 0.03, "delta_recall": 0.0, "delta_reward_mean": 0.0},
                "win_tie_loss": {"win": 1, "tie": 0, "loss": 0},
            },
            {
                "prompt_id": "p4",
                "prompt_name": "P4",
                "metrics": {"mrr": 0.30, "recall": 0.50, "reward_mean": 0.90},
                "deltas": {"delta_mrr": 0.03, "delta_recall": 0.0, "delta_reward_mean": 0.0},
                "win_tie_loss": {"win": 1, "tie": 0, "loss": 0},
            },
        ]

        best = select_best_prompt(prompt_results)
        leaderboard = rank_prompt_results(prompt_results)

        self.assertIsNotNone(best)
        self.assertEqual(best["prompt_id"], "p4")
        self.assertEqual(leaderboard[0]["prompt_id"], "p4")
        self.assertEqual(leaderboard[1]["prompt_id"], "p3")
        self.assertEqual(leaderboard[2]["prompt_id"], "p2")

    def test_stabilize_generated_rewrite_falls_back_on_polluted_output(self):
        spec = next(item for item in build_prompt_specs("sys", "{query}") if item.id == "p01_det")
        record = stabilize_generated_rewrite(
            "Assistant: because this query is better",
            source_query="guayana venezuela",
            prompt_spec=spec,
            reward_cfg=RewardConfig(),
        )

        self.assertTrue(record.fallback_to_original)
        self.assertEqual(record.final_query, "guayana venezuela")
        self.assertIn("format_fail", record.fallback_reasons)

    def test_stabilize_generated_rewrite_preserves_locked_terms(self):
        spec = next(item for item in build_prompt_specs("sys", "{query}") if item.id == "p05_det")
        record = stabilize_generated_rewrite(
            "best laptop",
            source_query="best laptop under 1000 2024",
            prompt_spec=spec,
            reward_cfg=RewardConfig(),
        )

        self.assertTrue(record.fallback_to_original)
        self.assertEqual(record.final_query, "best laptop under 1000 2024")
        self.assertIn("lost_numeric", record.fallback_reasons)


if __name__ == "__main__":
    unittest.main()
