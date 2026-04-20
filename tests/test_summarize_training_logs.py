import unittest

from summarize_training_logs import build_group_steps, build_summary, build_train_steps, merge_steps


class SummarizeTrainingLogsTests(unittest.TestCase):
    def test_legacy_train_metrics_fallback_to_canonical_top20_fields(self):
        train_rows = [
            {
                "phase": "train",
                "epoch": 1,
                "step": 10,
                "reward_mean": 0.5,
                "mrr_mean": 0.2,
                "recall_mean": 0.3,
                "recall_dense_mean": 0.4,
                "flat_reward_group_ratio": 0.25,
                "sampled": 8,
                "valid_samples": 8,
                "loss": 0.1,
                "loss_pg": 0.05,
                "loss_kl": 0.05,
            }
        ]
        group_rows = [
            {
                "epoch": 1,
                "step": 10,
                "group_rewards": [0.5, 0.7],
                "group_mrr": [0.2, 0.3],
                "group_recall": [0.1, 0.3],
                "group_recall_dense": [0.4, 0.6],
                "group_copy_penalties": [1.0, 0.0],
                "group_final_queries": ["query a", "query b"],
                "group_fallback_to_original": [False, True],
                "collapsed_group": False,
                "reward_gap_met": True,
                "reward_gap_stop_reason": "threshold_reached",
                "reward_gap_raw": 0.2,
                "reward_gap_threshold": 0.08,
                "initial_group_size": 2,
                "generated_sample_count": 2,
                "extra_sample_count": 0,
            }
        ]

        train_steps = build_train_steps(train_rows)
        group_steps = build_group_steps(group_rows)
        merged_steps = merge_steps(train_steps, group_steps)
        summary = build_summary(train_rows, group_rows, merged_steps)

        self.assertEqual(train_steps[0]["rewrite_mrr20_mean"], 0.2)
        self.assertEqual(train_steps[0]["rewrite_recall20_mean"], 0.3)
        self.assertEqual(train_steps[0]["rewrite_recall50_mean"], 0.4)
        self.assertEqual(train_steps[0]["main_reward_mean"], 0.5)
        self.assertEqual(train_steps[0]["flat_main_reward_group_ratio"], 0.25)
        self.assertAlmostEqual(group_steps[0]["trace_unsafe_copy_penalty_mean"], 0.5)
        self.assertIsNone(group_steps[0]["trace_orig_mrr20_mean"])
        self.assertEqual(summary["best_train_rewrite_mrr20_mean"]["rewrite_mrr20_mean"], 0.2)
        self.assertEqual(
            summary["lowest_flat_main_reward_group_ratio"]["flat_main_reward_group_ratio"],
            0.25,
        )
        self.assertIsNone(summary["averages"]["trace_orig_mrr20_mean"])

    def test_new_fields_take_precedence_over_legacy_aliases(self):
        train_rows = [
            {
                "phase": "train",
                "epoch": 2,
                "step": 5,
                "reward_mean": 0.1,
                "mrr_mean": 0.2,
                "recall_mean": 0.3,
                "recall_dense_mean": 0.4,
                "rewrite_mrr20_mean": 0.7,
                "rewrite_recall20_mean": 0.8,
                "rewrite_recall50_mean": 0.9,
                "main_reward_mean": 0.6,
                "flat_reward_group_ratio": 0.4,
                "flat_main_reward_group_ratio": 0.15,
                "sampled": 4,
                "valid_samples": 4,
                "loss": 0.2,
                "loss_pg": 0.1,
                "loss_kl": 0.1,
            }
        ]
        group_rows = [
            {
                "epoch": 2,
                "step": 5,
                "group_rewards": [0.1, 0.2],
                "group_main_rewards": [0.05, 0.15],
                "group_mrr": [0.7, 0.8],
                "group_recall": [0.2, 0.4],
                "group_recall_dense": [0.5, 0.9],
                "group_orig_mrr": [0.1, 0.2],
                "group_delta_mrr": [0.6, 0.6],
                "group_unsafe_copy_penalties": [0.25, 0.75],
                "group_final_queries": ["query c", "query d"],
                "group_fallback_to_original": [False, False],
                "collapsed_group": False,
                "reward_gap_met": True,
                "reward_gap_stop_reason": "threshold_reached",
                "reward_gap_raw": 0.1,
                "reward_gap_threshold": 0.08,
                "initial_group_size": 2,
                "generated_sample_count": 2,
                "extra_sample_count": 0,
            }
        ]

        train_steps = build_train_steps(train_rows)
        group_steps = build_group_steps(group_rows)

        self.assertEqual(train_steps[0]["rewrite_mrr20_mean"], 0.7)
        self.assertEqual(train_steps[0]["rewrite_recall20_mean"], 0.8)
        self.assertEqual(train_steps[0]["rewrite_recall50_mean"], 0.9)
        self.assertEqual(train_steps[0]["main_reward_mean"], 0.6)
        self.assertEqual(train_steps[0]["flat_main_reward_group_ratio"], 0.15)
        self.assertAlmostEqual(group_steps[0]["trace_unsafe_copy_penalty_mean"], 0.5)
        self.assertAlmostEqual(group_steps[0]["trace_main_reward_mean"], 0.1)
        self.assertAlmostEqual(group_steps[0]["trace_flat_main_reward_group_ratio"], 0.0)
        self.assertAlmostEqual(group_steps[0]["trace_orig_mrr20_mean"], 0.15)


if __name__ == "__main__":
    unittest.main()
