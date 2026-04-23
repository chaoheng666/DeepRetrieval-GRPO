import unittest

from train import (
    ensure_num_epochs_covers_max_steps,
    resolve_phase_metric_key,
    resolve_phase_metric_value,
    update_best_so_far_early_stop,
)


class EarlyStopControlTests(unittest.TestCase):
    def test_fluctuation_within_threshold_does_not_stop(self):
        state = update_best_so_far_early_stop(
            current_metric=0.493,
            best_metric_so_far=0.50,
            degrade_streak=0,
            eval_count=4,
            drop_threshold=0.01,
            degrade_patience=2,
            warmup_evals=3,
        )

        self.assertAlmostEqual(state["drop_from_best"], 0.007)
        self.assertEqual(state["degrade_streak"], 0)
        self.assertFalse(state["should_stop"])
        self.assertEqual(state["stop_reason"], "within_tolerance")

    def test_two_consecutive_significant_drops_trigger_stop(self):
        first = update_best_so_far_early_stop(
            current_metric=0.488,
            best_metric_so_far=0.50,
            degrade_streak=0,
            eval_count=4,
            drop_threshold=0.01,
            degrade_patience=2,
            warmup_evals=3,
        )
        second = update_best_so_far_early_stop(
            current_metric=0.487,
            best_metric_so_far=float(first["best_metric_so_far"]),
            degrade_streak=int(first["degrade_streak"]),
            eval_count=5,
            drop_threshold=0.01,
            degrade_patience=2,
            warmup_evals=3,
        )

        self.assertEqual(first["degrade_streak"], 1)
        self.assertFalse(first["should_stop"])
        self.assertEqual(first["stop_reason"], "significant_drop")
        self.assertEqual(second["degrade_streak"], 2)
        self.assertTrue(second["should_stop"])
        self.assertEqual(second["stop_reason"], "significant_drop_stop")

    def test_warmup_evals_do_not_trigger_degrade_streak(self):
        state = update_best_so_far_early_stop(
            current_metric=0.470,
            best_metric_so_far=0.50,
            degrade_streak=1,
            eval_count=3,
            drop_threshold=0.01,
            degrade_patience=2,
            warmup_evals=3,
        )

        self.assertEqual(state["degrade_streak"], 0)
        self.assertFalse(state["should_stop"])
        self.assertEqual(state["stop_reason"], "warmup_baseline")


class PhaseMetricSelectionTests(unittest.TestCase):
    def test_phase1_uses_recall_metric(self):
        eval_metrics = {
            "rewrite_recall20_mean": 0.61,
            "rewrite_mrr20_mean": 0.18,
            "recall_mean": 0.61,
            "mrr_mean": 0.18,
        }

        metric_key = resolve_phase_metric_key("phase1")

        self.assertEqual(metric_key, "rewrite_recall20_mean")
        self.assertAlmostEqual(resolve_phase_metric_value(eval_metrics, metric_key), 0.61)

    def test_phase2_uses_mrr_metric(self):
        eval_metrics = {
            "rewrite_recall20_mean": 0.58,
            "rewrite_mrr20_mean": 0.21,
            "recall_mean": 0.58,
            "mrr_mean": 0.21,
        }

        metric_key = resolve_phase_metric_key("phase2")

        self.assertEqual(metric_key, "rewrite_mrr20_mean")
        self.assertAlmostEqual(resolve_phase_metric_value(eval_metrics, metric_key), 0.21)


class NumEpochCoverageTests(unittest.TestCase):
    def test_extends_num_epochs_to_cover_max_steps(self):
        resolved = ensure_num_epochs_covers_max_steps(
            num_epochs=1,
            num_train_queries=100,
            batch_size=24,
            max_steps=20,
        )

        self.assertEqual(resolved, 4)

    def test_keeps_existing_num_epochs_when_already_enough(self):
        resolved = ensure_num_epochs_covers_max_steps(
            num_epochs=5,
            num_train_queries=100,
            batch_size=24,
            max_steps=20,
        )

        self.assertEqual(resolved, 5)


if __name__ == "__main__":
    unittest.main()
