import unittest

from app_config import PromptConfig, RewardConfig
from data.loader import QueryExample
from eval_compare import build_per_qid_rows, evaluate_original, evaluate_with_model, select_sample_cases


class _DummyReward:
    def __init__(self, total: float, mrr: float, recall: float, recall_dense: float, **overrides):
        self.total = total
        self.mrr = mrr
        self.recall = recall
        self.recall_dense = recall_dense
        self.rank_bonus = overrides.get("rank_bonus", 0.0)
        self.orig_mrr = overrides.get("orig_mrr", 0.0)
        self.orig_recall = overrides.get("orig_recall", 0.0)
        self.orig_recall_aux = overrides.get("orig_recall_aux", 0.0)
        self.orig_rank_bonus = overrides.get("orig_rank_bonus", 0.0)
        self.delta_mrr = overrides.get("delta_mrr", 0.0)
        self.delta_recall = overrides.get("delta_recall", 0.0)
        self.delta_recall_aux = overrides.get("delta_recall_aux", 0.0)
        self.delta_rank_bonus = overrides.get("delta_rank_bonus", 0.0)
        self.main_reward = overrides.get("main_reward", 0.0)
        self.overedit_penalty = overrides.get("overedit_penalty", 0.0)
        self.bad_format_penalty = overrides.get("bad_format_penalty", 0.0)
        self.unsafe_copy_penalty = overrides.get("unsafe_copy_penalty", 0.0)


class _DummyRewarder:
    def __init__(self, mrr_k: int, recall_k: int, recall_dense_k: int = 100):
        self.mrr_k = mrr_k
        self.recall_k = recall_k
        self.recall_dense_k = recall_dense_k
        self.cfg = RewardConfig()

    def score(self, qid: str, rewritten_query: str, source_query: str | None = None):
        del qid, rewritten_query, source_query
        return _DummyReward(total=0.6, mrr=0.4, recall=0.25, recall_dense=0.5)


class _DummyModel:
    def generate_rewrite(
        self,
        query: str,
        *,
        policy: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        del policy, max_new_tokens, temperature, top_p
        return f"rewritten::{query}"


class _TrackingRewarder(_DummyRewarder):
    def __init__(self, mrr_k: int, recall_k: int, recall_dense_k: int = 100):
        super().__init__(mrr_k=mrr_k, recall_k=recall_k, recall_dense_k=recall_dense_k)
        self.seen_queries: list[str] = []

    def score(self, qid: str, rewritten_query: str, source_query: str | None = None):
        del qid, source_query
        self.seen_queries.append(rewritten_query)
        return _DummyReward(total=0.6, mrr=0.4, recall=0.25, recall_dense=0.5)


class _PollutedModel:
    def generate_rewrite(
        self,
        query: str,
        *,
        policy: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        del query, policy, max_new_tokens, temperature, top_p
        return "anemia symptoms women\n\nUser query: what are symptoms of anemia in women\nBetter BM25 query:"


class EvalCompareMetricTests(unittest.TestCase):
    def test_evaluate_original_exposes_generic_and_topk_metric_keys(self):
        rewarder = _DummyRewarder(mrr_k=20, recall_k=50)
        queries = [QueryExample(qid="q1", text="query one")]

        metrics, _ = evaluate_original(queries, rewarder, progress_every=1)

        self.assertAlmostEqual(metrics["mrr"], 0.4)
        self.assertAlmostEqual(metrics["mrr@20"], 0.4)
        self.assertAlmostEqual(metrics["recall"], 0.25)
        self.assertAlmostEqual(metrics["recall@50"], 0.25)
        self.assertAlmostEqual(metrics["recall_dense"], 0.5)
        self.assertAlmostEqual(metrics["recall@100"], 0.5)
        self.assertAlmostEqual(metrics["reward_mean"], 0.6)

    def test_evaluate_with_model_exposes_generic_and_topk_metric_keys(self):
        rewarder = _DummyRewarder(mrr_k=10, recall_k=50)
        model = _DummyModel()
        queries = [QueryExample(qid="q1", text="query one")]

        metrics, _ = evaluate_with_model(
            model,
            queries,
            rewarder,
            max_new_tokens=8,
            stage_name="zero-shot",
            progress_every=1,
        )

        self.assertAlmostEqual(metrics["mrr"], 0.4)
        self.assertAlmostEqual(metrics["mrr@10"], 0.4)
        self.assertAlmostEqual(metrics["recall"], 0.25)
        self.assertAlmostEqual(metrics["recall@50"], 0.25)
        self.assertAlmostEqual(metrics["recall_dense"], 0.5)
        self.assertAlmostEqual(metrics["recall@100"], 0.5)
        self.assertAlmostEqual(metrics["reward_mean"], 0.6)

    def test_evaluate_with_model_scores_stabilized_final_query_when_guardrail_is_enabled(self):
        rewarder = _TrackingRewarder(mrr_k=10, recall_k=50)
        model = _PollutedModel()
        queries = [QueryExample(qid="q1", text="what are symptoms of anemia in women")]

        _, per_qid = evaluate_with_model(
            model,
            queries,
            rewarder,
            guardrail_cfg=PromptConfig(min_terms=3, max_terms=11, fallback_mode="balanced"),
            max_new_tokens=8,
            stage_name="rl",
            progress_every=1,
        )

        self.assertEqual(rewarder.seen_queries, ["anemia symptoms women"])
        self.assertEqual(per_qid["q1"][0], "anemia symptoms women")

    def test_build_per_qid_rows_exposes_full_metric_payload_and_sample_preview(self):
        queries = [
            QueryExample(qid="q1", text="query one"),
            QueryExample(qid="q2", text="query two"),
        ]
        original_by_qid = {
            "q1": _DummyReward(
                total=0.1,
                mrr=0.2,
                recall=0.3,
                recall_dense=0.4,
                rank_bonus=0.5,
                orig_mrr=0.2,
                orig_recall=0.3,
                orig_recall_aux=0.4,
                orig_rank_bonus=0.5,
                main_reward=0.6,
            ),
            "q2": _DummyReward(total=0.0, mrr=0.0, recall=0.0, recall_dense=0.0),
        }
        zero_by_qid = {
            "q1": (
                "zero rewrite one",
                _DummyReward(
                    total=0.7,
                    mrr=0.8,
                    recall=0.9,
                    recall_dense=1.0,
                    rank_bonus=0.4,
                    orig_mrr=0.2,
                    orig_recall=0.3,
                    orig_recall_aux=0.4,
                    orig_rank_bonus=0.1,
                    delta_mrr=0.6,
                    delta_recall=0.6,
                    delta_recall_aux=0.6,
                    delta_rank_bonus=0.3,
                    main_reward=0.55,
                    overedit_penalty=0.05,
                    bad_format_penalty=0.02,
                    unsafe_copy_penalty=0.01,
                ),
            ),
            "q2": ("zero rewrite two", _DummyReward(total=0.0, mrr=0.0, recall=0.0, recall_dense=0.0)),
        }
        rl_by_qid = {
            "q1": ("rl rewrite one", _DummyReward(total=0.9, mrr=1.0, recall=1.0, recall_dense=1.0)),
            "q2": ("rl rewrite two", _DummyReward(total=0.1, mrr=0.2, recall=0.3, recall_dense=0.4)),
        }

        per_qid = build_per_qid_rows(queries, original_by_qid, zero_by_qid, rl_by_qid)
        samples = select_sample_cases(per_qid, 1)

        self.assertEqual([row["qid"] for row in per_qid], ["q1", "q2"])
        self.assertEqual(samples, per_qid[:1])
        self.assertEqual(per_qid[0]["zero_rewrite"], "zero rewrite one")
        self.assertEqual(per_qid[0]["rl_rewrite"], "rl rewrite one")
        self.assertEqual(per_qid[0]["original"]["mrr"], 0.2)
        self.assertEqual(per_qid[0]["original"]["recall"], 0.3)
        self.assertEqual(per_qid[0]["original"]["recall_dense"], 0.4)
        self.assertEqual(per_qid[0]["zero_shot"]["rank_bonus"], 0.4)
        self.assertEqual(per_qid[0]["zero_shot"]["orig_mrr"], 0.2)
        self.assertEqual(per_qid[0]["zero_shot"]["orig_recall"], 0.3)
        self.assertEqual(per_qid[0]["zero_shot"]["orig_recall_aux"], 0.4)
        self.assertEqual(per_qid[0]["zero_shot"]["orig_rank_bonus"], 0.1)
        self.assertEqual(per_qid[0]["zero_shot"]["delta_mrr"], 0.6)
        self.assertEqual(per_qid[0]["zero_shot"]["delta_recall"], 0.6)
        self.assertEqual(per_qid[0]["zero_shot"]["delta_recall_aux"], 0.6)
        self.assertEqual(per_qid[0]["zero_shot"]["delta_rank_bonus"], 0.3)
        self.assertEqual(per_qid[0]["zero_shot"]["main_reward"], 0.55)
        self.assertEqual(per_qid[0]["zero_shot"]["overedit_penalty"], 0.05)
        self.assertEqual(per_qid[0]["zero_shot"]["bad_format_penalty"], 0.02)
        self.assertEqual(per_qid[0]["zero_shot"]["unsafe_copy_penalty"], 0.01)


if __name__ == "__main__":
    unittest.main()
