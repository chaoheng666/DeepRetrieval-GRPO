import unittest

from app_config import PromptConfig, RewardConfig
from data.loader import QueryExample
from eval_compare import evaluate_original, evaluate_with_model


class _DummyReward:
    def __init__(self, total: float, mrr: float, recall: float, recall_dense: float):
        self.total = total
        self.mrr = mrr
        self.recall = recall
        self.recall_dense = recall_dense


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


if __name__ == "__main__":
    unittest.main()
