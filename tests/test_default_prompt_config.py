import unittest
from types import SimpleNamespace

from app_config import get_default_config
from data.loader import QueryExample
from train import evaluate_policy


class _CaptureModel:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def generate_rewrite(
        self,
        query: str,
        *,
        policy: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        self.calls.append(
            {
                "query": query,
                "policy": policy,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        return f"rewritten::{query}"


class _DummyRewarder:
    def score(self, qid: str, rewritten_query: str, source_query: str | None = None):
        del qid, rewritten_query, source_query
        return SimpleNamespace(
            total=0.5,
            mrr=0.25,
            recall=0.1,
            copy_penalty=0.0,
            format_penalty=0.0,
        )


class DefaultPromptConfigTests(unittest.TestCase):
    def test_default_prompt_uses_p23_demo_profile(self):
        config = get_default_config()

        self.assertEqual(config.prompt.prompt_id, "p23_demo")
        self.assertEqual(config.prompt.max_new_tokens, 16)
        self.assertEqual(config.prompt.temperature, 0.0)
        self.assertEqual(config.prompt.top_p, 1.0)
        self.assertEqual(config.prompt.stop_on, "\n")
        self.assertTrue(config.prompt.enforce_single_line)
        self.assertIn("Strategy ID: P23 [FewShot]", config.prompt.system_prompt)
        self.assertIn("anemia symptoms women", config.prompt.template)

    def test_training_defaults_remain_stochastic(self):
        config = get_default_config()

        self.assertEqual(config.train.max_new_tokens, 24)
        self.assertEqual(config.train.temperature, 1.0)
        self.assertEqual(config.train.top_p, 0.95)


class TrainEvaluationDecodeTests(unittest.TestCase):
    def test_evaluate_policy_uses_passed_decode_settings(self):
        model = _CaptureModel()
        rewarder = _DummyRewarder()
        queries = [QueryExample(qid="q1", text="what are symptoms of anemia in women")]

        metrics = evaluate_policy(
            model,
            rewarder,
            queries,
            max_queries=None,
            max_new_tokens=16,
            temperature=0.0,
            top_p=1.0,
        )

        self.assertAlmostEqual(metrics["mrr_mean"], 0.25)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["policy"], "actor")
        self.assertEqual(model.calls[0]["max_new_tokens"], 16)
        self.assertEqual(model.calls[0]["temperature"], 0.0)
        self.assertEqual(model.calls[0]["top_p"], 1.0)


if __name__ == "__main__":
    unittest.main()
