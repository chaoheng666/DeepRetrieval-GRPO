import unittest
from types import SimpleNamespace

from app_config import RewardConfig, get_default_config
from data.loader import QueryExample
from train import evaluate_policy, resolve_eval_decode_settings


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
    def __init__(self):
        self.cfg = RewardConfig()

    def score(self, qid: str, rewritten_query: str, source_query: str | None = None):
        del qid, rewritten_query, source_query
        return SimpleNamespace(
            total=0.5,
            mrr=0.25,
            recall=0.1,
            copy_penalty=0.0,
            exact_copy_penalty=0.0,
            format_penalty=0.0,
        )


class _CaptureRewarder(_DummyRewarder):
    def __init__(self):
        super().__init__()
        self.seen_queries: list[str] = []

    def score(self, qid: str, rewritten_query: str, source_query: str | None = None):
        self.seen_queries.append(rewritten_query)
        return super().score(qid, rewritten_query, source_query)


class DefaultPromptConfigTests(unittest.TestCase):
    def test_default_prompt_uses_p23_demo_profile(self):
        config = get_default_config()

        self.assertEqual(config.prompt.prompt_id, "p24_diverse_lexical")
        self.assertEqual(config.prompt.max_new_tokens, 16)
        self.assertEqual(config.prompt.temperature, 0.0)
        self.assertEqual(config.prompt.top_p, 1.0)
        self.assertEqual(config.prompt.stop_on, "\n")
        self.assertTrue(config.prompt.enforce_single_line)
        self.assertIn("Strategy ID: P23 [FewShot]", config.prompt.system_prompt)
        self.assertIn("anemia symptoms women", config.prompt.template)

    def test_training_defaults_remain_stochastic(self):
        config = get_default_config()

        self.assertEqual(config.train.max_new_tokens, 18)
        self.assertEqual(config.train.temperature, 0.8)
        self.assertEqual(config.train.top_p, 0.95)

    def test_reward_defaults_penalize_copy_and_duplicates_more(self):
        config = get_default_config()

        self.assertEqual(config.reward.w_copy, 0.4)
        self.assertEqual(config.reward.exact_copy_penalty, 0.15)


class TrainEvaluationDecodeTests(unittest.TestCase):
    def test_resolve_eval_decode_settings_defaults_to_train_decode(self):
        config = get_default_config()
        config.train.max_new_tokens = 12
        config.train.temperature = 0.6
        config.train.top_p = 0.9

        settings = resolve_eval_decode_settings(
            config,
            SimpleNamespace(eval_max_new_tokens=None, eval_temperature=None, eval_top_p=None),
        )

        self.assertEqual(settings["max_new_tokens"], 12)
        self.assertEqual(settings["temperature"], 0.6)
        self.assertEqual(settings["top_p"], 0.9)

    def test_resolve_eval_decode_settings_honors_explicit_overrides(self):
        config = get_default_config()
        settings = resolve_eval_decode_settings(
            config,
            SimpleNamespace(eval_max_new_tokens=24, eval_temperature=0.2, eval_top_p=0.85),
        )

        self.assertEqual(settings["max_new_tokens"], 24)
        self.assertEqual(settings["temperature"], 0.2)
        self.assertEqual(settings["top_p"], 0.85)

    def test_evaluate_policy_uses_passed_decode_settings(self):
        model = _CaptureModel()
        rewarder = _DummyRewarder()
        config = get_default_config()
        queries = [QueryExample(qid="q1", text="what are symptoms of anemia in women")]

        metrics = evaluate_policy(
            model,
            rewarder,
            queries,
            guardrail_cfg=config.prompt,
            max_queries=None,
            max_new_tokens=16,
            temperature=0.0,
            top_p=1.0,
        )

        self.assertAlmostEqual(metrics["mrr_mean"], 0.25)
        self.assertAlmostEqual(metrics["exact_copy_penalty_mean"], 0.0)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["policy"], "actor")
        self.assertEqual(model.calls[0]["max_new_tokens"], 16)
        self.assertEqual(model.calls[0]["temperature"], 0.0)
        self.assertEqual(model.calls[0]["top_p"], 1.0)

    def test_evaluate_policy_scores_stabilized_final_query(self):
        class _PollutedModel(_CaptureModel):
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
                return (
                    "anemia symptoms women\n\n"
                    "User query: what are symptoms of anemia in women\n"
                    "Better BM25 query:"
                )

        model = _PollutedModel()
        rewarder = _CaptureRewarder()
        config = get_default_config()
        queries = [QueryExample(qid="q1", text="what are symptoms of anemia in women")]

        evaluate_policy(
            model,
            rewarder,
            queries,
            guardrail_cfg=config.prompt,
            max_queries=None,
            max_new_tokens=16,
            temperature=0.0,
            top_p=1.0,
        )

        self.assertEqual(rewarder.seen_queries, ["anemia symptoms women"])


if __name__ == "__main__":
    unittest.main()
