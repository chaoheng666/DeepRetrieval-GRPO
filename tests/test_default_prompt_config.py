import unittest
from types import SimpleNamespace

from app_config import RewardConfig, apply_reward_mode_prompt_defaults, get_default_config
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


class _ModeTrackingActor:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = bool(mode)
        return self


class _BatchCaptureModel:
    def __init__(self):
        self.actor_model = _ModeTrackingActor()
        self.batch_calls: list[dict[str, object]] = []
        self.training_flags: list[bool] = []

    def generate_rewrite_batch(
        self,
        queries: list[str],
        *,
        policy: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        self.training_flags.append(self.actor_model.training)
        self.batch_calls.append(
            {
                "queries": list(queries),
                "policy": policy,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        return [f"batch::{query}" for query in queries]


class _DummyRewarder:
    def __init__(self):
        self.cfg = RewardConfig()

    def score(self, qid: str, rewritten_query: str, source_query: str | None = None):
        del qid, rewritten_query, source_query
        return SimpleNamespace(
            total=0.5,
            mrr=0.25,
            recall=0.1,
            recall_dense=0.2,
            term_preserve=1.0,
            length_score=1.0,
            clean_format=1.0,
            bad_format_penalty=0.0,
            unsafe_copy_penalty=0.0,
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

        self.assertEqual(config.prompt.prompt_id, "p24_diverse_lexical_top20")
        self.assertEqual(config.prompt.max_new_tokens, 10)
        self.assertEqual(config.prompt.temperature, 0.0)
        self.assertEqual(config.prompt.top_p, 1.0)
        self.assertEqual(config.prompt.stop_on, "\n")
        self.assertTrue(config.prompt.enforce_single_line)
        self.assertIn("Strategy ID: P23 [FewShot]", config.prompt.system_prompt)
        self.assertIn("MRR@20", config.prompt.system_prompt)
        self.assertIn("Recall@50", config.prompt.system_prompt)
        self.assertIn("anemia symptoms women", config.prompt.template)

    def test_training_defaults_remain_stochastic(self):
        config = get_default_config()

        self.assertEqual(config.train.batch_size, 24)
        self.assertEqual(config.train.group_size, 8)
        self.assertEqual(config.train.max_group_size, 12)
        self.assertEqual(config.train.actor_chunk_size, 2)
        self.assertEqual(config.train.kl_beta, 0.04)
        self.assertEqual(config.train.max_new_tokens, 10)
        self.assertEqual(config.train.temperature, 0.82)
        self.assertEqual(config.train.top_p, 0.93)
        self.assertTrue(config.train.curriculum_enable)
        self.assertEqual(config.model.projection_chunk_size, 64)

    def test_reward_defaults_use_top20_delta_formula(self):
        config = get_default_config()

        self.assertEqual(config.reward.mrr_k, 20)
        self.assertEqual(config.reward.recall_k, 20)
        self.assertEqual(config.reward.recall_dense_k, 50)
        self.assertEqual(config.reward.reward_mode, "top20_delta")
        self.assertEqual(config.reward.w_mrr, 0.52)
        self.assertEqual(config.reward.w_recall, 0.22)
        self.assertEqual(config.reward.w_recall_dense, 0.16)
        self.assertEqual(config.reward.w_rank_bonus, 0.10)
        self.assertEqual(config.reward.w_bad_format, 0.18)
        self.assertEqual(config.reward.w_unsafe_copy, 0.14)
        self.assertEqual(config.reward.w_overedit, 0.08)
        self.assertEqual(config.reward.overedit_tau, 0.45)
        self.assertEqual(config.reward.recall_drop_lambda, 0.80)
        self.assertEqual(config.reward.anchor_bonus_value, 0.05)

    def test_top20_reward_mode_updates_stock_prompt_wording(self):
        config = get_default_config()
        config.prompt.prompt_id = "p24_diverse_lexical"
        config.reward.reward_mode = "top20_delta"

        updated = apply_reward_mode_prompt_defaults(config)

        self.assertEqual(updated.prompt.prompt_id, "p24_diverse_lexical_top20")
        self.assertIn("MRR@20", updated.prompt.system_prompt)
        self.assertIn("Recall@20", updated.prompt.system_prompt)
        self.assertIn("Recall@50", updated.prompt.system_prompt)


class TrainEvaluationDecodeTests(unittest.TestCase):
    def test_resolve_eval_decode_settings_defaults_to_train_decode(self):
        config = get_default_config()
        config.train.max_new_tokens = 12
        config.train.temperature = 0.6
        config.train.top_p = 0.9

        settings = resolve_eval_decode_settings(
            config,
            SimpleNamespace(
                eval_max_new_tokens=None,
                eval_temperature=None,
                eval_top_p=None,
                eval_query_batch_size=None,
            ),
        )

        self.assertEqual(settings["max_new_tokens"], 12)
        self.assertEqual(settings["temperature"], 0.6)
        self.assertEqual(settings["top_p"], 0.9)
        self.assertEqual(settings["query_batch_size"], config.train.batch_size)

    def test_resolve_eval_decode_settings_honors_explicit_overrides(self):
        config = get_default_config()
        settings = resolve_eval_decode_settings(
            config,
            SimpleNamespace(
                eval_max_new_tokens=24,
                eval_temperature=0.2,
                eval_top_p=0.85,
                eval_query_batch_size=6,
            ),
        )

        self.assertEqual(settings["max_new_tokens"], 24)
        self.assertEqual(settings["temperature"], 0.2)
        self.assertEqual(settings["top_p"], 0.85)
        self.assertEqual(settings["query_batch_size"], 6)

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
        self.assertAlmostEqual(metrics["recall_dense_mean"], 0.2)
        self.assertAlmostEqual(metrics["unsafe_copy_penalty_mean"], 0.0)
        self.assertAlmostEqual(metrics["anchor_bonus_mean"], 0.0)
        self.assertAlmostEqual(metrics["recall_drop_penalty_mean"], 0.0)
        self.assertIn("orig_mrr20_mean", metrics)
        self.assertIn("rewrite_mrr20_mean", metrics)
        self.assertIn("delta_mrr20_positive_ratio", metrics)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["policy"], "actor")
        self.assertEqual(model.calls[0]["max_new_tokens"], 16)
        self.assertEqual(model.calls[0]["temperature"], 0.0)
        self.assertEqual(model.calls[0]["top_p"], 1.0)

    def test_evaluate_policy_uses_batch_generation_when_available(self):
        model = _BatchCaptureModel()
        rewarder = _DummyRewarder()
        config = get_default_config()
        queries = [
            QueryExample(qid="q1", text="what are symptoms of anemia in women"),
            QueryExample(qid="q2", text="bm25 query rewrite methods"),
        ]

        metrics = evaluate_policy(
            model,
            rewarder,
            queries,
            guardrail_cfg=config.prompt,
            max_queries=None,
            max_new_tokens=12,
            temperature=0.3,
            top_p=0.85,
            query_batch_size=2,
        )

        self.assertAlmostEqual(metrics["reward_mean"], 0.5)
        self.assertEqual(len(model.batch_calls), 1)
        self.assertEqual(model.batch_calls[0]["queries"], [query.text for query in queries])
        self.assertEqual(model.batch_calls[0]["policy"], "actor")
        self.assertEqual(model.batch_calls[0]["max_new_tokens"], 12)
        self.assertEqual(model.batch_calls[0]["temperature"], 0.3)
        self.assertEqual(model.batch_calls[0]["top_p"], 0.85)
        self.assertEqual(model.training_flags, [False])
        self.assertTrue(model.actor_model.training)

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
