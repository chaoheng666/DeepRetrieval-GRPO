import argparse
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from app_config import ModelConfig, PromptConfig, RewardConfig, get_default_config
from core.grpo_engine import GRPOEngine
from core.model_wrapper import GeneratedSample, ModelWrapper
from core.reward_func import RewardBreakdown
from data.loader import QueryExample
from train import apply_low_mem_mode, apply_runtime_mode_adjustments


class RuntimeConfigTests(unittest.TestCase):
    def test_disable4bit_lowmem_no_auto_downscale(self):
        cfg = apply_low_mem_mode(get_default_config())
        original_group = cfg.train.group_size
        original_tokens = cfg.train.max_new_tokens
        original_ref_map = cfg.model.ref_device_map

        args = argparse.Namespace(low_mem_mode=True, disable_4bit=True)
        with patch("torch.cuda.is_available", return_value=True):
            adjusted = apply_runtime_mode_adjustments(cfg, args)

        self.assertEqual(adjusted.train.group_size, original_group)
        self.assertEqual(adjusted.train.max_new_tokens, original_tokens)
        self.assertEqual(adjusted.model.ref_device_map, original_ref_map)

    def test_lowmem_cpu_runtime_forces_cpu_loading(self):
        cfg = apply_low_mem_mode(get_default_config())
        args = argparse.Namespace(low_mem_mode=True, disable_4bit=False)
        with patch("torch.cuda.is_available", return_value=False):
            adjusted = apply_runtime_mode_adjustments(cfg, args)

        self.assertFalse(adjusted.model.load_in_4bit)
        self.assertEqual(adjusted.model.actor_device_map, "cpu")
        self.assertEqual(adjusted.model.ref_device_map, "cpu")

    def test_runtime_adjustments_clamp_invalid_numeric_values(self):
        cfg = get_default_config()
        cfg.train.num_epochs = 0
        cfg.train.batch_size = 0
        cfg.train.eval_every_steps = 0
        cfg.train.max_new_tokens = 0
        cfg.train.group_size = 1
        cfg.train.max_group_size = 1
        cfg.train.reward_gap_threshold = -0.5
        cfg.train.gap_sampling_temperature_delta = -0.25
        cfg.train.max_steps = 0
        cfg.reward.mrr_k = 0
        cfg.reward.recall_k = 0
        cfg.data.max_train_queries = -1
        cfg.data.max_val_queries = -2
        args = argparse.Namespace(low_mem_mode=False, disable_4bit=False)

        with patch("torch.cuda.is_available", return_value=True):
            adjusted = apply_runtime_mode_adjustments(cfg, args)

        self.assertEqual(adjusted.train.num_epochs, 1)
        self.assertEqual(adjusted.train.batch_size, 1)
        self.assertEqual(adjusted.train.eval_every_steps, 1)
        self.assertEqual(adjusted.train.max_new_tokens, 1)
        self.assertEqual(adjusted.train.group_size, 2)
        self.assertEqual(adjusted.train.max_group_size, 2)
        self.assertEqual(adjusted.train.reward_gap_threshold, 0.0)
        self.assertEqual(adjusted.train.gap_sampling_temperature_delta, 0.0)
        self.assertEqual(adjusted.train.max_steps, 1)
        self.assertEqual(adjusted.reward.mrr_k, 1)
        self.assertEqual(adjusted.reward.recall_k, 1)
        self.assertEqual(adjusted.data.max_train_queries, 0)
        self.assertEqual(adjusted.data.max_val_queries, 0)

    def test_runtime_adjustments_reject_invalid_train_ratio(self):
        cfg = get_default_config()
        cfg.data.train_ratio = 1.0
        args = argparse.Namespace(low_mem_mode=False, disable_4bit=False)

        with patch("torch.cuda.is_available", return_value=True):
            with self.assertRaises(ValueError):
                apply_runtime_mode_adjustments(cfg, args)

    def test_runtime_adjustments_force_4bit_ref_on_24g_auto_mode(self):
        cfg = get_default_config()
        cfg.model.load_in_4bit = True
        cfg.model.ref_precision_mode = "auto"
        args = argparse.Namespace(low_mem_mode=False, disable_4bit=False)

        fake_props = SimpleNamespace(total_memory=24 * 1024**3)
        with patch.dict("os.environ", {}, clear=True), patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_properties",
            return_value=fake_props,
        ):
            adjusted = apply_runtime_mode_adjustments(cfg, args)

        self.assertEqual(adjusted.model.ref_precision_mode, "4bit")

    def test_runtime_adjustments_preserve_explicit_full_ref_mode(self):
        cfg = get_default_config()
        cfg.model.load_in_4bit = True
        cfg.model.ref_precision_mode = "full"
        args = argparse.Namespace(low_mem_mode=False, disable_4bit=False)

        fake_props = SimpleNamespace(total_memory=24 * 1024**3)
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_properties",
            return_value=fake_props,
        ):
            adjusted = apply_runtime_mode_adjustments(cfg, args)

        self.assertEqual(adjusted.model.ref_precision_mode, "full")


class _ToyModelWrapper:
    def __init__(self):
        self.actor_model = torch.nn.Linear(1, 1, bias=False)
        self.prompt_cfg = PromptConfig()
        self._responses = iter(["normal query", "乱码¤¤"])
        self.training_flags: list[bool] = []

    def build_prompt(self, query: str) -> str:
        return f"prompt::{query}"

    def generate_with_logprob(self, prompt: str, *, max_new_tokens: int, temperature: float, top_p: float) -> GeneratedSample:
        self.training_flags.append(self.actor_model.training)
        text = next(self._responses)
        return GeneratedSample(
            response_text=text,
            response_token_ids=[1, 2],
            logprob_old=torch.tensor([0.0, 0.0], dtype=torch.float32),
        )

    def compute_logprob(
        self,
        prompt: str,
        response_token_ids: list[int],
        *,
        policy: str = "actor",
        no_grad: bool = False,
    ) -> torch.Tensor:
        self.training_flags.append(self.actor_model.training)
        n = len(response_token_ids)
        if policy == "actor":
            base = self.actor_model.weight.view(-1)[0]
            return (base + 0.1).expand(n)
        return torch.zeros(n, dtype=torch.float32)

    def trainable_parameters(self):
        return [p for p in self.actor_model.parameters() if p.requires_grad]


class _ToyRewarder:
    def __init__(self):
        self.cfg = RewardConfig()

    def score(self, qid: str, rewritten_query: str, source_query: str | None = None) -> RewardBreakdown:
        bad_format_penalty = 1.0 if "¤" in rewritten_query else 0.0
        return RewardBreakdown(
            total=1.0 - (0.1 * bad_format_penalty),
            mrr=0.5,
            recall=0.25,
            overlap=0.4,
            bad_format_penalty=bad_format_penalty,
            clean_format=0.0 if bad_format_penalty else 1.0,
            hit_rank=1,
            retrieved_relevant_count=1,
            relevant_total=4,
            rewritten_query=f"clean::{rewritten_query}",
        )


class _RecordingRewarder:
    def __init__(self):
        self.seen_queries: list[str] = []
        self.cfg = RewardConfig()

    def score(self, qid: str, rewritten_query: str, source_query: str | None = None) -> RewardBreakdown:
        self.seen_queries.append(rewritten_query)
        return RewardBreakdown(
            total=1.0,
            mrr=0.5,
            recall=0.25,
            overlap=0.4,
            hit_rank=1,
            retrieved_relevant_count=1,
            relevant_total=4,
            rewritten_query=rewritten_query,
        )


class EngineTraceTests(unittest.TestCase):
    def test_group_trace_metrics(self):
        wrapper = _ToyModelWrapper()
        rewarder = _ToyRewarder()
        optimizer = torch.optim.SGD(wrapper.trainable_parameters(), lr=1e-2)
        engine = GRPOEngine(
            model_wrapper=wrapper,
            rewarder=rewarder,
            optimizer=optimizer,
            group_size=2,
            clip_range=0.2,
            kl_beta=0.01,
            grad_clip_norm=1.0,
            max_new_tokens=8,
            temperature=0.8,
            top_p=0.95,
            reward_gap_threshold=0.0,
        )

        metrics = engine.train_step([QueryExample(qid="q1", text="input query")], collect_best_queries=True)
        summaries = metrics["group_query_summaries"]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["qid"], "q1")
        self.assertEqual(summaries[0]["input_query"], "input query")
        self.assertEqual(len(summaries[0]["group_raw_responses"]), 2)
        self.assertEqual(len(summaries[0]["group_rollout_responses"]), 2)
        self.assertEqual(len(summaries[0]["group_cleaned_queries"]), 2)
        self.assertEqual(len(summaries[0]["group_final_queries"]), 2)
        self.assertEqual(len(summaries[0]["group_fallback_to_original"]), 2)
        self.assertEqual(len(summaries[0]["group_fallback_reasons"]), 2)
        self.assertEqual(len(summaries[0]["group_rewards"]), 2)
        self.assertEqual(len(summaries[0]["group_recall"]), 2)
        self.assertEqual(len(summaries[0]["group_term_preserve"]), 2)
        self.assertEqual(len(summaries[0]["group_bad_format_penalties"]), 2)
        self.assertGreaterEqual(metrics["bad_format_penalty_mean"], 0.0)
        self.assertIn("unique_final_query_mean", metrics)
        self.assertIn("collapsed_group_ratio", metrics)
        self.assertIn("all_same_final_query_ratio", metrics)
        self.assertIn("flat_reward_group_ratio", metrics)
        self.assertIn("loss_pg_abs_mean", metrics)
        self.assertIn("kl_dominance_ratio", metrics)
        self.assertGreaterEqual(metrics["kl_dominance_ratio"], 0.0)
        self.assertLessEqual(metrics["kl_dominance_ratio"], 1.0)
        self.assertTrue(all(flag is False for flag in wrapper.training_flags))
        self.assertTrue(wrapper.actor_model.training)

    def test_train_step_uses_batched_logprob_when_available(self):
        wrapper = _ToyBatchModelWrapper()
        rewarder = _ToyRewarder()
        optimizer = torch.optim.SGD(wrapper.trainable_parameters(), lr=1e-2)
        engine = GRPOEngine(
            model_wrapper=wrapper,
            rewarder=rewarder,
            optimizer=optimizer,
            group_size=2,
            clip_range=0.2,
            kl_beta=0.01,
            grad_clip_norm=1.0,
            max_new_tokens=8,
            temperature=0.8,
            top_p=0.95,
            reward_gap_threshold=0.0,
        )

        metrics = engine.train_step([QueryExample(qid="q1", text="input query")], collect_best_queries=False)

        self.assertEqual(wrapper.single_compute_calls, 0)
        self.assertEqual(len(wrapper.batch_compute_calls), 2)
        self.assertEqual(wrapper.batch_compute_calls[0]["policy"], "actor")
        self.assertFalse(wrapper.batch_compute_calls[0]["no_grad"])
        self.assertEqual(wrapper.batch_compute_calls[1]["policy"], "ref")
        self.assertTrue(wrapper.batch_compute_calls[1]["no_grad"])
        self.assertGreaterEqual(metrics["valid_samples"], 1.0)

    def test_train_step_postprocesses_rollout_queries_before_reward(self):
        wrapper = _ToyModelWrapper()
        wrapper.prompt_cfg = SimpleNamespace(stop_on="\n", enforce_single_line=True)
        wrapper._responses = iter(
            [
                "finderscope\n\nUser query: what is a finderscope\nBetter BM25 query:",
                "mastoidectomy\n\nUser query: what is the capital of the united states\nBetter BM25",
            ]
        )
        rewarder = _RecordingRewarder()
        optimizer = torch.optim.SGD(wrapper.trainable_parameters(), lr=1e-2)
        engine = GRPOEngine(
            model_wrapper=wrapper,
            rewarder=rewarder,
            optimizer=optimizer,
            group_size=2,
            clip_range=0.2,
            kl_beta=0.01,
            grad_clip_norm=1.0,
            max_new_tokens=8,
            temperature=0.8,
            top_p=0.95,
            reward_gap_threshold=0.0,
        )

        engine.train_step([QueryExample(qid="q1", text="input query")], collect_best_queries=False)

        self.assertEqual(rewarder.seen_queries, ["finderscope", "mastoidectomy"])


class _CharTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    class _TokenizedBatch(dict):
        def __getattr__(self, item):
            return self[item]

    def __call__(self, text: str, return_tensors: str = "pt"):
        if return_tensors != "pt":
            raise ValueError("Only return_tensors='pt' is supported in tests.")
        ids = self.encode(text)
        return self._TokenizedBatch({"input_ids": torch.tensor([ids], dtype=torch.long)})

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(i) for i in ids)

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return [ord(ch) for ch in text]


class ModelWrapperCleanupTests(unittest.TestCase):
    def test_resolve_local_model_source_uses_hf_cache_snapshot_from_refs_main(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "models--Qwen--Qwen3.5-0.8B"
            snapshot_name = "abc123snapshot"
            snapshot_dir = cache_root / "snapshots" / snapshot_name
            snapshot_dir.mkdir(parents=True)
            (cache_root / "refs").mkdir(parents=True)
            (cache_root / "refs" / "main").write_text(snapshot_name, encoding="utf-8")

            model_source, local_only = ModelWrapper._resolve_local_model_source(str(cache_root))

        self.assertTrue(local_only)
        self.assertEqual(model_source, str(snapshot_dir))

    def test_finalize_generated_sample_truncates_template_continuation(self):
        wrapper = ModelWrapper.__new__(ModelWrapper)
        wrapper.prompt_cfg = PromptConfig(
            stop_on="\n",
            stop_strings=("\n", "\nUser query:", "\nBetter BM25 query:"),
            enforce_single_line=True,
        )
        wrapper.tokenizer = _CharTokenizer()

        polluted = "finderscope\n\nUser query: what is a finderscope\nBetter BM25 query:"
        response_ids = [ord(ch) for ch in polluted]
        sample = wrapper._finalize_generated_sample(
            response_ids,
            scores=None,
            sequence_index=0,
            with_logprob=False,
        )

        self.assertEqual(sample.raw_response_text, polluted)
        self.assertEqual(sample.response_text, "finderscope")
        self.assertEqual(sample.response_token_ids, [ord(ch) for ch in "finderscope"])


class _ToyLogprobModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, use_cache: bool = False):
        del attention_mask, use_cache
        batch_size, seq_len = input_ids.shape
        logits = torch.full(
            (batch_size, seq_len, self.vocab_size),
            -4.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        token_ids = input_ids.clamp_min(0).clamp_max(self.vocab_size - 1)
        logits.scatter_(
            2,
            token_ids.unsqueeze(-1),
            (2.0 + self.bias).expand(batch_size, seq_len, 1),
        )
        return SimpleNamespace(logits=logits)


class ModelWrapperBatchLogprobTests(unittest.TestCase):
    def test_compute_logprob_batch_matches_single_sample_path(self):
        wrapper = ModelWrapper.__new__(ModelWrapper)
        wrapper.tokenizer = _CharTokenizer()
        wrapper.actor_model = _ToyLogprobModel()
        wrapper.ref_model = _ToyLogprobModel()

        prompts = ["ab", "query", ""]
        responses = [
            [ord("c"), ord("d")],
            [ord("x")],
            [],
        ]

        expected = [
            wrapper.compute_logprob(prompt, response, policy="actor", no_grad=True)
            for prompt, response in zip(prompts, responses)
        ]
        actual = wrapper.compute_logprob_batch(
            prompts,
            responses,
            policy="actor",
            no_grad=True,
        )

        self.assertEqual(len(actual), len(expected))
        for lhs, rhs in zip(expected, actual):
            self.assertTrue(torch.allclose(lhs.cpu(), rhs.cpu()))


class _DuplicateToyModelWrapper(_ToyModelWrapper):
    def __init__(self):
        super().__init__()
        self.prompt_cfg = PromptConfig(stop_on="\n", enforce_single_line=True, min_terms=2, max_terms=8)
        self._duplicate_old_logprobs = iter([0.0, -0.2, -0.4])
        self._responses = iter(
            [
                "same query\n\nUser query: ignored\nBetter BM25 query:",
                "same query\n\nUser query: ignored\nBetter BM25 query:",
                "same query\n\nUser query: ignored\nBetter BM25 query:",
            ]
        )

    def generate_with_logprob(self, prompt: str, *, max_new_tokens: int, temperature: float, top_p: float) -> GeneratedSample:
        del prompt, max_new_tokens, temperature, top_p
        self.training_flags.append(self.actor_model.training)
        text = next(self._responses)
        old_logprob = next(self._duplicate_old_logprobs)
        return GeneratedSample(
            response_text=text,
            response_token_ids=[1, 2],
            logprob_old=torch.tensor([old_logprob, old_logprob], dtype=torch.float32),
        )


class _ToyBatchModelWrapper(_ToyModelWrapper):
    def __init__(self):
        super().__init__()
        self.single_compute_calls = 0
        self.batch_compute_calls: list[dict[str, object]] = []

    def compute_logprob(
        self,
        prompt: str,
        response_token_ids: list[int],
        *,
        policy: str = "actor",
        no_grad: bool = False,
    ) -> torch.Tensor:
        self.single_compute_calls += 1
        return super().compute_logprob(
            prompt,
            response_token_ids,
            policy=policy,
            no_grad=no_grad,
        )

    def compute_logprob_batch(
        self,
        prompts: list[str],
        response_token_ids_batch: list[list[int]],
        *,
        policy: str = "actor",
        no_grad: bool = False,
    ) -> list[torch.Tensor]:
        self.batch_compute_calls.append(
            {
                "prompts": list(prompts),
                "policy": policy,
                "no_grad": no_grad,
            }
        )
        return [
            _ToyModelWrapper.compute_logprob(
                self,
                prompt,
                response_token_ids,
                policy=policy,
                no_grad=no_grad,
            )
            for prompt, response_token_ids in zip(prompts, response_token_ids_batch)
        ]


class _ResamplingToyModelWrapper(_ToyModelWrapper):
    def __init__(self):
        super().__init__()
        self.prompt_cfg = PromptConfig(stop_on="\n", enforce_single_line=True, min_terms=2, max_terms=8)
        self._responses = iter(
            [
                "same query\n\nUser query: ignored\nBetter BM25 query:",
                "same query\n\nUser query: ignored\nBetter BM25 query:",
                "same query\n\nUser query: ignored\nBetter BM25 query:",
                "novel query one\n\nUser query: ignored\nBetter BM25 query:",
                "novel query two\n\nUser query: ignored\nBetter BM25 query:",
                "novel query three\n\nUser query: ignored\nBetter BM25 query:",
            ]
        )
        self._old_logprobs = iter([0.0, -0.2, -0.4, -0.1, -0.3, -0.5])

    def generate_with_logprob(self, prompt: str, *, max_new_tokens: int, temperature: float, top_p: float) -> GeneratedSample:
        del prompt, max_new_tokens, temperature, top_p
        self.training_flags.append(self.actor_model.training)
        text = next(self._responses)
        old_logprob = next(self._old_logprobs)
        return GeneratedSample(
            response_text=text.split("\n", 1)[0],
            response_token_ids=[1, 2],
            logprob_old=torch.tensor([old_logprob, old_logprob], dtype=torch.float32),
            raw_response_text=text,
        )


class _ConstantRewarder:
    def __init__(self):
        self.cfg = RewardConfig()

    def score(self, qid: str, rewritten_query: str, source_query: str | None = None) -> RewardBreakdown:
        return RewardBreakdown(
            total=1.0,
            mrr=0.5,
            recall=1.0,
            overlap=0.2,
            hit_rank=1,
            retrieved_relevant_count=1,
            relevant_total=1,
            rewritten_query=rewritten_query,
        )


class CollapsedGroupTests(unittest.TestCase):
    def test_collapsed_group_skips_update_without_artificial_reward_offsets(self):
        wrapper = _DuplicateToyModelWrapper()
        rewarder = _ConstantRewarder()
        optimizer = torch.optim.SGD(wrapper.trainable_parameters(), lr=1e-2)
        engine = GRPOEngine(
            model_wrapper=wrapper,
            rewarder=rewarder,
            optimizer=optimizer,
            group_size=3,
            clip_range=0.2,
            kl_beta=0.01,
            grad_clip_norm=1.0,
            max_new_tokens=8,
            temperature=0.8,
            top_p=0.95,
            reward_gap_threshold=0.0,
            max_regen_rounds=0,
        )

        metrics = engine.train_step([QueryExample(qid="q1", text="same query")], collect_best_queries=False)
        summary = metrics["group_query_summaries"][0]

        self.assertEqual(summary["group_final_queries"], ["same query", "same query", "same query"])
        self.assertTrue(summary["collapsed_group"])
        self.assertEqual(summary["group_rewards"], [1.0, 1.0, 1.0])
        self.assertEqual(metrics["collapsed_group_ratio"], 1.0)
        self.assertEqual(metrics["unique_final_query_mean"], 1.0)
        self.assertEqual(metrics["all_same_final_query_ratio"], 1.0)
        self.assertEqual(metrics["flat_reward_group_ratio"], 1.0)
        self.assertEqual(metrics["adv_std"], 0.0)
        self.assertEqual(metrics["loss_pg_abs_mean"], 0.0)
        self.assertEqual(metrics["loss_pg"], 0.0)
        self.assertEqual(metrics["loss_kl"], 0.0)
        self.assertEqual(metrics["updated"], 0.0)
        self.assertEqual(metrics["valid_samples"], 0.0)

    def test_regeneration_breaks_group_collapse_with_novel_queries(self):
        wrapper = _ResamplingToyModelWrapper()
        rewarder = _ConstantRewarder()
        optimizer = torch.optim.SGD(wrapper.trainable_parameters(), lr=1e-2)
        engine = GRPOEngine(
            model_wrapper=wrapper,
            rewarder=rewarder,
            optimizer=optimizer,
            group_size=3,
            clip_range=0.2,
            kl_beta=0.01,
            grad_clip_norm=1.0,
            max_new_tokens=8,
            temperature=0.8,
            top_p=0.95,
            min_unique_final_queries=2,
            max_regen_rounds=1,
            regen_temperature_delta=0.15,
            reward_gap_threshold=0.0,
        )

        metrics = engine.train_step([QueryExample(qid="q1", text="same query")], collect_best_queries=False)
        summary = metrics["group_query_summaries"][0]

        self.assertGreaterEqual(metrics["unique_final_query_mean"], 2.0)
        self.assertEqual(metrics["collapsed_group_ratio"], 0.0)
        self.assertEqual(metrics["all_same_final_query_ratio"], 0.0)
        self.assertGreaterEqual(len(set(summary["group_final_queries"])), 2)
        self.assertIn("novel query one", summary["group_final_queries"])


class _AdaptiveSamplingToyModelWrapper(_ToyModelWrapper):
    def __init__(self, responses: list[str]):
        super().__init__()
        self.prompt_cfg = PromptConfig(stop_on="\n", enforce_single_line=True, min_terms=1, max_terms=8)
        self._responses = iter(responses)
        self.temperatures: list[float] = []

    def generate_with_logprob(self, prompt: str, *, max_new_tokens: int, temperature: float, top_p: float) -> GeneratedSample:
        del prompt, max_new_tokens, top_p
        self.training_flags.append(self.actor_model.training)
        self.temperatures.append(float(temperature))
        text = next(self._responses)
        old_logprob = -0.1 * float(len(self.temperatures))
        return GeneratedSample(
            response_text=text.split("\n", 1)[0],
            response_token_ids=[1, 2],
            logprob_old=torch.tensor([old_logprob, old_logprob], dtype=torch.float32),
            raw_response_text=text,
        )


class _ParallelAdaptiveSamplingToyModelWrapper(_AdaptiveSamplingToyModelWrapper):
    def __init__(self, responses: list[str]):
        super().__init__(responses)
        self.parallel_calls: list[tuple[int, float]] = []

    def generate_group_with_logprob(
        self,
        prompt: str,
        *,
        num_return_sequences: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[GeneratedSample]:
        self.parallel_calls.append((num_return_sequences, float(temperature)))
        return [
            self.generate_with_logprob(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            for _ in range(num_return_sequences)
        ]


class _MappedRewarder:
    def __init__(self, scores: dict[str, float]):
        self.cfg = RewardConfig()
        self.scores = dict(scores)
        self.score_calls: list[str] = []

    def score(self, qid: str, rewritten_query: str, source_query: str | None = None) -> RewardBreakdown:
        del qid, source_query
        self.score_calls.append(rewritten_query)
        total = float(self.scores.get(rewritten_query, 0.0))
        return RewardBreakdown(
            total=total,
            mrr=total,
            recall=0.0,
            overlap=0.0,
            hit_rank=1 if total > 0.0 else None,
            retrieved_relevant_count=0,
            relevant_total=0,
            rewritten_query=rewritten_query,
        )


class AdaptiveGapSamplingTests(unittest.TestCase):
    def test_gap_sampling_stops_when_initial_gap_is_enough(self):
        wrapper = _AdaptiveSamplingToyModelWrapper(["low query", "high query"])
        rewarder = _MappedRewarder({"low query": 0.1, "high query": 0.5})
        optimizer = torch.optim.SGD(wrapper.trainable_parameters(), lr=1e-2)
        engine = GRPOEngine(
            model_wrapper=wrapper,
            rewarder=rewarder,
            optimizer=optimizer,
            group_size=2,
            max_group_size=5,
            clip_range=0.2,
            kl_beta=0.01,
            grad_clip_norm=1.0,
            max_new_tokens=8,
            temperature=0.8,
            top_p=0.95,
            reward_gap_threshold=0.3,
            gap_sampling_temperature_delta=0.15,
            max_regen_rounds=0,
        )

        metrics = engine.train_step([QueryExample(qid="q1", text="input query")], collect_best_queries=False)
        summary = metrics["group_query_summaries"][0]

        self.assertEqual(summary["initial_group_size"], 2)
        self.assertEqual(summary["generated_sample_count"], 2)
        self.assertEqual(summary["extra_sample_count"], 0)
        self.assertAlmostEqual(summary["reward_gap_raw"], 0.4)
        self.assertTrue(summary["reward_gap_met"])
        self.assertEqual(summary["reward_gap_stop_reason"], "threshold_reached")
        self.assertEqual(summary["gap_sampling_rounds"], 0)
        self.assertEqual(summary["gap_sampling_temperatures"], [])
        self.assertEqual(metrics["generated_sample_count_mean"], 2.0)
        self.assertEqual(metrics["generated_sample_count_max"], 2.0)
        self.assertEqual(metrics["extra_sample_ratio"], 0.0)
        self.assertEqual(metrics["reward_gap_met_ratio"], 1.0)
        self.assertEqual(metrics["max_group_size_hit_ratio"], 0.0)

    def test_gap_sampling_adds_samples_until_threshold_is_met(self):
        wrapper = _AdaptiveSamplingToyModelWrapper(["flat query", "flat query", "high query"])
        rewarder = _MappedRewarder({"flat query": 0.1, "high query": 0.4})
        optimizer = torch.optim.SGD(wrapper.trainable_parameters(), lr=1e-2)
        engine = GRPOEngine(
            model_wrapper=wrapper,
            rewarder=rewarder,
            optimizer=optimizer,
            group_size=2,
            max_group_size=5,
            clip_range=0.2,
            kl_beta=0.01,
            grad_clip_norm=1.0,
            max_new_tokens=8,
            temperature=0.8,
            top_p=0.95,
            reward_gap_threshold=0.2,
            gap_sampling_temperature_delta=0.15,
            max_regen_rounds=0,
        )

        metrics = engine.train_step([QueryExample(qid="q1", text="input query")], collect_best_queries=False)
        summary = metrics["group_query_summaries"][0]

        self.assertEqual(summary["generated_sample_count"], 3)
        self.assertEqual(summary["extra_sample_count"], 1)
        self.assertTrue(summary["reward_gap_met"])
        self.assertEqual(summary["reward_gap_stop_reason"], "threshold_reached")
        self.assertEqual(summary["gap_sampling_rounds"], 1)
        self.assertEqual(summary["gap_sampling_temperatures"], [0.95])
        self.assertEqual(wrapper.temperatures, [0.8, 0.8, 0.95])
        self.assertEqual(summary["generated_sample_count"], len(summary["group_rewards"]))
        self.assertEqual(summary["generated_sample_count"], len(summary["group_final_queries"]))
        self.assertAlmostEqual(metrics["extra_sample_ratio"], 1.0 / 3.0)

    def test_gap_sampling_stops_at_max_group_size_when_gap_never_met(self):
        wrapper = _AdaptiveSamplingToyModelWrapper(
            ["flat query", "flat query", "flat query", "flat query"]
        )
        rewarder = _MappedRewarder({"flat query": 0.1})
        optimizer = torch.optim.SGD(wrapper.trainable_parameters(), lr=1e-2)
        engine = GRPOEngine(
            model_wrapper=wrapper,
            rewarder=rewarder,
            optimizer=optimizer,
            group_size=2,
            max_group_size=4,
            clip_range=0.2,
            kl_beta=0.01,
            grad_clip_norm=1.0,
            max_new_tokens=8,
            temperature=0.8,
            top_p=0.95,
            reward_gap_threshold=0.2,
            gap_sampling_temperature_delta=0.15,
            max_regen_rounds=0,
        )

        metrics = engine.train_step([QueryExample(qid="q1", text="input query")], collect_best_queries=False)
        summary = metrics["group_query_summaries"][0]

        self.assertEqual(summary["generated_sample_count"], 4)
        self.assertEqual(summary["extra_sample_count"], 2)
        self.assertFalse(summary["reward_gap_met"])
        self.assertEqual(summary["reward_gap_stop_reason"], "max_group_size_reached")
        self.assertEqual(summary["gap_sampling_rounds"], 2)
        self.assertEqual(summary["gap_sampling_temperatures"], [0.95, 1.1])
        self.assertEqual(wrapper.temperatures, [0.8, 0.8, 0.95, 1.1])
        self.assertEqual(metrics["reward_gap_met_ratio"], 0.0)
        self.assertEqual(metrics["max_group_size_hit_ratio"], 1.0)
        self.assertEqual(metrics["generated_sample_count_max"], 4.0)

    def test_gap_sampling_uses_raw_reward_gap_not_duplicate_penalized_reward(self):
        wrapper = _AdaptiveSamplingToyModelWrapper(["same query", "same query", "same query"])
        rewarder = _MappedRewarder({"same query": 1.0})
        optimizer = torch.optim.SGD(wrapper.trainable_parameters(), lr=1e-2)
        engine = GRPOEngine(
            model_wrapper=wrapper,
            rewarder=rewarder,
            optimizer=optimizer,
            group_size=3,
            max_group_size=3,
            clip_range=0.2,
            kl_beta=0.01,
            grad_clip_norm=1.0,
            max_new_tokens=8,
            temperature=0.8,
            top_p=0.95,
            reward_gap_threshold=0.02,
            gap_sampling_temperature_delta=0.15,
            max_regen_rounds=0,
        )

        metrics = engine.train_step([QueryExample(qid="q1", text="input query")], collect_best_queries=False)
        summary = metrics["group_query_summaries"][0]

        self.assertTrue(summary["collapsed_group"])
        self.assertEqual(summary["group_rewards"], [1.0, 1.0, 1.0])
        self.assertAlmostEqual(summary["reward_gap_raw"], 0.0)
        self.assertFalse(summary["reward_gap_met"])
        self.assertEqual(summary["reward_gap_stop_reason"], "max_group_size_reached")

    def test_gap_sampling_reuses_reward_cache_for_repeated_queries(self):
        wrapper = _AdaptiveSamplingToyModelWrapper(
            ["same query", "same query", "same query", "high query"]
        )
        rewarder = _MappedRewarder({"same query": 0.1, "high query": 0.5})
        optimizer = torch.optim.SGD(wrapper.trainable_parameters(), lr=1e-2)
        engine = GRPOEngine(
            model_wrapper=wrapper,
            rewarder=rewarder,
            optimizer=optimizer,
            group_size=2,
            max_group_size=4,
            clip_range=0.2,
            kl_beta=0.01,
            grad_clip_norm=1.0,
            max_new_tokens=8,
            temperature=0.8,
            top_p=0.95,
            reward_gap_threshold=0.2,
            gap_sampling_temperature_delta=0.15,
            max_regen_rounds=0,
        )

        metrics = engine.train_step([QueryExample(qid="q1", text="input query")], collect_best_queries=False)
        summary = metrics["group_query_summaries"][0]

        self.assertEqual(rewarder.score_calls, ["same query", "high query"])
        self.assertEqual(summary["generated_sample_count"], 4)
        self.assertEqual(summary["gap_sampling_rounds"], 2)
        self.assertTrue(summary["reward_gap_met"])

    def test_parallel_initial_sampling_still_works_with_gap_resampling(self):
        wrapper = _ParallelAdaptiveSamplingToyModelWrapper(
            ["flat query", "flat query", "high query"]
        )
        rewarder = _MappedRewarder({"flat query": 0.1, "high query": 0.5})
        optimizer = torch.optim.SGD(wrapper.trainable_parameters(), lr=1e-2)
        engine = GRPOEngine(
            model_wrapper=wrapper,
            rewarder=rewarder,
            optimizer=optimizer,
            group_size=2,
            max_group_size=3,
            clip_range=0.2,
            kl_beta=0.01,
            grad_clip_norm=1.0,
            max_new_tokens=8,
            temperature=0.8,
            top_p=0.95,
            reward_gap_threshold=0.2,
            gap_sampling_temperature_delta=0.15,
            max_regen_rounds=0,
            parallel_group_generate=True,
        )

        metrics = engine.train_step([QueryExample(qid="q1", text="input query")], collect_best_queries=False)
        summary = metrics["group_query_summaries"][0]

        self.assertEqual(wrapper.parallel_calls, [(2, 0.8)])
        self.assertEqual(summary["generated_sample_count"], 3)
        self.assertTrue(summary["reward_gap_met"])


class _FakeTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.name_or_path = "dummy/model"

    def __len__(self):
        return 32


class _FakeEmb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(32, 8))


class _FakeModel(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1))
        self._emb = _FakeEmb()
        self.config = SimpleNamespace(_name_or_path=name)
        self.hf_device_map = {"model.embed_tokens": "cuda:0", "lm_head": "cpu"}

    def get_input_embeddings(self):
        return self._emb


class RefPrecisionFallbackTests(unittest.TestCase):
    def test_ref_auto_fallback_from_full_to_4bit_on_oom(self):
        calls: list[dict] = []

        def _fake_from_pretrained(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return _FakeModel("dummy/model")
            if len(calls) == 2:
                raise RuntimeError("CUDA out of memory.")
            return _FakeModel("dummy/model")

        cfg = ModelConfig(
            model_name="dummy/model",
            trust_remote_code=False,
            load_in_4bit=False,
            actor_device_map="auto",
            ref_device_map="auto",
            ref_precision_mode="auto",
        )
        prompt_cfg = PromptConfig()

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=_FakeTokenizer()), patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            side_effect=_fake_from_pretrained,
        ), patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.is_bf16_supported", return_value=False
        ):
            wrapper = ModelWrapper(
                model_cfg=cfg,
                prompt_cfg=prompt_cfg,
                train_mode=False,
                enable_lora=False,
                load_ref_model=True,
            )

        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0]["device_map"], "auto")
        self.assertEqual(calls[1]["device_map"], "auto")
        self.assertEqual(calls[2]["device_map"], "auto")
        self.assertIsNone(calls[1]["quantization_config"])
        self.assertIsNotNone(calls[2]["quantization_config"])
        self.assertEqual(wrapper.ref_precision_used, "4bit")
        self.assertEqual(wrapper._infer_model_device(wrapper.ref_model), torch.device("cuda:0"))


if __name__ == "__main__":
    unittest.main()
