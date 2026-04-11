import argparse
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from app_config import ModelConfig, PromptConfig, get_default_config
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
        cfg.train.max_steps = 0
        cfg.reward.topk = 0
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
        self.assertEqual(adjusted.train.max_steps, 1)
        self.assertEqual(adjusted.reward.topk, 1)
        self.assertEqual(adjusted.data.max_train_queries, 0)
        self.assertEqual(adjusted.data.max_val_queries, 0)

    def test_runtime_adjustments_reject_invalid_train_ratio(self):
        cfg = get_default_config()
        cfg.data.train_ratio = 1.0
        args = argparse.Namespace(low_mem_mode=False, disable_4bit=False)

        with patch("torch.cuda.is_available", return_value=True):
            with self.assertRaises(ValueError):
                apply_runtime_mode_adjustments(cfg, args)


class _ToyModelWrapper:
    def __init__(self):
        self.actor_model = torch.nn.Linear(1, 1, bias=False)
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
    def score(self, qid: str, rewritten_query: str, source_query: str | None = None) -> RewardBreakdown:
        unread_penalty = 0.3 if "¤" in rewritten_query else 0.0
        return RewardBreakdown(
            total=1.0 - unread_penalty,
            mrr=0.5,
            overlap=0.4,
            penalty=unread_penalty,
            hit_rank=1,
            short_penalty=0.0,
            repeat_penalty=0.0,
            unreadable_penalty=unread_penalty,
            rewritten_query=f"clean::{rewritten_query}",
        )


class EngineTraceTests(unittest.TestCase):
    def test_group_trace_and_unreadable_ratio_metrics(self):
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
        )

        metrics = engine.train_step([QueryExample(qid="q1", text="input query")], collect_best_queries=True)
        summaries = metrics["group_query_summaries"]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["qid"], "q1")
        self.assertEqual(summaries[0]["input_query"], "input query")
        self.assertEqual(len(summaries[0]["group_raw_responses"]), 2)
        self.assertEqual(len(summaries[0]["group_cleaned_queries"]), 2)
        self.assertEqual(len(summaries[0]["group_rewards"]), 2)
        self.assertEqual(len(summaries[0]["group_unreadable_penalties"]), 2)
        self.assertGreater(metrics["unreadable_ratio_mean"], 0.0)
        self.assertTrue(all(flag is False for flag in wrapper.training_flags))
        self.assertTrue(wrapper.actor_model.training)


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
        self.assertEqual(calls[0]["device_map"], {"": 0})
        self.assertEqual(calls[1]["device_map"], {"": 0})
        self.assertEqual(calls[2]["device_map"], {"": 0})
        self.assertIsNone(calls[1]["quantization_config"])
        self.assertIsNotNone(calls[2]["quantization_config"])
        self.assertEqual(wrapper.ref_precision_used, "4bit")
        self.assertEqual(wrapper._infer_model_device(wrapper.ref_model), torch.device("cuda:0"))


if __name__ == "__main__":
    unittest.main()
