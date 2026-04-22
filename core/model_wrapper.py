from __future__ import annotations

"""模型封装层：负责 actor/ref 加载、生成、logprob 重算。

训练主线只需要三类能力：
1. 用 actor 生成 rewrite，并记录采样时的 old logprob；
2. 用 frozen ref 计算 KL 参考 logprob；
3. 用当前 actor 重算 new logprob，供 PPO/GRPO loss 使用。
"""

from contextlib import nullcontext
from dataclasses import dataclass
import gc
import json
from pathlib import Path
from typing import Any, Literal, Sequence

import torch

from app_config import ModelConfig, PromptConfig

PolicyName = Literal["actor", "ref"]


@dataclass(frozen=True, slots=True)
class GeneratedSample:
    """一次生成结果，包含文本、token 和采样时 logprob。"""

    # 清洗截断后的模型输出。
    response_text: str
    # 与 response_text 对齐的 token ids；后续重算 logprob 只看这些 token。
    response_token_ids: list[int]
    # 采样时 actor 对 response_token_ids 的 token-level logprob。
    logprob_old: torch.Tensor
    # 原始解码文本，保留给 trace 诊断模型是否输出了多余内容。
    raw_response_text: str = ""
    # The exact prompt used for this rollout. Retry rollouts may include diversity hints.
    prompt_text: str = ""


def _str_to_dtype(dtype_name: str) -> torch.dtype:
    """Internal helper."""

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[key]


class ModelWrapper:
    @staticmethod
    def _resolve_local_model_source(model_name: str) -> tuple[str, bool]:
        """Resolve local model directories, including Hugging Face cache roots."""

        model_dir = Path(model_name).expanduser()
        if not model_dir.is_dir():
            return model_name, False
        return str(ModelWrapper._resolve_hf_cache_snapshot(model_dir)), True

    @staticmethod
    def _resolve_hf_cache_snapshot(model_dir: Path) -> Path:
        """Map a Hugging Face cache root to its active snapshot when possible."""

        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.is_dir():
            return model_dir

        ref_main = model_dir / "refs" / "main"
        if ref_main.is_file():
            snapshot_name = ref_main.read_text(encoding="utf-8").strip()
            if snapshot_name:
                candidate = snapshots_dir / snapshot_name
                if candidate.is_dir():
                    return candidate

        snapshot_dirs = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if len(snapshot_dirs) == 1:
            return snapshot_dirs[0]
        return model_dir

    def __init__(
        self,
        model_cfg: ModelConfig,
        prompt_cfg: PromptConfig,
        *,
        train_mode: bool,
        enable_lora: bool = True,
        load_ref_model: bool = False,
        adapter_path: str | None = None,
        strict_tokenizer_model_match: bool = False,
    ) -> None:
        """初始化 actor，并按需加载 frozen reference model。"""

        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # 1) 解析模型路径：支持普通路径，也支持 Hugging Face cache root。
        self.model_cfg = model_cfg
        self.prompt_cfg = prompt_cfg
        self.train_mode = train_mode
        self.enable_lora = enable_lora
        self.adapter_path = adapter_path
        self.projection_chunk_size = max(1, int(getattr(model_cfg, "projection_chunk_size", 64)))
        self._rollout_batch_prompt_counts: list[int] = []
        self.ref_precision_used: str | None = None
        self.ref_dtype_used: torch.dtype | None = None
        self.actor_device_map = self._resolve_runtime_device_map(model_cfg.actor_device_map, model_role="actor")
        self.ref_device_map = self._resolve_runtime_device_map(model_cfg.ref_device_map, model_role="ref")
        original_model_source = str(Path(model_cfg.model_name).expanduser())
        self.model_source, self.local_model_only = self._resolve_local_model_source(model_cfg.model_name)
        if self.local_model_only:
            if self.model_source != original_model_source:
                print(f"[model] resolved local Hugging Face cache to snapshot: {self.model_source}")
            else:
                print(f"[model] using local model directory: {self.model_source}")

        if model_cfg.load_in_4bit and not torch.cuda.is_available():
            print("[warn] CUDA is unavailable; disabling 4-bit quantization and loading full precision on CPU.")
            model_cfg.load_in_4bit = False

        if model_cfg.load_in_4bit and model_cfg.bnb_4bit_compute_dtype.lower() == "bfloat16":
            bf16_supported = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            if not bf16_supported:
                print("[warn] bfloat16 4-bit compute dtype is unsupported on this runtime; fallback to float16.")
                model_cfg.bnb_4bit_compute_dtype = "float16"

        # 2) tokenizer 必须和基座模型对齐；pad_token 缺失时用 eos 兜底。
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_source,
            trust_remote_code=model_cfg.trust_remote_code,
            use_fast=False,
            local_files_only=self.local_model_only,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.quantization_config = self._build_4bit_config(enabled=model_cfg.load_in_4bit)
        actor_dtype = (
            _str_to_dtype(model_cfg.bnb_4bit_compute_dtype)
            if model_cfg.load_in_4bit
            else self._preferred_full_precision_dtype()
        )
        if not torch.cuda.is_available():
            actor_dtype = torch.float32

        # 3) 加载 actor 基座。训练主线默认 4bit + LoRA。
        base_actor = AutoModelForCausalLM.from_pretrained(
            self.model_source,
            trust_remote_code=model_cfg.trust_remote_code,
            quantization_config=self.quantization_config,
            device_map=self.actor_device_map,
            torch_dtype=actor_dtype,
            local_files_only=self.local_model_only,
        )
        self._validate_tokenizer_model_match(
            tokenizer=self.tokenizer,
            model=base_actor,
            model_name=self.model_source,
            adapter_path=adapter_path,
            strict=strict_tokenizer_model_match,
        )

        if enable_lora:
            if train_mode:
                base_actor = prepare_model_for_kbit_training(base_actor)

            # phase2/eval 会从已有 adapter 热启动；phase1 则新建 LoRA。
            if adapter_path:
                self.actor_model = PeftModel.from_pretrained(base_actor, adapter_path, is_trainable=train_mode)
            else:
                lora_cfg = LoraConfig(
                    r=model_cfg.lora_r,
                    lora_alpha=model_cfg.lora_alpha,
                    lora_dropout=model_cfg.lora_dropout,
                    target_modules=list(model_cfg.lora_target_modules),
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.actor_model = get_peft_model(base_actor, lora_cfg)
        else:
            self.actor_model = base_actor

        self.actor_model.train(train_mode)
        if train_mode and hasattr(self.actor_model, "config"):
            self.actor_model.config.use_cache = False

        self.ref_model = None
        if load_ref_model:
            # ref 只参与 KL，不训练；保持 eval 模式并冻结所有参数。
            self.ref_model = self._load_ref_model(AutoModelForCausalLM)
            self.ref_model.eval()
            if hasattr(self.ref_model, "config"):
                self.ref_model.config.use_cache = False
            for param in self.ref_model.parameters():
                param.requires_grad = False

    def _build_4bit_config(self, *, enabled: bool):
        """Internal helper."""

        if not enabled:
            return None
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.model_cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.model_cfg.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=_str_to_dtype(self.model_cfg.bnb_4bit_compute_dtype),
        )

    @staticmethod
    def _is_cuda_oom_error(exc: BaseException) -> bool:
        """Internal helper."""

        text = str(exc).lower()
        return "out of memory" in text and ("cuda" in text or "cublas" in text)

    @staticmethod
    def _preferred_full_precision_dtype() -> torch.dtype:
        """Internal helper."""

        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    @staticmethod
    def _resolve_runtime_device_map(configured_map: Any, *, model_role: str) -> Any:
        """Resolve runtime device_map without forcing a specific GPU index."""

        if torch.cuda.is_available() and isinstance(configured_map, str) and configured_map.strip().lower() == "auto":
            print(f"[model] using {model_role} device_map='auto' (no hard-coded GPU index).")
        return configured_map

    @staticmethod
    def _to_device_from_map_value(value: Any) -> torch.device | None:
        """Internal helper."""

        if isinstance(value, torch.device):
            return value
        if isinstance(value, int):
            return torch.device(f"cuda:{value}")
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered.startswith("cuda"):
                return torch.device(value)
            if lowered.startswith("cpu"):
                return torch.device("cpu")
        return None

    @classmethod
    def _extract_hf_device_map(cls, model: torch.nn.Module) -> dict[str, Any] | None:
        """Internal helper."""

        direct = getattr(model, "hf_device_map", None)
        if isinstance(direct, dict):
            return direct

        for attr in ("base_model", "model"):
            nested = getattr(model, attr, None)
            if nested is None:
                continue
            nested_map = getattr(nested, "hf_device_map", None)
            if isinstance(nested_map, dict):
                return nested_map
            deeper = getattr(nested, "model", None)
            deeper_map = getattr(deeper, "hf_device_map", None) if deeper is not None else None
            if isinstance(deeper_map, dict):
                return deeper_map
        return None

    @classmethod
    def _summarize_device_map(cls, model: torch.nn.Module) -> dict[str, int]:
        """Internal helper."""

        device_map = cls._extract_hf_device_map(model)
        if not device_map:
            try:
                return {str(next(model.parameters()).device): 1}
            except StopIteration:
                return {"cpu": 0}

        summary: dict[str, int] = {}
        for value in device_map.values():
            device = cls._to_device_from_map_value(value)
            key = str(device) if device is not None else str(value)
            summary[key] = summary.get(key, 0) + 1
        return summary

    def _load_ref_model(self, auto_model_cls):
        """加载 top20 curriculum 主线的 frozen KL reference model。"""

        # ref 默认跟 actor 一样走 4bit，避免再引入 auto/full 分支。
        use_4bit = bool(self.model_cfg.load_in_4bit)
        dtype = _str_to_dtype(self.model_cfg.bnb_4bit_compute_dtype) if use_4bit else self._preferred_full_precision_dtype()
        quant_cfg = self._build_4bit_config(enabled=use_4bit)
        ref_model = auto_model_cls.from_pretrained(
            self.model_source,
            trust_remote_code=self.model_cfg.trust_remote_code,
            quantization_config=quant_cfg,
            device_map=self.ref_device_map,
            torch_dtype=dtype,
            local_files_only=self.local_model_only,
        )
        self.ref_precision_used = "4bit" if use_4bit else "full"
        self.ref_dtype_used = dtype

        print(
            "[ref] loaded: "
            f"precision={self.ref_precision_used}, "
            f"dtype={self.ref_dtype_used}, "
            f"device_map={self._summarize_device_map(ref_model)}"
        )
        return ref_model

    @classmethod
    def _infer_model_device(cls, model: torch.nn.Module) -> torch.device:
        """Internal helper."""

        device_map = cls._extract_hf_device_map(model)
        if isinstance(device_map, dict) and device_map:
            preferred_keys = (
                "model.embed_tokens",
                "model.decoder.embed_tokens",
                "transformer.wte",
                "embed_tokens",
            )
            for key in preferred_keys:
                if key in device_map:
                    preferred = cls._to_device_from_map_value(device_map[key])
                    if preferred is not None and preferred.type == "cuda":
                        return preferred

            for value in device_map.values():
                candidate = cls._to_device_from_map_value(value)
                if candidate is not None and candidate.type == "cuda":
                    return candidate

            for value in device_map.values():
                candidate = cls._to_device_from_map_value(value)
                if candidate is not None:
                    return candidate

        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _infer_module_device(module: torch.nn.Module) -> torch.device:
        """Infer the execution device for standalone modules such as lm_head."""

        try:
            return next(module.parameters()).device
        except StopIteration:
            try:
                return next(module.buffers()).device
            except StopIteration:
                return torch.device("cpu")

    @classmethod
    def _resolve_causal_lm_stack(cls, model: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module] | None:
        """Find the causal-LM backbone and lm_head through wrapper layers."""

        queue: list[torch.nn.Module] = [model]
        seen: set[int] = set()

        get_base_model = getattr(model, "get_base_model", None)
        if callable(get_base_model):
            try:
                base = get_base_model()
            except TypeError:
                base = None
            if isinstance(base, torch.nn.Module):
                queue.append(base)

        while queue:
            candidate = queue.pop(0)
            candidate_id = id(candidate)
            if candidate_id in seen:
                continue
            seen.add(candidate_id)

            backbone = getattr(candidate, "model", None)
            lm_head = getattr(candidate, "lm_head", None)
            if isinstance(backbone, torch.nn.Module) and isinstance(lm_head, torch.nn.Module):
                # PEFT 包装层可能把完整 *ForCausalLM 放在 `.model` 里；
                # 这里一路向内找到真正返回 hidden_states 的 decoder backbone。
                while True:
                    nested_backbone = getattr(backbone, "model", None)
                    nested_lm_head = getattr(backbone, "lm_head", None)
                    if isinstance(nested_backbone, torch.nn.Module) and isinstance(nested_lm_head, torch.nn.Module):
                        backbone = nested_backbone
                        continue
                    break
                return backbone, lm_head

            for attr in ("base_model", "model"):
                nested = getattr(candidate, attr, None)
                if isinstance(nested, torch.nn.Module):
                    queue.append(nested)
        return None

    @staticmethod
    def _gather_selected_logprobs(
        *,
        lm_head: torch.nn.Module,
        selected_hidden_states: torch.Tensor,
        target_ids: torch.Tensor,
        projection_chunk_size: int = 32,
    ) -> torch.Tensor:
        """只投影 response token 位置，避免全 vocab projection 撑爆显存。"""

        if selected_hidden_states.numel() == 0:
            return torch.empty(0, dtype=torch.float32, device=selected_hidden_states.device)

        chunk_size = max(1, int(projection_chunk_size))
        gathered: list[torch.Tensor] = []
        for start in range(0, int(selected_hidden_states.shape[0]), chunk_size):
            end = start + chunk_size
            hidden_chunk = selected_hidden_states[start:end]
            target_chunk = target_ids[start:end]
            logits_chunk = lm_head(hidden_chunk)
            target_logits = logits_chunk.gather(-1, target_chunk.unsqueeze(-1)).squeeze(-1).to(torch.float32)
            log_norm = torch.logsumexp(logits_chunk.to(torch.float32), dim=-1)
            gathered.append(target_logits - log_norm)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def _normalize_model_id(name: str) -> str:
        """Internal helper."""

        return name.replace("\\", "/").rstrip("/").lower()

    @classmethod
    def _same_model_id(cls, lhs: str, rhs: str) -> bool:
        """Internal helper."""

        left = cls._normalize_model_id(lhs)
        right = cls._normalize_model_id(rhs)
        if not left or not right:
            return False
        if left == right:
            return True
        if left.endswith("/" + right) or right.endswith("/" + left):
            return True
        return Path(left).name == Path(right).name

    def _validate_tokenizer_model_match(
        self,
        *,
        tokenizer,
        model: torch.nn.Module,
        model_name: str,
        adapter_path: str | None,
        strict: bool,
    ) -> None:
        """Internal helper."""

        tok_name = str(getattr(tokenizer, "name_or_path", ""))
        cfg_name = str(getattr(getattr(model, "config", None), "_name_or_path", "")) or model_name
        tok_vocab = int(len(tokenizer))
        emb = model.get_input_embeddings()
        model_vocab = int(emb.weight.shape[0]) if emb is not None else -1

        if strict and tok_name and not self._same_model_id(tok_name, model_name):
            raise ValueError(
                f"Tokenizer path/name '{tok_name}' does not match runtime model '{model_name}'. "
                "Please use the tokenizer from the same base model."
            )

        if model_vocab > 0 and tok_vocab > model_vocab:
            raise ValueError(
                f"Tokenizer vocab ({tok_vocab}) is larger than model embedding size ({model_vocab}). "
                "Tokenizer/model are incompatible."
            )
        if model_vocab > 0 and tok_vocab != model_vocab:
            delta = abs(tok_vocab - model_vocab)
            msg = (
                f"[warn] tokenizer/model vocab size mismatch: tokenizer={tok_vocab}, model={model_vocab}. "
                "Small gaps are often harmless (reserved/unused embeddings), large gaps may be risky."
            )
            if strict and delta > 2048:
                raise ValueError(msg)
            print(msg)

        adapter_base = None
        if adapter_path:
            adapter_cfg_path = Path(adapter_path) / "adapter_config.json"
            if adapter_cfg_path.exists():
                try:
                    adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
                    adapter_base = str(adapter_cfg.get("base_model_name_or_path", "")).strip() or None
                except Exception:
                    adapter_base = None
            if adapter_base is not None:
                norm_adapter = self._normalize_model_id(adapter_base)
                norm_expected = self._normalize_model_id(model_name)
                if norm_adapter != norm_expected:
                    msg = (
                        f"[warn] adapter base model mismatch: adapter expects '{adapter_base}', "
                        f"but runtime model is '{model_name}'."
                    )
                    if strict:
                        raise ValueError(msg)
                    print(msg)

        print(
            "[sanity] tokenizer/model check: "
            f"tokenizer='{tok_name}', model='{cfg_name}', "
            f"tok_vocab={tok_vocab}, model_vocab={model_vocab}, "
            f"adapter_base='{adapter_base or '-'}', strict={strict}"
        )

    def build_prompt(self, query: str) -> str:
        """把原始 query 填进 top20 prompt 模板。"""

        query_clean = " ".join(query.strip().split())
        return f"{self.prompt_cfg.system_prompt}\n\n{self.prompt_cfg.template.format(query=query_clean)}"

    @staticmethod
    def _dedupe_keep_order(values: Sequence[str]) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return tuple(ordered)

    def _generation_stop_strings(self) -> tuple[str, ...]:
        stop_strings: list[str] = []
        stop_on = getattr(self.prompt_cfg, "stop_on", None)
        if isinstance(stop_on, str) and stop_on:
            stop_strings.append(stop_on)
        configured = getattr(self.prompt_cfg, "stop_strings", ()) or ()
        stop_strings.extend(str(value) for value in configured if value)
        stop_strings.extend(
            [
                "\nUser query:",
                "\nBetter BM25 query:",
                "\nSearch query:",
                "\nRewritten query:",
                "\nExample",
            ]
        )
        return self._dedupe_keep_order(stop_strings)

    def _truncate_generated_text(self, text: str) -> str:
        # 模型可能继续生成下一段示例或标签；这里按 stop strings 截断到第一行 query。
        raw = text or ""
        if not raw:
            return ""

        cut_positions: list[int] = []
        if getattr(self.prompt_cfg, "enforce_single_line", False):
            for marker in ("\n", "\r"):
                idx = raw.find(marker)
                if idx >= 0:
                    cut_positions.append(idx)

        for stop in self._generation_stop_strings():
            idx = raw.find(stop)
            if idx >= 0:
                cut_positions.append(idx)

        lowered = raw.lower()
        for marker in (
            "\nuser query:",
            "\nbetter bm25 query:",
            "\nsearch query:",
            "\nrewritten query:",
            "\nexample\n",
        ):
            idx = lowered.find(marker)
            if idx >= 0:
                cut_positions.append(idx)
        for prefix in (
            "user query:",
            "better bm25 query:",
            "search query:",
            "rewritten query:",
            "example\n",
        ):
            if lowered.startswith(prefix):
                cut_positions.append(0)

        cutoff = min(cut_positions) if cut_positions else len(raw)
        return raw[:cutoff].strip()

    def _aligned_prefix_length(self, response_ids: list[int], target_text: str) -> int:
        target = (target_text or "").strip()
        if not target:
            return 0

        encoded_target = self.tokenizer.encode(target, add_special_tokens=False)
        if encoded_target and response_ids[: len(encoded_target)] == encoded_target:
            return len(encoded_target)

        for prefix_len in range(1, len(response_ids) + 1):
            prefix_text = self.tokenizer.decode(response_ids[:prefix_len], skip_special_tokens=True).strip()
            if self._truncate_generated_text(prefix_text) == target:
                return prefix_len
        return len(response_ids)

    def _build_generate_kwargs(
        self,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        with_logprob: bool,
        num_return_sequences: int | None = None,
    ) -> dict[str, Any]:
        do_sample = temperature > 0.0
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": max(temperature, 1e-6) if do_sample else 1.0,
            "top_p": top_p if do_sample else 1.0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
        }
        if with_logprob:
            kwargs["output_scores"] = True
        if num_return_sequences is not None:
            kwargs["num_return_sequences"] = num_return_sequences

        stop_strings = list(self._generation_stop_strings())
        if stop_strings:
            kwargs["stop_strings"] = stop_strings
            kwargs["tokenizer"] = self.tokenizer
        return kwargs

    def _finalize_generated_sample(
        self,
        response_ids: list[int],
        *,
        scores: Sequence[torch.Tensor] | None,
        sequence_index: int,
        with_logprob: bool,
        prompt_text: str = "",
    ) -> GeneratedSample:
        # generate 返回的是完整序列；这里切掉 prompt，只保留 response，
        # 并把 token logprob 对齐到截断后的 response_text。
        raw_response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        response_text = self._truncate_generated_text(raw_response_text)
        prefix_len = self._aligned_prefix_length(response_ids, response_text)
        truncated_ids = response_ids[:prefix_len]

        if not with_logprob or not truncated_ids:
            return GeneratedSample(
                response_text=response_text,
                response_token_ids=truncated_ids,
                logprob_old=torch.empty(0),
                raw_response_text=raw_response_text,
                prompt_text=prompt_text,
            )

        score_steps = list(scores or [])
        steps = min(len(score_steps), len(truncated_ids))
        if steps == 0:
            return GeneratedSample(
                response_text=response_text,
                response_token_ids=truncated_ids,
                logprob_old=torch.empty(0),
                raw_response_text=raw_response_text,
                prompt_text=prompt_text,
            )

        token_logprobs: list[torch.Tensor] = []
        for step_idx in range(steps):
            logits_step = score_steps[step_idx][sequence_index].float()
            token_id = truncated_ids[step_idx]
            logprob_step = torch.log_softmax(logits_step, dim=-1)[token_id]
            token_logprobs.append(logprob_step.detach().cpu())

        return GeneratedSample(
            response_text=response_text,
            response_token_ids=truncated_ids[:steps],
            logprob_old=torch.stack(token_logprobs).to(torch.float32),
            raw_response_text=raw_response_text,
            prompt_text=prompt_text,
        )

    def _policy_model(self, policy: PolicyName) -> torch.nn.Module:
        """Internal helper."""

        if policy == "actor":
            return self.actor_model
        if self.ref_model is None:
            raise ValueError("Reference model is not loaded. Set load_ref_model=True.")
        return self.ref_model

    def _generate(
        self,
        model: torch.nn.Module,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        with_logprob: bool,
    ) -> GeneratedSample:
        """执行一次生成；with_logprob=True 时同时收集 old policy logprob。"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = self._infer_model_device(model)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        kwargs = self._build_generate_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            with_logprob=with_logprob,
        )

        with torch.no_grad():
            output = model.generate(**inputs, **kwargs)

        sequence = output.sequences[0]
        prompt_len = int(inputs["input_ids"].shape[1])
        response_ids = sequence[prompt_len:].tolist()
        return self._finalize_generated_sample(
            response_ids,
            scores=output.scores,
            sequence_index=0,
            with_logprob=with_logprob,
            prompt_text=prompt,
        )

    def generate_with_logprob(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> GeneratedSample:
        """Internal helper."""

        return self._generate(
            self.actor_model,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            with_logprob=True,
        )

    def reset_rollout_batch_stats(self) -> None:
        self._rollout_batch_prompt_counts.clear()

    def consume_rollout_batch_stats(self) -> list[int]:
        counts = list(self._rollout_batch_prompt_counts)
        self._rollout_batch_prompt_counts.clear()
        return counts

    def _generate_with_logprob_batch_once(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[GeneratedSample]:
        """Generate one sampled response per prompt in a single actor forward."""

        prompt_list = list(prompts)
        if not prompt_list:
            return []

        original_padding_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.tokenizer(
                prompt_list,
                return_tensors="pt",
                padding=True,
            )
        finally:
            self.tokenizer.padding_side = original_padding_side

        device = self._infer_model_device(self.actor_model)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[1])
        kwargs = self._build_generate_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            with_logprob=True,
        )

        with torch.no_grad():
            output = self.actor_model.generate(**inputs, **kwargs)
        self._rollout_batch_prompt_counts.append(len(prompt_list))

        sequences = output.sequences
        scores = output.scores or []
        results: list[GeneratedSample] = []
        for seq_idx in range(int(sequences.shape[0])):
            sequence = sequences[seq_idx]
            response_ids = sequence[prompt_len:].tolist()
            results.append(
                self._finalize_generated_sample(
                    response_ids,
                    scores=scores,
                    sequence_index=seq_idx,
                    with_logprob=True,
                    prompt_text=prompt_list[seq_idx],
                )
            )
        return results

    def generate_with_logprob_batch(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[GeneratedSample]:
        """Generate one rollout per prompt, splitting automatically on CUDA OOM."""

        prompt_list = list(prompts)
        if not prompt_list:
            return []
        if len(prompt_list) == 1:
            sample = self.generate_with_logprob(
                prompt_list[0],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            self._rollout_batch_prompt_counts.append(1)
            return [sample]

        try:
            return self._generate_with_logprob_batch_once(
                prompt_list,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        except RuntimeError as exc:
            if not (torch.cuda.is_available() and self._is_cuda_oom_error(exc)):
                raise
            gc.collect()
            torch.cuda.empty_cache()
            mid = max(1, len(prompt_list) // 2)
            left = self.generate_with_logprob_batch(
                prompt_list[:mid],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            right = self.generate_with_logprob_batch(
                prompt_list[mid:],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return left + right

    def generate_group_with_logprob(
        self,
        prompt: str,
        *,
        num_return_sequences: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[GeneratedSample]:
        """Internal helper."""

        requested = max(1, int(num_return_sequences))
        if requested == 1:
            return [
                self.generate_with_logprob(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            ]

        if temperature <= 0.0:
            return [
                self.generate_with_logprob(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                for _ in range(requested)
            ]

        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = self._infer_model_device(self.actor_model)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[1])
        kwargs = self._build_generate_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            with_logprob=True,
            num_return_sequences=requested,
        )

        with torch.no_grad():
            output = self.actor_model.generate(**inputs, **kwargs)

        sequences = output.sequences
        scores = output.scores or []
        sample_count = int(sequences.shape[0])

        results: list[GeneratedSample] = []
        for seq_idx in range(sample_count):
            sequence = sequences[seq_idx]
            response_ids = sequence[prompt_len:].tolist()
            results.append(
                self._finalize_generated_sample(
                    response_ids,
                    scores=scores,
                    sequence_index=seq_idx,
                    with_logprob=True,
                    prompt_text=prompt,
                )
            )

        if len(results) < requested:
            results.extend(
                [
                    self.generate_with_logprob(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    for _ in range(requested - len(results))
                ]
            )
        return results[:requested]

    def generate_rewrite(
        self,
        query: str,
        *,
        policy: PolicyName = "actor",
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Internal helper."""

        model = self._policy_model(policy)
        prompt = self.build_prompt(query)
        generated = self._generate(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            with_logprob=False,
        )
        return generated.response_text

    def _generate_rewrite_batch_once(
        self,
        queries: Sequence[str],
        *,
        policy: PolicyName,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        """Generate rewrites for a batch of queries in a single forward pass."""

        model = self._policy_model(policy)
        prompts = [self.build_prompt(query) for query in queries]
        if not prompts:
            return []

        original_padding_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
            )
        finally:
            self.tokenizer.padding_side = original_padding_side

        device = self._infer_model_device(model)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[1])
        kwargs = self._build_generate_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            with_logprob=False,
        )

        with torch.no_grad():
            output = model.generate(**inputs, **kwargs)

        sequences = output.sequences
        response_ids = sequences[:, prompt_len:]
        response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        return [self._truncate_generated_text(text.strip()) for text in response_texts]

    def generate_rewrite_batch(
        self,
        queries: Sequence[str],
        *,
        policy: PolicyName = "actor",
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        """Generate rewrites for a query batch, with OOM-safe split fallback."""

        query_list = list(queries)
        if not query_list:
            return []
        if len(query_list) == 1:
            return [
                self.generate_rewrite(
                    query_list[0],
                    policy=policy,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            ]

        try:
            return self._generate_rewrite_batch_once(
                query_list,
                policy=policy,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        except RuntimeError as exc:
            if not (torch.cuda.is_available() and self._is_cuda_oom_error(exc)):
                raise
            # Split-and-retry keeps evaluation running when a large batch hits VRAM limit.
            gc.collect()
            torch.cuda.empty_cache()
            mid = max(1, len(query_list) // 2)
            left = self.generate_rewrite_batch(
                query_list[:mid],
                policy=policy,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            right = self.generate_rewrite_batch(
                query_list[mid:],
                policy=policy,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return left + right

    def _compute_logprob_batch_from_full_logits(
        self,
        prompts: Sequence[str],
        response_token_ids_batch: Sequence[Sequence[int]],
        *,
        policy: PolicyName,
        no_grad: bool,
    ) -> list[torch.Tensor]:
        """Compatibility path that materializes full logits."""

        prompt_list = list(prompts)
        response_list = [list(token_ids) for token_ids in response_token_ids_batch]
        if len(prompt_list) != len(response_list):
            raise ValueError("prompts and response_token_ids_batch must have the same length.")
        if not prompt_list:
            return []

        model = self._policy_model(policy)
        device = self._infer_model_device(model)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise RuntimeError("Tokenizer must define pad_token_id or eos_token_id for batched logprob.")

        outputs: list[torch.Tensor | None] = [None] * len(prompt_list)
        active_indices: list[int] = []
        full_sequences: list[torch.Tensor] = []
        prompt_lengths: list[int] = []
        response_tensors: list[torch.Tensor] = []

        for index, (prompt, response_token_ids) in enumerate(zip(prompt_list, response_list)):
            if not response_token_ids:
                outputs[index] = torch.empty(0, device=device)
                continue

            prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            if prompt_ids.shape[1] == 0:
                eos_id = self.tokenizer.eos_token_id
                if eos_id is None:
                    raise RuntimeError("Tokenizer has no eos token id and prompt is empty.")
                prompt_ids = torch.tensor([[eos_id]], dtype=torch.long, device=device)

            prompt_row = prompt_ids.squeeze(0)
            response_tensor = torch.tensor(response_token_ids, dtype=torch.long, device=device)
            full_sequences.append(torch.cat([prompt_row, response_tensor], dim=0))
            prompt_lengths.append(int(prompt_row.shape[0]))
            response_tensors.append(response_tensor)
            active_indices.append(index)

        if not active_indices:
            return [tensor if tensor is not None else torch.empty(0, device=device) for tensor in outputs]

        max_seq_len = max(int(sequence.shape[0]) for sequence in full_sequences)
        batch_input_ids = torch.full(
            (len(full_sequences), max_seq_len),
            int(pad_token_id),
            dtype=torch.long,
            device=device,
        )
        batch_attention_mask = torch.zeros(
            (len(full_sequences), max_seq_len),
            dtype=torch.long,
            device=device,
        )
        for row_index, sequence in enumerate(full_sequences):
            seq_len = int(sequence.shape[0])
            batch_input_ids[row_index, :seq_len] = sequence
            batch_attention_mask[row_index, :seq_len] = 1

        grad_ctx = torch.no_grad() if no_grad else nullcontext()
        with grad_ctx:
            logits = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                use_cache=False,
            ).logits

        for row_index, original_index in enumerate(active_indices):
            response_tensor = response_tensors[row_index]
            prompt_len = prompt_lengths[row_index]
            start = max(prompt_len - 1, 0)
            end = start + int(response_tensor.shape[0])
            token_logits = logits[row_index : row_index + 1, start:end, :]
            target_ids = response_tensor.unsqueeze(0)[:, : token_logits.shape[1]]
            log_probs = torch.log_softmax(token_logits, dim=-1)
            outputs[original_index] = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)

        return [tensor if tensor is not None else torch.empty(0, device=device) for tensor in outputs]

    def _compute_logprob_batch_once(
        self,
        prompts: Sequence[str],
        response_token_ids_batch: Sequence[Sequence[int]],
        *,
        policy: PolicyName,
        no_grad: bool,
    ) -> list[torch.Tensor]:
        """Recompute token logprobs while only projecting response positions."""

        prompt_list = list(prompts)
        response_list = [list(token_ids) for token_ids in response_token_ids_batch]
        if len(prompt_list) != len(response_list):
            raise ValueError("prompts and response_token_ids_batch must have the same length.")
        if not prompt_list:
            return []

        model = self._policy_model(policy)
        device = self._infer_model_device(model)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise RuntimeError("Tokenizer must define pad_token_id or eos_token_id for batched logprob.")

        outputs: list[torch.Tensor | None] = [None] * len(prompt_list)
        active_indices: list[int] = []
        full_sequences: list[torch.Tensor] = []
        prompt_lengths: list[int] = []
        response_tensors: list[torch.Tensor] = []

        for index, (prompt, response_token_ids) in enumerate(zip(prompt_list, response_list)):
            if not response_token_ids:
                outputs[index] = torch.empty(0, device=device)
                continue

            prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            if prompt_ids.shape[1] == 0:
                eos_id = self.tokenizer.eos_token_id
                if eos_id is None:
                    raise RuntimeError("Tokenizer has no eos token id and prompt is empty.")
                prompt_ids = torch.tensor([[eos_id]], dtype=torch.long, device=device)

            prompt_row = prompt_ids.squeeze(0)
            response_tensor = torch.tensor(response_token_ids, dtype=torch.long, device=device)
            full_sequences.append(torch.cat([prompt_row, response_tensor], dim=0))
            prompt_lengths.append(int(prompt_row.shape[0]))
            response_tensors.append(response_tensor)
            active_indices.append(index)

        if not active_indices:
            return [tensor if tensor is not None else torch.empty(0, device=device) for tensor in outputs]

        causal_lm_stack = self._resolve_causal_lm_stack(model)
        if causal_lm_stack is None:
            return self._compute_logprob_batch_from_full_logits(
                prompt_list,
                response_list,
                policy=policy,
                no_grad=no_grad,
            )
        backbone, lm_head = causal_lm_stack

        max_seq_len = max(int(sequence.shape[0]) for sequence in full_sequences)
        batch_input_ids = torch.full(
            (len(full_sequences), max_seq_len),
            int(pad_token_id),
            dtype=torch.long,
            device=device,
        )
        batch_attention_mask = torch.zeros(
            (len(full_sequences), max_seq_len),
            dtype=torch.long,
            device=device,
        )
        for row_index, sequence in enumerate(full_sequences):
            seq_len = int(sequence.shape[0])
            batch_input_ids[row_index, :seq_len] = sequence
            batch_attention_mask[row_index, :seq_len] = 1

        grad_ctx = torch.no_grad() if no_grad else nullcontext()
        with grad_ctx:
            try:
                backbone_outputs = backbone(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            except TypeError:
                return self._compute_logprob_batch_from_full_logits(
                    prompt_list,
                    response_list,
                    policy=policy,
                    no_grad=no_grad,
                )

            hidden_states = getattr(backbone_outputs, "last_hidden_state", None)
            if hidden_states is None:
                if isinstance(backbone_outputs, tuple) and backbone_outputs:
                    hidden_states = backbone_outputs[0]
                elif getattr(backbone_outputs, "logits", None) is not None:
                    return self._compute_logprob_batch_from_full_logits(
                        prompt_list,
                        response_list,
                        policy=policy,
                        no_grad=no_grad,
                    )
                else:
                    raise RuntimeError("Backbone forward did not return last_hidden_state.")

            selected_hidden_states: list[torch.Tensor] = []
            selected_target_ids: list[torch.Tensor] = []
            token_counts: list[int] = []
            for row_index, response_tensor in enumerate(response_tensors):
                prompt_len = prompt_lengths[row_index]
                start = max(prompt_len - 1, 0)
                end = start + int(response_tensor.shape[0])
                token_hidden_states = hidden_states[row_index, start:end, :]
                target_ids = response_tensor[: int(token_hidden_states.shape[0])]
                selected_hidden_states.append(token_hidden_states)
                selected_target_ids.append(target_ids)
                token_counts.append(int(target_ids.shape[0]))

            flat_hidden_states = torch.cat(selected_hidden_states, dim=0)
            flat_target_ids = torch.cat(selected_target_ids, dim=0)
            lm_head_device = self._infer_module_device(lm_head)
            if flat_hidden_states.device != lm_head_device:
                flat_hidden_states = flat_hidden_states.to(lm_head_device)
            if flat_target_ids.device != lm_head_device:
                flat_target_ids = flat_target_ids.to(lm_head_device)
            flat_logprobs = self._gather_selected_logprobs(
                lm_head=lm_head,
                selected_hidden_states=flat_hidden_states,
                target_ids=flat_target_ids,
                projection_chunk_size=self.projection_chunk_size,
            )

        offset = 0
        for row_index, original_index in enumerate(active_indices):
            token_count = token_counts[row_index]
            outputs[original_index] = flat_logprobs[offset : offset + token_count]
            offset += token_count

        return [tensor if tensor is not None else torch.empty(0, device=device) for tensor in outputs]

    def compute_logprob_batch(
        self,
        prompts: Sequence[str],
        response_token_ids_batch: Sequence[Sequence[int]],
        *,
        policy: PolicyName = "actor",
        no_grad: bool = False,
    ) -> list[torch.Tensor]:
        """批量重算 logprob；遇到 CUDA OOM 会自动二分降批。"""

        prompt_list = list(prompts)
        response_list = [list(token_ids) for token_ids in response_token_ids_batch]
        if len(prompt_list) != len(response_list):
            raise ValueError("prompts and response_token_ids_batch must have the same length.")
        if not prompt_list:
            return []

        try:
            return self._compute_logprob_batch_once(
                prompt_list,
                response_list,
                policy=policy,
                no_grad=no_grad,
            )
        except RuntimeError as exc:
            if not (torch.cuda.is_available() and self._is_cuda_oom_error(exc) and len(prompt_list) > 1):
                raise
            gc.collect()
            torch.cuda.empty_cache()
            mid = max(1, len(prompt_list) // 2)
            left = self.compute_logprob_batch(
                prompt_list[:mid],
                response_list[:mid],
                policy=policy,
                no_grad=no_grad,
            )
            right = self.compute_logprob_batch(
                prompt_list[mid:],
                response_list[mid:],
                policy=policy,
                no_grad=no_grad,
            )
            return left + right

    def compute_logprob(
        self,
        prompt: str,
        response_token_ids: list[int],
        *,
        policy: PolicyName = "actor",
        no_grad: bool = False,
    ) -> torch.Tensor:
        """Internal helper."""

        return self.compute_logprob_batch(
            [prompt],
            [response_token_ids],
            policy=policy,
            no_grad=no_grad,
        )[0]

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Internal helper."""

        return [p for p in self.actor_model.parameters() if p.requires_grad]

    def save_adapter(self, output_dir: str) -> None:
        """Internal helper."""

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.actor_model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
