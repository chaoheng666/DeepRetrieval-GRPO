from __future__ import annotations

"""模型包装层：统一管理 Actor/Ref 策略与 token 级 logprob 接口。

核心功能：
1. 按配置加载 tokenizer、actor、ref（支持 4bit/全精度/自动回退）
2. 统一生成接口（可选返回 rollout 阶段 old logprob）
3. 对固定 response 重新计算 actor/ref 的 token 级 logprob
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
    """单条采样结果（文本 + token + old logprob）。"""

    response_text: str
    response_token_ids: list[int]
    logprob_old: torch.Tensor
    raw_response_text: str = ""


def _str_to_dtype(dtype_name: str) -> torch.dtype:
    """将字符串精度映射到 torch dtype。"""

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
        """初始化模型组件。

        - train_mode=True 时 actor 以训练模式创建
        - load_ref_model=True 时额外加载冻结 ref 模型用于 KL 项
        """

        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_cfg = model_cfg
        self.prompt_cfg = prompt_cfg
        self.train_mode = train_mode
        self.enable_lora = enable_lora
        self.adapter_path = adapter_path
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

        # 4bit + bf16 在部分运行环境不稳定，这里做安全回退。
        if model_cfg.load_in_4bit and model_cfg.bnb_4bit_compute_dtype.lower() == "bfloat16":
            bf16_supported = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            if not bf16_supported:
                print("[warn] bfloat16 4-bit compute dtype is unsupported on this runtime; fallback to float16.")
                model_cfg.bnb_4bit_compute_dtype = "float16"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_source,
            trust_remote_code=model_cfg.trust_remote_code,
            use_fast=False,
            local_files_only=self.local_model_only,
        )
        # 保证 decoder-only 模型有 pad_token，避免 batch/generate 报错。
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.quantization_config = self._build_4bit_config(enabled=model_cfg.load_in_4bit)
        actor_dtype = (
            _str_to_dtype(model_cfg.bnb_4bit_compute_dtype)
            if model_cfg.load_in_4bit
            else self._preferred_full_precision_dtype()
        )
        # CPU 环境下统一用 float32，避免无意义的半精度设置。
        if not torch.cuda.is_available():
            actor_dtype = torch.float32
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
                # k-bit 训练前的标准准备流程（PEFT 推荐）。
                base_actor = prepare_model_for_kbit_training(base_actor)

            if adapter_path:
                # 从已有 adapter 恢复（续训或评估）。
                self.actor_model = PeftModel.from_pretrained(base_actor, adapter_path, is_trainable=train_mode)
            else:
                # 新建 LoRA adapter。
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
            # ref 模型始终冻结，只作为 KL anchor。
            self.ref_model = self._load_ref_model(AutoModelForCausalLM)
            self.ref_model.eval()
            if hasattr(self.ref_model, "config"):
                self.ref_model.config.use_cache = False
            for param in self.ref_model.parameters():
                param.requires_grad = False

    def _build_4bit_config(self, *, enabled: bool):
        """按需构建 bitsandbytes 4bit 配置。"""

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
        """判断异常是否属于 CUDA OOM。"""

        text = str(exc).lower()
        return "out of memory" in text and ("cuda" in text or "cublas" in text)

    @staticmethod
    def _preferred_full_precision_dtype() -> torch.dtype:
        """全精度优先策略：CUDA 上优先 bf16，其次 fp16。"""

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
        """将 hf_device_map 的值转换为 torch.device。"""

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
        """从模型或其嵌套基类中提取 hf_device_map。"""

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
        """统计模型分片落在哪些设备上（用于日志）。"""

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
        """加载 ref 模型，支持 auto/full/4bit 三种精度模式。

        auto 策略：
        1) 先尝试全精度（bf16/fp16）
        2) 若 CUDA OOM，则自动回退到 4bit（保持 GPU 路径）
        """

        mode = str(getattr(self.model_cfg, "ref_precision_mode", "auto")).strip().lower()
        if mode not in {"auto", "full", "4bit"}:
            raise ValueError(f"Unsupported ref_precision_mode: {self.model_cfg.ref_precision_mode}")

        full_dtype = self._preferred_full_precision_dtype()
        quant_dtype = _str_to_dtype(self.model_cfg.bnb_4bit_compute_dtype)
        load_kwargs = {
            "trust_remote_code": self.model_cfg.trust_remote_code,
            "device_map": self.ref_device_map,
            "local_files_only": self.local_model_only,
        }

        def _load(*, use_4bit: bool, dtype: torch.dtype):
            # use_4bit=True 时传入 4bit 量化配置，否则走全精度加载。
            quant_cfg = self._build_4bit_config(enabled=use_4bit)
            return auto_model_cls.from_pretrained(
                self.model_source,
                quantization_config=quant_cfg,
                torch_dtype=dtype,
                **load_kwargs,
            )

        if mode == "4bit":
            ref_model = _load(use_4bit=True, dtype=quant_dtype)
            self.ref_precision_used = "4bit"
            self.ref_dtype_used = quant_dtype
        elif mode == "full":
            ref_model = _load(use_4bit=False, dtype=full_dtype)
            self.ref_precision_used = "full"
            self.ref_dtype_used = full_dtype
        else:
            try:
                ref_model = _load(use_4bit=False, dtype=full_dtype)
                self.ref_precision_used = "full"
                self.ref_dtype_used = full_dtype
            except RuntimeError as exc:
                if torch.cuda.is_available() and self._is_cuda_oom_error(exc):
                    # 仅在 CUDA OOM 时回退，其他错误继续抛出便于定位。
                    print("[warn] ref full-precision load hit CUDA OOM; fallback to 4-bit on GPU path.")
                    gc.collect()
                    torch.cuda.empty_cache()
                    ref_model = _load(use_4bit=True, dtype=quant_dtype)
                    self.ref_precision_used = "4bit"
                    self.ref_dtype_used = quant_dtype
                else:
                    raise

        print(
            "[ref] loaded: "
            f"precision={self.ref_precision_used}, "
            f"dtype={self.ref_dtype_used}, "
            f"device_map={self._summarize_device_map(ref_model)}"
        )
        return ref_model

    @classmethod
    def _infer_model_device(cls, model: torch.nn.Module) -> torch.device:
        """推断执行设备：优先依据 hf_device_map，其次参数设备。"""

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
    def _normalize_model_id(name: str) -> str:
        """标准化模型 ID（路径分隔符与大小写）。"""

        return name.replace("\\", "/").rstrip("/").lower()

    @classmethod
    def _same_model_id(cls, lhs: str, rhs: str) -> bool:
        """宽松判断两个模型标识是否可视为同一模型。"""

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
        """校验 tokenizer/model 兼容性与 adapter 基模型匹配。"""

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
        """构建重写 prompt，先压缩输入空白字符。"""

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
    ) -> GeneratedSample:
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
            )

        score_steps = list(scores or [])
        steps = min(len(score_steps), len(truncated_ids))
        if steps == 0:
            return GeneratedSample(
                response_text=response_text,
                response_token_ids=truncated_ids,
                logprob_old=torch.empty(0),
                raw_response_text=raw_response_text,
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
        )

    def _policy_model(self, policy: PolicyName) -> torch.nn.Module:
        """按策略名选择 actor 或 ref 模型。"""

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
        """执行生成，并可选返回 rollout 阶段 old logprob。"""

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
        )

    def generate_with_logprob(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> GeneratedSample:
        """从 actor 采样，返回 old-policy token logprob。"""

        return self._generate(
            self.actor_model,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            with_logprob=True,
        )

    def generate_group_with_logprob(
        self,
        prompt: str,
        *,
        num_return_sequences: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[GeneratedSample]:
        """从 actor 一次采样返回一组样本，减少 group 内串行生成开销。"""

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

        # 贪心时多样本意义不大且常需 beam 配置；保持与旧行为一致，逐条生成。
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
        """使用指定策略生成重写文本。"""

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

    def compute_logprob(
        self,
        prompt: str,
        response_token_ids: list[int],
        *,
        policy: PolicyName = "actor",
        no_grad: bool = False,
    ) -> torch.Tensor:
        """对固定 response 重算 token 级 logprob。"""

        if not response_token_ids:
            model = self._policy_model(policy)
            device = self._infer_model_device(model)
            return torch.empty(0, device=device)

        model = self._policy_model(policy)
        device = self._infer_model_device(model)

        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        response_ids = torch.tensor([response_token_ids], dtype=torch.long, device=device)

        if prompt_ids.shape[1] == 0:
            eos_id = self.tokenizer.eos_token_id
            if eos_id is None:
                raise RuntimeError("Tokenizer has no eos token id and prompt is empty.")
            prompt_ids = torch.tensor([[eos_id]], dtype=torch.long, device=device)

        full_input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.ones_like(full_input_ids, device=device)

        grad_ctx = torch.no_grad() if no_grad else nullcontext()
        with grad_ctx:
            logits = model(
                input_ids=full_input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits

        # 因果 LM 对齐：token[t] 由位置 t-1 的 logits 预测。
        prompt_len = int(prompt_ids.shape[1])
        start = max(prompt_len - 1, 0)
        end = start + response_ids.shape[1]
        token_logits = logits[:, start:end, :]
        target_ids = response_ids[:, : token_logits.shape[1]]
        log_probs = torch.log_softmax(token_logits, dim=-1)
        gathered = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)
        return gathered

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """返回可训练参数（通常是 LoRA 参数）。"""

        return [p for p in self.actor_model.parameters() if p.requires_grad]

    def save_adapter(self, output_dir: str) -> None:
        """保存 actor adapter 与 tokenizer。"""

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.actor_model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
