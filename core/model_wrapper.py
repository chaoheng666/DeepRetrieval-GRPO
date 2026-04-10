from __future__ import annotations

"""模型包装层：统一管理 Actor/Ref 策略与 token 级 logprob 接口。

核心功能：
1. 按配置加载 tokenizer、actor、ref（支持 4bit/全精度/自动回退）
2. 统一生成接口（可选返回 rollout 阶段 old logprob）
3. 对固定 response 重新计算 actor/ref 的 token 级 logprob
"""

from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import torch

from app_config import ModelConfig, PromptConfig

PolicyName = Literal["actor", "ref"]


@dataclass(frozen=True, slots=True)
class GeneratedSample:
    """单条采样结果（文本 + token + old logprob）。"""

    response_text: str
    response_token_ids: list[int]
    logprob_old: torch.Tensor


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

        # 4bit + bf16 在部分运行环境不稳定，这里做安全回退。
        if model_cfg.load_in_4bit and model_cfg.bnb_4bit_compute_dtype.lower() == "bfloat16":
            bf16_supported = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            if not bf16_supported:
                print("[warn] bfloat16 4-bit compute dtype is unsupported on this runtime; fallback to float16.")
                model_cfg.bnb_4bit_compute_dtype = "float16"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.model_name,
            trust_remote_code=model_cfg.trust_remote_code,
            use_fast=False,
        )
        # 保证 decoder-only 模型有 pad_token，避免 batch/generate 报错。
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.quantization_config = self._build_4bit_config(enabled=model_cfg.load_in_4bit)
        actor_dtype = _str_to_dtype(model_cfg.bnb_4bit_compute_dtype)
        # CPU 环境下统一用 float32，避免无意义的半精度设置。
        if not torch.cuda.is_available():
            actor_dtype = torch.float32
        base_actor = AutoModelForCausalLM.from_pretrained(
            model_cfg.model_name,
            trust_remote_code=model_cfg.trust_remote_code,
            quantization_config=self.quantization_config,
            device_map=model_cfg.actor_device_map,
            dtype=actor_dtype,
            cache_dir="D:/hf_cache",  
        )
        self._validate_tokenizer_model_match(
            tokenizer=self.tokenizer,
            model=base_actor,
            model_name=model_cfg.model_name,
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
            "device_map": self.model_cfg.ref_device_map,
        }

        def _load(*, use_4bit: bool, dtype: torch.dtype):
            # use_4bit=True 时传入 4bit 量化配置，否则走全精度加载。
            quant_cfg = self._build_4bit_config(enabled=use_4bit)
            return auto_model_cls.from_pretrained(
                self.model_cfg.model_name,
                quantization_config=quant_cfg,
                dtype=dtype,
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

        # 温度 <= 0 时按贪心解码处理。
        do_sample = temperature > 0.0
        kwargs = {
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

        with torch.no_grad():
            output = model.generate(**inputs, **kwargs)

        sequence = output.sequences[0]
        prompt_len = int(inputs["input_ids"].shape[1])
        response_ids = sequence[prompt_len:].tolist()
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        if not with_logprob or not response_ids:
            return GeneratedSample(response_text=response_text, response_token_ids=response_ids, logprob_old=torch.empty(0))

        scores = output.scores or []
        steps = min(len(scores), len(response_ids))
        if steps == 0:
            return GeneratedSample(response_text=response_text, response_token_ids=response_ids, logprob_old=torch.empty(0))

        token_logprobs: list[torch.Tensor] = []
        for i in range(steps):
            logits_step = scores[i][0].float()
            logprob_step = torch.log_softmax(logits_step, dim=-1)[response_ids[i]]
            token_logprobs.append(logprob_step.detach().cpu())

        return GeneratedSample(
            response_text=response_text,
            response_token_ids=response_ids[:steps],
            logprob_old=torch.stack(token_logprobs).to(torch.float32),
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
