from __future__ import annotations

"""模型包装层（Actor/Ref 策略 + token 级 logprob 接口）。

本模块承担四类职责：

1. 加载基础模型（支持 4-bit 量化，适配 QLoRA 显存约束）。
2. 构建 Actor 策略（带 LoRA，可训练）。
3. 构建 Ref 策略（冻结，仅用于 KL 正则）。
4. 提供两个关键能力：
   - 采样时返回 logprob_old（PPO ratio 的旧策略项）
   - 对固定 response 重算 logprob（new/ref 策略对齐比较）
"""

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from app_config import ModelConfig, PromptConfig

PolicyName = Literal["actor", "ref"]


@dataclass(frozen=True, slots=True)
class GeneratedSample:
    """采样输出结构。"""

    response_text: str
    response_token_ids: list[int]
    logprob_old: torch.Tensor


def _str_to_dtype(dtype_name: str) -> torch.dtype:
    """把配置里的字符串精度映射到 torch dtype。"""

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
    ) -> None:
        """初始化 tokenizer、actor 模型，以及可选的 ref 模型。

        参数说明：
        - train_mode:
          True 时 actor 进入训练态，LoRA 参数可更新；False 时用于推理/评估。
        - enable_lora:
          True 时构建 LoRA actor；False 时使用纯基础模型（用于 zero-shot baseline）。
        - load_ref_model:
          True 时额外加载一个冻结参考模型，供 KL 惩罚使用。
        - adapter_path:
          若提供则从该目录加载已训练 LoRA adapter；否则新建 adapter。
        """

        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model_cfg = model_cfg
        self.prompt_cfg = prompt_cfg
        self.train_mode = train_mode
        self.enable_lora = enable_lora
        self.adapter_path = adapter_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.model_name,
            trust_remote_code=model_cfg.trust_remote_code,
            use_fast=False,
        )
        if self.tokenizer.pad_token is None:
            # 许多 decoder-only 模型没有显式 pad_token。
            # 复用 eos_token 可避免 generate/batch 输入报错。
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.quantization_config = None
        if model_cfg.load_in_4bit:
            # QLoRA 路径：显著降低显存占用，适合单卡训练。
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=model_cfg.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=_str_to_dtype(model_cfg.bnb_4bit_compute_dtype),
            )

        base_actor = AutoModelForCausalLM.from_pretrained(
            model_cfg.model_name,
            trust_remote_code=model_cfg.trust_remote_code,
            quantization_config=self.quantization_config,
            device_map=model_cfg.actor_device_map,
            torch_dtype=_str_to_dtype(model_cfg.bnb_4bit_compute_dtype),
        )

        if enable_lora:
            if train_mode:
                # k-bit 训练前的标准准备步骤（PEFT 建议）。
                base_actor = prepare_model_for_kbit_training(base_actor)

            if adapter_path:
                # 从已有 adapter 恢复（继续训练或评估）。
                self.actor_model = PeftModel.from_pretrained(base_actor, adapter_path, is_trainable=train_mode)
            else:
                # 新建 LoRA adapter 并挂载到基础模型。
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

        self.ref_model = None
        if load_ref_model:
            # 参考策略固定不训练，作为 KL anchor 分布。
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                model_cfg.model_name,
                trust_remote_code=model_cfg.trust_remote_code,
                quantization_config=self.quantization_config,
                device_map=model_cfg.ref_device_map,
                torch_dtype=_str_to_dtype(model_cfg.bnb_4bit_compute_dtype),
            )
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

    @staticmethod
    def _infer_model_device(model: torch.nn.Module) -> torch.device:
        """从参数推断模型所在设备。"""

        return next(model.parameters()).device

    def build_prompt(self, query: str) -> str:
        """构建查询重写 prompt。

        这里会先做空白字符压缩，减少输入格式噪声，保证提示模板稳定。
        """

        query_clean = " ".join(query.strip().split())
        return f"{self.prompt_cfg.system_prompt}\n\n{self.prompt_cfg.template.format(query=query_clean)}"

    def _policy_model(self, policy: PolicyName) -> torch.nn.Module:
        """根据策略名选择 actor 或 ref 模型。"""

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
        """执行文本生成，并可选返回采样时 token 级 logprob_old。

        关键说明：
        - with_logprob=True 时，会请求 generate 返回每步 logits（output_scores）。
        - 这些 logits 对应“采样当下”的策略概率，是 PPO 中 old policy 的来源。
        - 后续训练时会再重算 logprob_new，与 logprob_old 组成 ratio。
        """

        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = self._infer_model_device(model)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 温度为 0 时按贪心解码处理，避免采样随机性。
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
            # scores[i] 是第 i 步输出 token 的 logits，取 softmax 后再索引采样 token。
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
        """从 actor 采样，并返回采样时 old logprob。"""

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
        """使用指定策略生成重写 query 文本。"""

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
        """对“固定 response token 序列”重算 token 级 logprob。

        用途：
        - actor + no_grad=False：得到可反传的 logprob_new
        - ref + no_grad=True：得到冻结参考概率 logprob_ref

        计算细节：
        - 将 prompt_ids 与 response_ids 拼接后送入因果 LM。
        - 利用 shift 对齐规则，从 logits 中截取 response 对应位置。
        - 对每个目标 token 收集对应 log softmax 值。
        """

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
            logits = model(input_ids=full_input_ids, attention_mask=attention_mask).logits

        prompt_len = int(prompt_ids.shape[1])
        # 因果语言模型的对齐关系：token[t] 由位置 t-1 的 logits 预测。
        start = max(prompt_len - 1, 0)
        end = start + response_ids.shape[1]
        token_logits = logits[:, start:end, :]
        target_ids = response_ids[:, : token_logits.shape[1]]
        log_probs = torch.log_softmax(token_logits, dim=-1)
        gathered = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)
        return gathered

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """返回可训练参数列表（实际通常为 LoRA 参数）。"""

        return [p for p in self.actor_model.parameters() if p.requires_grad]

    def save_adapter(self, output_dir: str) -> None:
        """保存 actor adapter 与 tokenizer，供后续评估/恢复训练使用。"""

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.actor_model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
