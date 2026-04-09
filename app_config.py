from __future__ import annotations

"""全局配置模块。

本文件只负责一件事：集中管理训练/评测所需的全部可调参数，并给出一套
“快速验证可跑通”的默认值。这样做有几个好处：

1. 训练脚本与评测脚本不需要散落硬编码常量，便于排错和复现实验。
2. 新机器迁移时，可以先用默认配置跑通，再逐步覆盖参数做正式实验。
3. CLI 覆盖逻辑简单明了：命令行仅覆盖非空字段，避免意外改动。
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DataConfig:
    """数据与检索后端配置。

    关键约束：数据与检索都必须直接走 Pyserini 预编译接口。
    """

    # Pyserini 主题集名称，会传给 get_topics/get_qrels。
    topic_name: str = "msmarco-passage-dev-subset"
    # Pyserini 预编译 Lucene 索引名，会传给 from_prebuilt_index。
    prebuilt_index: str = "msmarco-v1-passage"
    # 训练/验证切分比例（固定随机种子保证可复现）。
    train_ratio: float = 0.8
    # 数据切分与训练乱序的随机种子。
    seed: int = 42
    # 可选上限：用于快速验证，减少训练与验证样本数。
    max_train_queries: int | None = 2000
    max_val_queries: int | None = 400


@dataclass(slots=True)
class ModelConfig:
    """模型加载与 LoRA/QLoRA 配置。"""

    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    trust_remote_code: bool = True
    # 是否启用 4-bit 量化（QLoRA 关键开关）。
    # 默认开启，目标是让 24G 显存可运行。
    load_in_4bit: bool = True
    # bitsandbytes 的量化细节参数：
    # - nf4：常用且稳定的 4-bit 量化格式
    # - compute_dtype：前向/反向计算精度
    # - double_quant：进一步压缩权重存储
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    actor_device_map: str = "auto"
    ref_device_map: str = "auto"
    # Ref 精度模式：
    # - auto：优先全精度（bf16/fp16），若 CUDA OOM 自动回退 4bit
    # - full：始终全精度
    # - 4bit：始终 4bit 量化
    ref_precision_mode: str = "auto"
    # LoRA 超参数：
    # r/alpha/dropout 与 target_modules 共同决定可训练低秩适配器的容量。
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass(slots=True)
class TrainConfig:
    """GRPO 训练超参数。"""

    num_epochs: int = 1
    batch_size: int = 2
    # GRPO 的 K：每条 query 采样的重写候选数。
    group_size: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    # PPO clip 参数 epsilon。
    clip_range: float = 0.2
    # KL 惩罚系数 beta，用于约束新策略不要偏离参考策略过远。
    kl_beta: float = 0.02
    # 梯度裁剪阈值，避免梯度爆炸导致训练不稳定。
    grad_clip_norm: float = 1.0
    # 生成长度与采样策略。
    max_new_tokens: int = 24
    temperature: float = 1.0
    top_p: float = 0.95
    # 每隔多少个 step 在验证集上评估一次。
    eval_every_steps: int = 20
    # 可选总步数上限（用于快速调试）。
    max_steps: int | None = None
    # checkpoint 与训练日志输出位置。
    save_dir: str = "artifacts/checkpoints"
    log_path: str = "artifacts/train_log.jsonl"
    # 每个 query 一条 group 采样明细日志（每个训练 step 追加多行）。
    group_trace_log_path: str = "artifacts/group_trace_log.jsonl"


@dataclass(slots=True)
class RewardConfig:
    """奖励函数配置：检索奖励 + 文本惩罚。"""

    # 检索截断深度：计算 MRR@topk。
    topk: int = 50
    # 奖励组合权重：
    # total = mrr_weight * mrr + overlap_weight * lexical_overlap - penalty
    mrr_weight: float = 1.0
    overlap_weight: float = 0.2
    # 生成 query 过短时的阈值与惩罚。
    min_query_chars: int = 3
    # 重复比例与不可读字符比例阈值（用于轻量文本质量约束）。
    max_repeat_ratio: float = 0.35
    max_unreadable_char_ratio: float = 0.30
    # 各项惩罚强度。
    penalty_short: float = 0.20
    penalty_repeat: float = 0.20
    penalty_unreadable: float = 0.30


@dataclass(slots=True)
class PromptConfig:
    """查询重写提示词模板。"""

    system_prompt: str = (
        "You are a high-precision query rewriter for sparse lexical retrieval (BM25-style). "
        "Rewrite the user query into exactly one English search query line for passage retrieval. "
        "\n"
        "Hard output constraints:\n"
        "1) Output English only. Do not output Chinese, Cyrillic, mixed-script text, emoji, or symbols from other scripts.\n"
        "2) Output exactly one line with only the final query text. No prefix, no numbering, no quotes, no markdown.\n"
        "3) Do not output explanations, reasoning, alternatives, or multiple candidates.\n"
        "\n"
        "Retrieval-oriented rewriting rules:\n"
        "1) Preserve the original intent and constraints exactly; do not change the task.\n"
        "2) Preserve key entities, names, product terms, numbers, years, units, negations, and domain constraints.\n"
        "3) If the user query is not in English, translate it to natural English retrieval terms while preserving meaning.\n"
        "4) Prefer concrete lexical terms likely to match documents: canonical entity names, aliases, and discriminative nouns.\n"
        "5) Remove filler words and conversational phrasing; keep it concise and specific.\n"
        "6) Keep explicit comparison or temporal intent when present (for example: vs, before, after, latest, 2021).\n"
        "7) Keep the query as a compact keyword-rich phrase, not a full explanatory sentence.\n"
        "\n"
        "Safety/quality guardrails:\n"
        "1) Do not invent facts, entities, dates, or constraints that were not in the user query.\n"
        "2) If the input is already a strong English retrieval query, keep it very close with only minimal normalization.\n"
        "3) Avoid malformed or repetitive tokens."
    )
    template: str = "User Query: {query}\nSearch Query:"


@dataclass(slots=True)
class AppConfig:
    """顶层配置对象，训练/评测脚本统一通过它访问参数。"""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)

    def to_dict(self) -> dict:
        """序列化为普通字典，便于打印与落盘日志。"""
        return asdict(self)


def get_default_config() -> AppConfig:
    """返回默认配置（每次调用都会生成新的实例）。"""

    return AppConfig()


def ensure_runtime_dirs(config: AppConfig) -> None:
    """确保运行目录存在。

    训练入口会依赖以下目录：
    - checkpoint 保存目录
    - 日志文件父目录
    """

    save_dir = Path(config.train.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(config.train.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    group_trace_log_path = Path(config.train.group_trace_log_path)
    group_trace_log_path.parent.mkdir(parents=True, exist_ok=True)
