from __future__ import annotations

"""全局配置模块。

本文件只负责一件事：集中管理训练/评测所需的全部可调参数，并给出一套
“快速验证可跑通”的默认值。这样做有几个好处：

1. 训练脚本与评测脚本不需要散落硬编码常量，便于排错和复现实验。
2. 新机器迁移时，可以先用默认配置跑通，再逐步覆盖参数做正式实验。
3. CLI 覆盖逻辑简单明了：命令行仅覆盖非空字段，避免意外改动。
"""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_ARTIFACT_ROOT = "train_and_eval_data_model"
DEFAULT_EXP_NAME = "default"
DEFAULT_TRAIN_DIR = f"{DEFAULT_ARTIFACT_ROOT}/artifacts_{DEFAULT_EXP_NAME}_train"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_PYSERINI_CACHE = str((Path(__file__).resolve().parent.parent / "pyserini_cache"))
_PYSERINI_MIRROR_PATCH_DONE = False

# Keep user/exported HF_ENDPOINT untouched; only apply fallback when unset.
os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
# Pin Pyserini cache to a stable path beside this project (project parent / pyserini_cache).
os.environ["PYSERINI_CACHE"] = DEFAULT_PYSERINI_CACHE
Path(os.environ["PYSERINI_CACHE"]).mkdir(parents=True, exist_ok=True)


def _rewrite_hf_host(url: str, endpoint: str) -> str:
    """Rewrite Hugging Face host to the configured mirror endpoint."""

    if not isinstance(url, str):
        return url
    for host in ("https://huggingface.co", "http://huggingface.co"):
        if url.startswith(host):
            return endpoint.rstrip("/") + url[len(host) :]
    return url


def patch_pyserini_prebuilt_index_urls() -> None:
    """Patch Pyserini prebuilt index URLs so mirror endpoint is respected."""

    global _PYSERINI_MIRROR_PATCH_DONE
    if _PYSERINI_MIRROR_PATCH_DONE:
        return

    endpoint = os.environ.get("HF_ENDPOINT", "").strip()
    if not endpoint:
        return

    try:
        from pyserini import prebuilt_index_info as pinfo
    except Exception:
        return

    info_dict_names = (
        "TF_INDEX_INFO",
        "IMPACT_INDEX_INFO",
        "LUCENE_HNSW_INDEX_INFO",
        "LUCENE_FLAT_INDEX_INFO",
        "FAISS_INDEX_INFO",
    )
    for name in info_dict_names:
        info = getattr(pinfo, name, None)
        if not isinstance(info, dict):
            continue
        for meta in info.values():
            urls = meta.get("urls")
            if not isinstance(urls, list):
                continue
            meta["urls"] = [_rewrite_hf_host(u, endpoint) for u in urls]

    _PYSERINI_MIRROR_PATCH_DONE = True



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
    max_train_queries: int | None = 20000
    max_val_queries: int | None = 4000


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
    batch_size: int = 8
    # Initial group size for each query before adaptive gap-driven resampling.
    group_size: int = 10
    # Hard cap for adaptive gap-driven resampling.
    max_group_size: int = 14
    learning_rate: float = 1.5e-5
    weight_decay: float = 0.0
    # PPO clip 参数 epsilon。
    clip_range: float = 0.2
    # KL 惩罚系数 beta，用于约束新策略不要偏离参考策略过远。
    kl_beta: float = 0.03
    # 梯度裁剪阈值，避免梯度爆炸导致训练不稳定。
    grad_clip_norm: float = 1.0
    # 生成长度与采样策略。
    max_new_tokens: int = 12
    temperature: float = 0.85
    top_p: float = 0.95
    # Optionally diversify grouped sampling by slightly varying decode params
    # across samples within the same GRPO group.
    group_temperature_stride: float = 0.07
    group_top_p_stride: float = 0.015
    # When a sampled group collapses to too few distinct final queries, retry
    # duplicate/polluted slots with a slightly warmer decode.
    min_unique_final_queries: int = 4
    max_regen_rounds: int = 2
    regen_temperature_delta: float = 0.15
    # Keep sampling until best-minus-worst raw reward reaches this spread,
    # or until max_group_size is reached.
    reward_gap_threshold: float = 0.08
    # Only adaptive extra samples use a higher temperature.
    gap_sampling_temperature_delta: float = 0.15
    # 每隔多少个 step 在验证集上评估一次。
    eval_every_steps: int = 20
    # 可选总步数上限（用于快速调试）。
    max_steps: int | None = None
    # checkpoint 与训练日志输出位置。
    save_dir: str = f"{DEFAULT_TRAIN_DIR}/checkpoints"
    log_path: str = f"{DEFAULT_TRAIN_DIR}/train_log.jsonl"
    # 每个 query 一条 group 采样明细日志（每个训练 step 追加多行）。
    group_trace_log_path: str = f"{DEFAULT_TRAIN_DIR}/group_trace_log.jsonl"


@dataclass(slots=True)
class RewardConfig:
    """Dense BM25 rewrite reward configuration."""

    # Retrieval cutoffs.
    mrr_k: int = 10
    recall_k: int = 50
    recall_dense_k: int = 100
    # Pyserini batch_search thread count for retrieval-side parallelism.
    search_threads: int = 8
    # Fixed reward formula:
    # total = 0.40*mrr@10 + 0.20*recall@50 + 0.15*recall@100
    #       + 0.10*term_preserve + 0.08*length_score + 0.07*clean_format
    #       - 0.15*bad_format - 0.08*unsafe_copy
    w_mrr: float = 0.40
    w_recall: float = 0.20
    w_recall_dense: float = 0.15
    w_term_preserve: float = 0.10
    w_length_score: float = 0.08
    w_clean_format: float = 0.07
    w_bad_format: float = 0.15
    w_unsafe_copy: float = 0.08
    # Length score piecewise anchors.
    length_score_min_terms: int = 1
    length_score_ideal_min_terms: int = 4
    length_score_ideal_max_terms: int = 12
    length_score_max_terms: int = 20
    # Continuous bad-format settings.
    format_max_tokens: int = 16
    format_min_english_ratio: float = 0.80
    format_max_unreadable_ratio: float = 0.30
    bad_format_cap: float = 1.5


@dataclass(slots=True)
class PromptConfig:
    """Default prompt + deterministic decode config for inference/evaluation."""

    prompt_id: str = "p24_diverse_lexical"
    system_prompt: str = (
        "You rewrite search queries for DeepRetrieval-GRPO.\n"
        "The retriever is Lucene BM25 over MS MARCO passages.\n"
        "Your only goal is to improve sparse lexical retrieval MRR@10 over the original query.\n"
        "\n"
        "Hard output contract:\n"
        "1) Output exactly one line of English query text.\n"
        "2) Output only the final query: no explanation, no answer, no labels, no XML, no markdown.\n"
        "3) Never emit <think>, multiple options, bullet points, or reasoning traces.\n"
        "\n"
        "Strategy ID: P23 [FewShot]\n"
        "Strategy objective: Favor lexical forms that are common in explanatory passages rather than conversational wording.\n"
        "\n"
        "BM25 rules:\n"
        "- Preferred length: 3-11 meaningful terms.\n"
        "- Preserve named entities, rare technical terms, acronyms, numbers, years, versions, units, and negations.\n"
        "- If the original query is already concise and retrieval-ready, keep it close but prefer a small lexical improvement over an exact copy when a safe variant exists.\n"
        "- Prefer exact terms likely to appear verbatim in relevant passages.\n"
        "- Remove chatty wrappers and helper verbs when safe.\n"
        "- Avoid speculative synonyms, broadening, and answer-style prose.\n"
        "- Match the demonstration style exactly and emit only the live rewrite.\n"
        "- Stop immediately after the rewrite; never continue with another "
        "\"User query\" or \"Better BM25 query\" block.\n"
        "- Strategy-specific rules:\n"
        "  - Prefer content nouns and modifiers that are likely to appear in passage text.\n"
        "  - Avoid answer-style sentences and keep the query keyword-like.\n"
        "  - Avoid copying the source query verbatim when a nearby lexical variant is equally safe.\n"
        "  - Small reordering, clarification, or insertion of one high-value lexical term is better than no rewrite."
    )
    template: str = (
        "Example\n"
        "User query: what are symptoms of anemia in women\n"
        "Better BM25 query: anemia symptoms women\n"
        "\n"
        "User query: windows media player amr files\n"
        "Better BM25 query: windows media player play amr files\n"
        "\n"
        "User query: {query}\n"
        "Better BM25 query:"
    )
    max_new_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    stop_on: str | None = "\n"
    stop_strings: tuple[str, ...] = (
        "\n",
        "\nUser query:",
        "\nBetter BM25 query:",
        "\nSearch query:",
        "\nRewritten query:",
        "\nExample",
    )
    enforce_single_line: bool = True
    min_terms: int = 3
    max_terms: int = 11
    fallback_mode: str = "balanced"


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
