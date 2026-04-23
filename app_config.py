from __future__ import annotations

"""Top20 Delta Curriculum 主线配置中心。

主线只服务 4B top20_delta curriculum 训练：
- actor/ref 默认 4bit；
- 奖励默认围绕 MRR@20、Recall@20、Recall@50；
- prompt 默认使用 top20 lexical rewrite 模板。

注意：这里仍保留少量兼容字段，避免 prompt_eval 等旁路工具 import 失败；
但主入口不会再暴露 legacy/low-mem 行为。
"""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_ARTIFACT_ROOT = "train_and_eval_data_model_0421"
DEFAULT_EXP_NAME = "4b_top20_delta_curriculum"
DEFAULT_RUN_ROOT = f"{DEFAULT_ARTIFACT_ROOT}/artifacts_{DEFAULT_EXP_NAME}"
DEFAULT_TRAIN_DIR = f"{DEFAULT_RUN_ROOT}/phase1"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_PYSERINI_CACHE = str((Path(__file__).resolve().parent.parent / "pyserini_cache"))
_PYSERINI_MIRROR_PATCH_DONE = False

os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
os.environ["PYSERINI_CACHE"] = DEFAULT_PYSERINI_CACHE
Path(os.environ["PYSERINI_CACHE"]).mkdir(parents=True, exist_ok=True)


def _rewrite_hf_host(url: str, endpoint: str) -> str:
    if not isinstance(url, str):
        return url
    for host in ("https://huggingface.co", "http://huggingface.co"):
        if url.startswith(host):
            return endpoint.rstrip("/") + url[len(host) :]
    return url


def patch_pyserini_prebuilt_index_urls() -> None:
    """Patch Pyserini prebuilt-index URLs to respect HF_ENDPOINT."""

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

    for name in (
        "TF_INDEX_INFO",
        "IMPACT_INDEX_INFO",
        "LUCENE_HNSW_INDEX_INFO",
        "LUCENE_FLAT_INDEX_INFO",
        "FAISS_INDEX_INFO",
    ):
        info = getattr(pinfo, name, None)
        if not isinstance(info, dict):
            continue
        for meta in info.values():
            urls = meta.get("urls")
            if isinstance(urls, list):
                meta["urls"] = [_rewrite_hf_host(url, endpoint) for url in urls]

    _PYSERINI_MIRROR_PATCH_DONE = True


@dataclass(slots=True)
class DataConfig:
    # 数据源和切分策略。主线使用 MS MARCO passage dev subset + Pyserini 预建 BM25 索引。
    topic_name: str = "msmarco-passage-dev-subset"
    prebuilt_index: str = "msmarco-v1-passage"
    train_ratio: float = 0.8
    seed: int = 42
    max_train_queries: int | None = 20000
    max_val_queries: int | None = 400


@dataclass(slots=True)
class ModelConfig:
    # 4B 基座模型和 LoRA/4bit 加载配置；训练时 actor/ref 都按 4bit 路径加载。
    model_name: str = "/root/autodl-tmp/hf_models/Qwen3-4B-Instruct-2507"
    trust_remote_code: bool = True
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    projection_chunk_size: int = 64
    actor_device_map: str = "auto"
    ref_device_map: str = "auto"
    # 兼容字段。主线会在 CUDA 4bit 可用时固定按 4bit 加载 ref。
    ref_precision_mode: str = "4bit"
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
    # GRPO 训练超参。phase1/phase2 的差异主要由目标 bash 入口覆盖。
    num_epochs: int = 1
    batch_size: int = 24
    group_size: int = 8
    max_group_size: int = 12
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.0
    clip_range: float = 0.2
    kl_beta: float = 0.04
    grad_clip_norm: float = 1.0
    max_new_tokens: int = 10
    temperature: float = 0.82
    top_p: float = 0.93
    group_temperature_stride: float = 0.08
    group_top_p_stride: float = 0.02
    # 采样多样性控制：组内候选过于重复时，会提高温度补采样。
    min_unique_final_queries: int = 5
    max_regen_rounds: int = 3
    regen_temperature_delta: float = 0.15
    reward_gap_threshold: float = 0.12
    gap_sampling_temperature_delta: float = 0.18
    actor_chunk_size: int = 2
    eval_every_steps: int = 20
    max_steps: int | None = 100
    save_dir: str = f"{DEFAULT_TRAIN_DIR}/checkpoints"
    log_path: str = f"{DEFAULT_TRAIN_DIR}/train_log.jsonl"
    group_trace_log_path: str = f"{DEFAULT_TRAIN_DIR}/group_trace_log.jsonl"
    # curriculum metadata 记录每个 query 的原始检索难度，用来按 phase 抽样。
    curriculum_enable: bool = True
    curriculum_phase: str = "phase1"
    curriculum_metadata_path: str | None = f"{DEFAULT_RUN_ROOT}/curriculum_query_metadata.jsonl"
    early_stop_patience: int = 2
    early_stop_degrade_threshold: float = 0.01
    early_stop_degrade_patience: int = 2
    early_stop_warmup_evals: int = 3


@dataclass(slots=True)
class RewardConfig:
    # 兼容字段，方便旁路报告展示；实际奖励公式固定为 top20_delta。
    reward_mode: str = "top20_delta"
    # 主奖励观测窗口：MRR@20 + Recall@20 + Recall@50。
    mrr_k: int = 20
    recall_k: int = 20
    recall_dense_k: int = 50
    search_threads: int = 16
    w_mrr: float = 0.52
    w_recall: float = 0.22
    w_recall_dense: float = 0.16
    w_rank_bonus: float = 0.10
    w_bad_format: float = 0.18
    w_unsafe_copy: float = 0.14
    w_overedit: float = 0.08
    # 保护语义不被过度改写，同时对 recall 下降给强惩罚。
    overedit_tau: float = 0.45
    recall_drop_lambda: float = 0.80
    anchor_bonus_value: float = 0.05
    format_max_tokens: int = 12
    format_min_english_ratio: float = 0.85
    format_max_unreadable_ratio: float = 0.20
    bad_format_cap: float = 1.5

    # 旁路兼容字段；主线 reward composition 不再使用这些 legacy 项。
    w_term_preserve: float = 0.0
    w_length_score: float = 0.0
    w_clean_format: float = 0.0
    length_score_min_terms: int = 1
    length_score_ideal_min_terms: int = 4
    length_score_ideal_max_terms: int = 12
    length_score_max_terms: int = 20


@dataclass(slots=True)
class PromptConfig:
    # top20 prompt 要求模型只输出“一行检索 query”，不能输出解释或思考过程。
    prompt_id: str = "p24_diverse_lexical_top20"
    system_prompt: str = (
        "You rewrite search queries for DeepRetrieval-GRPO.\n"
        "The retriever is Lucene BM25 over MS MARCO passages.\n"
        "Your only goal is to improve top-20 sparse lexical retrieval over the original query.\n"
        "Optimize primarily for MRR@20, with supporting focus on Recall@20 and Recall@50.\n"
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
        "- Prefer exact terms likely to appear verbatim in top-ranked relevant passages.\n"
        "- Favor edits that can surface a relevant document in the top 20 while also improving Recall@20 and Recall@50.\n"
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
    max_new_tokens: int = 10
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
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def get_default_config() -> AppConfig:
    return AppConfig()


def apply_reward_mode_prompt_defaults(config: AppConfig) -> AppConfig:
    """旧调用方兼容层。

    默认已经是 top20 prompt；如果旁路还带着旧 prompt_id 进来，
    这里只做一次轻量升级，避免破坏 import 稳定性。
    """

    if config.reward.reward_mode == "top20_delta" and config.prompt.prompt_id == "p24_diverse_lexical":
        config.prompt.prompt_id = "p24_diverse_lexical_top20"
        config.prompt.system_prompt = PromptConfig().system_prompt
    return config


def ensure_runtime_dirs(config: AppConfig) -> None:
    Path(config.train.save_dir).mkdir(parents=True, exist_ok=True)
    Path(config.train.log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.train.group_trace_log_path).parent.mkdir(parents=True, exist_ok=True)
