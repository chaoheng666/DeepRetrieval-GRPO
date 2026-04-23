import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs"
OUT.mkdir(exist_ok=True)

STAGES = [
    {
        "name": "Stage 1",
        "dir": ROOT / "阶段1",
        "train": "train_log (6).jsonl",
        "trace": "group_trace_log (4).jsonl",
        "bg": "#D9ECFF",
        "color": "#1F77B4",
    },
    {
        "name": "Stage 2",
        "dir": ROOT / "阶段2",
        "train": "train_log (7).jsonl",
        "trace": "group_trace_log (5).jsonl",
        "bg": "#FFE8CC",
        "color": "#D55E00",
    },
]


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def pct(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x * 100:.1f}%"


def num(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.4f}"


def add_stage_background(ax, spans, y_text=0.965):
    for span in spans:
        ax.axvspan(span["start"], span["end"], color=span["bg"], alpha=0.45, zorder=0)
        ax.text(
            (span["start"] + span["end"]) / 2,
            y_text,
            span["name"],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color="#333333",
            bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none", "pad": 2},
        )
    if len(spans) > 1:
        for span in spans[1:]:
            ax.axvline(span["start"], color="#555555", lw=1.0, ls="--", alpha=0.8)


def rolling(series: pd.Series, window=5) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def unique_ratio(values) -> float:
    if not isinstance(values, list) or not values:
        return np.nan
    return len(set(values)) / len(values)


def safe_mean(values) -> float:
    if not isinstance(values, list) or not values:
        return np.nan
    return float(np.mean(values))


def load_data():
    train_parts = []
    trace_parts = []
    spans = []
    offset = 0
    for stage_idx, stage in enumerate(STAGES, start=1):
        train = read_jsonl(stage["dir"] / stage["train"])
        trace = read_jsonl(stage["dir"] / stage["trace"])

        train["stage"] = stage["name"]
        train["stage_idx"] = stage_idx
        train["stage_step"] = np.arange(1, len(train) + 1)
        train["global_step"] = train["stage_step"] + offset
        train["time"] = pd.to_datetime(train["timestamp_utc"], utc=True, errors="coerce")

        step_map = train[["epoch", "step", "timestamp_utc", "global_step"]].drop_duplicates()
        trace = trace.merge(step_map, on=["epoch", "step", "timestamp_utc"], how="left")
        trace["stage"] = stage["name"]
        trace["stage_idx"] = stage_idx
        trace["time"] = pd.to_datetime(trace["timestamp_utc"], utc=True, errors="coerce")
        trace["unique_final_query_count"] = trace["group_final_queries"].map(
            lambda x: len(set(x)) if isinstance(x, list) else np.nan
        )
        trace["unique_final_query_ratio"] = trace["group_final_queries"].map(unique_ratio)
        trace["group_reward_mean"] = trace["group_rewards"].map(safe_mean)
        trace["group_mrr_mean"] = trace["group_mrr"].map(safe_mean)
        trace["group_recall_mean"] = trace["group_recall"].map(safe_mean)
        trace["group_delta_mrr_mean"] = trace["group_delta_mrr"].map(safe_mean)
        trace["group_delta_recall_mean"] = trace["group_delta_recall"].map(safe_mean)
        trace["group_keyword_preserve_mean"] = trace["group_keyword_preserve"].map(safe_mean)

        train_parts.append(train)
        trace_parts.append(trace)
        spans.append(
            {
                "name": stage["name"],
                "start": offset + 1,
                "end": offset + len(train),
                "bg": stage["bg"],
                "color": stage["color"],
            }
        )
        offset += len(train)
    return pd.concat(train_parts, ignore_index=True), pd.concat(trace_parts, ignore_index=True), spans


def summarize_train(train: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "reward_mean",
        "main_reward_mean",
        "mrr_mean",
        "recall_mean",
        "recall_dense_mean",
        "delta_mrr20_mean",
        "delta_recall20_mean",
        "delta_recall50_mean",
        "keyword_preserve_mean",
        "recall_drop_ratio",
        "trainable_group_ratio",
        "flat_reward_group_ratio",
        "extra_sample_ratio",
        "loss",
        "loss_kl",
        "kl_dominance_ratio",
    ]
    rows = []
    for stage, df in train.groupby("stage", sort=False):
        for metric in metrics:
            if metric not in df:
                continue
            rows.append(
                {
                    "stage": stage,
                    "metric": metric,
                    "first": df[metric].iloc[0],
                    "last": df[metric].iloc[-1],
                    "mean": df[metric].mean(),
                    "std": df[metric].std(),
                    "last_10_mean": df[metric].tail(min(10, len(df))).mean(),
                    "min": df[metric].min(),
                    "max": df[metric].max(),
                }
            )
    return pd.DataFrame(rows)


def summarize_trace(trace: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage, df in trace.groupby("stage", sort=False):
        stop_reasons = df["reward_gap_stop_reason"].value_counts(normalize=True).to_dict()
        rows.append(
            {
                "stage": stage,
                "groups": len(df),
                "reward_gap_met_rate": df["reward_gap_met"].mean(),
                "collapsed_group_rate": df["collapsed_group"].mean(),
                "mean_generated_sample_count": df["generated_sample_count"].mean(),
                "mean_extra_sample_count": df["extra_sample_count"].mean(),
                "mean_gap_sampling_rounds": df["gap_sampling_rounds"].mean(),
                "mean_unique_final_query_count": df["unique_final_query_count"].mean(),
                "mean_unique_final_query_ratio": df["unique_final_query_ratio"].mean(),
                "mean_reward_gap_raw": df["reward_gap_raw"].mean(),
                "median_reward_gap_raw": df["reward_gap_raw"].median(),
                "mean_group_reward": df["group_reward_mean"].mean(),
                "mean_group_mrr": df["group_mrr_mean"].mean(),
                "mean_group_recall": df["group_recall_mean"].mean(),
                "mean_group_delta_mrr": df["group_delta_mrr_mean"].mean(),
                "mean_group_delta_recall": df["group_delta_recall_mean"].mean(),
                "mean_group_keyword_preserve": df["group_keyword_preserve_mean"].mean(),
                "stop_threshold_reached_rate": stop_reasons.get("threshold_reached", 0.0),
                "stop_max_group_size_reached_rate": stop_reasons.get("max_group_size_reached", 0.0),
                "stop_collapsed_group_rate": stop_reasons.get("collapsed_group", 0.0),
            }
        )
    return pd.DataFrame(rows)


def summarize_eval() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    report_path = ROOT / "eval_compare_report_full (1).json"
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    eval_rows = []
    for name, section in [("Original", "original"), ("Zero-shot", "zero_shot"), ("RL", "rl")]:
        data = report[section]
        eval_rows.append(
            {
                "system": name,
                "mrr_mean": data.get("mrr_mean"),
                "recall20_mean": data.get("recall_mean"),
                "recall50_mean": data.get("recall_dense_mean"),
                "reward_mean": data.get("reward_mean"),
                "main_reward_mean": data.get("main_reward_mean"),
                "delta_mrr20_mean": data.get("delta_mrr20_mean"),
                "delta_recall20_mean": data.get("delta_recall20_mean"),
                "keyword_preserve_mean": data.get("keyword_preserve_mean"),
                "recall_drop_penalty_mean": data.get("recall_drop_penalty_mean"),
                "overedit_penalty_mean": data.get("overedit_penalty_mean"),
                "bad_format_penalty_mean": data.get("bad_format_penalty_mean"),
                "unsafe_copy_penalty_mean": data.get("unsafe_copy_penalty_mean"),
            }
        )

    per = pd.DataFrame(report["per_qid"])
    per_rows = []
    for _, row in per.iterrows():
        qid = row["qid"]
        orig = row["original"]
        zero = row["zero_shot"]
        rl = row["rl"]
        per_rows.append(
            {
                "qid": qid,
                "zero_minus_original_mrr": zero.get("mrr@20", zero.get("mrr", np.nan))
                - orig.get("mrr@20", orig.get("mrr", np.nan)),
                "rl_minus_original_mrr": rl.get("mrr@20", rl.get("mrr", np.nan))
                - orig.get("mrr@20", orig.get("mrr", np.nan)),
                "rl_minus_zero_mrr": rl.get("mrr@20", rl.get("mrr", np.nan))
                - zero.get("mrr@20", zero.get("mrr", np.nan)),
                "zero_minus_original_recall20": zero.get("recall@20", zero.get("recall", np.nan))
                - orig.get("recall@20", orig.get("recall", np.nan)),
                "rl_minus_original_recall20": rl.get("recall@20", rl.get("recall", np.nan))
                - orig.get("recall@20", orig.get("recall", np.nan)),
                "rl_minus_zero_recall20": rl.get("recall@20", rl.get("recall", np.nan))
                - zero.get("recall@20", zero.get("recall", np.nan)),
                "zero_reward": zero.get("reward", np.nan),
                "rl_reward": rl.get("reward", np.nan),
            }
        )
    return pd.DataFrame(eval_rows), pd.DataFrame(per_rows), report


def plot_training_overview(train: pd.DataFrame, spans):
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    axes = axes.ravel()
    panels = [
        ("reward_mean", "Reward mean", None),
        ("main_reward_mean", "Main reward mean", None),
        ("mrr_mean", "Rewrite MRR@20 mean", None),
        ("recall_mean", "Rewrite Recall@20 mean", None),
        ("delta_mrr20_mean", "Delta MRR@20 vs original", 0),
        ("delta_recall20_mean", "Delta Recall@20 vs original", 0),
    ]

    for ax, (metric, title, baseline) in zip(axes, panels):
        add_stage_background(ax, spans)
        ax.plot(train["global_step"], train[metric], color="#222222", lw=1.0, alpha=0.35, label="raw")
        ax.plot(train["global_step"], rolling(train[metric]), color="#111111", lw=2.0, label="5-step rolling")
        for span in spans:
            sdf = train[(train["global_step"] >= span["start"]) & (train["global_step"] <= span["end"])]
            ax.plot(sdf["global_step"], rolling(sdf[metric]), color=span["color"], lw=2.2)
        if baseline is not None:
            ax.axhline(baseline, color="#666666", ls=":", lw=1.0)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[-2].set_xlabel("Global training step")
    axes[-1].set_xlabel("Global training step")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Two-stage training metrics with stage backgrounds", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(OUT / "combined_training_metrics.png", dpi=180)
    plt.close(fig)


def plot_training_diagnostics(train: pd.DataFrame, spans):
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    axes = axes.ravel()
    panels = [
        ("keyword_preserve_mean", "Keyword preservation"),
        ("recall_drop_ratio", "Recall drop ratio"),
        ("trainable_group_ratio", "Trainable group ratio"),
        ("flat_reward_group_ratio", "Flat reward group ratio"),
        ("extra_sample_ratio", "Extra sample ratio"),
        ("kl_dominance_ratio", "KL dominance ratio"),
    ]
    for ax, (metric, title) in zip(axes, panels):
        add_stage_background(ax, spans)
        ax.plot(train["global_step"], train[metric], color="#222222", lw=1.0, alpha=0.3)
        for span in spans:
            sdf = train[(train["global_step"] >= span["start"]) & (train["global_step"] <= span["end"])]
            ax.plot(sdf["global_step"], rolling(sdf[metric]), color=span["color"], lw=2.2)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[-2].set_xlabel("Global training step")
    axes[-1].set_xlabel("Global training step")
    fig.suptitle("Sampling and optimization diagnostics", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(OUT / "combined_training_diagnostics.png", dpi=180)
    plt.close(fig)


def plot_group_trace(trace: pd.DataFrame, spans):
    per_step = (
        trace.groupby(["stage", "stage_idx", "global_step"], sort=False)
        .agg(
            reward_gap_met_rate=("reward_gap_met", "mean"),
            reward_gap_raw_mean=("reward_gap_raw", "mean"),
            generated_sample_count_mean=("generated_sample_count", "mean"),
            extra_sample_count_mean=("extra_sample_count", "mean"),
            unique_final_query_ratio_mean=("unique_final_query_ratio", "mean"),
            group_reward_mean=("group_reward_mean", "mean"),
            group_delta_mrr_mean=("group_delta_mrr_mean", "mean"),
            group_keyword_preserve_mean=("group_keyword_preserve_mean", "mean"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(4, 2, figsize=(15, 15), sharex=True)
    axes = axes.ravel()
    panels = [
        ("reward_gap_met_rate", "Reward gap met rate"),
        ("reward_gap_raw_mean", "Mean raw reward gap"),
        ("generated_sample_count_mean", "Generated samples per group"),
        ("extra_sample_count_mean", "Extra samples per group"),
        ("unique_final_query_ratio_mean", "Unique final query ratio"),
        ("group_reward_mean", "Mean group reward"),
        ("group_delta_mrr_mean", "Mean group delta MRR"),
        ("group_keyword_preserve_mean", "Mean group keyword preservation"),
    ]
    for ax, (metric, title) in zip(axes, panels):
        add_stage_background(ax, spans)
        ax.plot(per_step["global_step"], per_step[metric], color="#222222", lw=1.0, alpha=0.25)
        for span in spans:
            sdf = per_step[(per_step["global_step"] >= span["start"]) & (per_step["global_step"] <= span["end"])]
            ax.plot(sdf["global_step"], rolling(sdf[metric]), color=span["color"], lw=2.2)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[-2].set_xlabel("Global training step")
    axes[-1].set_xlabel("Global training step")
    fig.suptitle("Group trace dynamics", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(OUT / "combined_group_trace_dynamics.png", dpi=180)
    plt.close(fig)


def plot_stage_distribution(trace: pd.DataFrame):
    metrics = [
        ("reward_gap_raw", "Raw reward gap"),
        ("generated_sample_count", "Generated samples"),
        ("unique_final_query_count", "Unique final queries"),
        ("group_reward_mean", "Group reward mean"),
        ("group_delta_mrr_mean", "Group delta MRR mean"),
        ("group_keyword_preserve_mean", "Group keyword preservation"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.ravel()
    data_by_stage = [trace[trace["stage"] == s["name"]] for s in STAGES]
    colors = [s["color"] for s in STAGES]
    for ax, (metric, title) in zip(axes, metrics):
        vals = [df[metric].dropna().to_numpy() for df in data_by_stage]
        bp = ax.boxplot(vals, patch_artist=True, tick_labels=[s["name"] for s in STAGES], showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Per-group distribution by stage", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "stage_group_distributions.png", dpi=180)
    plt.close(fig)


def plot_eval(eval_summary: pd.DataFrame, per_qid: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["#6A6A6A", "#4C78A8", "#F58518"]

    ax = axes[0, 0]
    metrics = ["mrr_mean", "recall20_mean", "recall50_mean"]
    x = np.arange(len(metrics))
    width = 0.25
    for i, (_, row) in enumerate(eval_summary.iterrows()):
        ax.bar(x + (i - 1) * width, [row[m] for m in metrics], width, label=row["system"], color=colors[i])
    ax.set_xticks(x, ["MRR@20", "Recall@20", "Recall@50"])
    ax.set_title("Final evaluation retrieval metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    metrics = ["reward_mean", "main_reward_mean"]
    x = np.arange(len(metrics))
    for i, (_, row) in enumerate(eval_summary.iterrows()):
        ax.bar(x + (i - 1) * width, [row[m] for m in metrics], width, label=row["system"], color=colors[i])
    ax.axhline(0, color="#555555", lw=1)
    ax.set_xticks(x, ["Reward", "Main reward"])
    ax.set_title("Final evaluation reward")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    metrics = ["delta_mrr20_mean", "delta_recall20_mean"]
    x = np.arange(len(metrics))
    for i, (_, row) in enumerate(eval_summary.iterrows()):
        ax.bar(x + (i - 1) * width, [row[m] for m in metrics], width, label=row["system"], color=colors[i])
    ax.axhline(0, color="#555555", lw=1)
    ax.set_xticks(x, ["Delta MRR@20", "Delta Recall@20"])
    ax.set_title("Final evaluation deltas vs original")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    win_rates = {
        "RL > Zero MRR": (per_qid["rl_minus_zero_mrr"] > 0).mean(),
        "RL < Zero MRR": (per_qid["rl_minus_zero_mrr"] < 0).mean(),
        "RL > Original MRR": (per_qid["rl_minus_original_mrr"] > 0).mean(),
        "RL Recall gain": (per_qid["rl_minus_original_recall20"] > 0).mean(),
    }
    ax.bar(list(win_rates.keys()), list(win_rates.values()), color=["#54A24B", "#E45756", "#72B7B2", "#B279A2"])
    ax.set_ylim(0, 1)
    ax.set_title("Per-query win/loss rates")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Final evaluation comparison", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "eval_comparison_summary.png", dpi=180)
    plt.close(fig)


def write_report(
    train: pd.DataFrame,
    trace: pd.DataFrame,
    train_summary: pd.DataFrame,
    trace_summary: pd.DataFrame,
    eval_summary: pd.DataFrame,
    per_qid: pd.DataFrame,
    report: dict,
):
    stage1_steps = int((train["stage"] == "Stage 1").sum())
    stage2_steps = int((train["stage"] == "Stage 2").sum())
    stage1_groups = int((trace["stage"] == "Stage 1").sum())
    stage2_groups = int((trace["stage"] == "Stage 2").sum())

    def metric_summary(metric):
        rows = train_summary[train_summary["metric"] == metric].set_index("stage")
        return rows

    reward = metric_summary("reward_mean")
    main_reward = metric_summary("main_reward_mean")
    delta_mrr = metric_summary("delta_mrr20_mean")
    recall_drop = metric_summary("recall_drop_ratio")
    keyword = metric_summary("keyword_preserve_mean")
    trainable = metric_summary("trainable_group_ratio")

    eval_rl = eval_summary[eval_summary["system"] == "RL"].iloc[0]
    eval_zero = eval_summary[eval_summary["system"] == "Zero-shot"].iloc[0]
    eval_orig = eval_summary[eval_summary["system"] == "Original"].iloc[0]

    rl_vs_zero_win = (per_qid["rl_minus_zero_mrr"] > 0).mean()
    rl_vs_zero_loss = (per_qid["rl_minus_zero_mrr"] < 0).mean()
    rl_vs_orig_win = (per_qid["rl_minus_original_mrr"] > 0).mean()
    rl_recall_gain = (per_qid["rl_minus_original_recall20"] > 0).mean()

    lines = [
        "# 两阶段 DeepRetrieval-GRPO 实验分析",
        "",
        "## 数据概况",
        "",
        f"- 阶段 1：{stage1_steps} 个训练 step，{stage1_groups} 条 group trace。",
        f"- 阶段 2：{stage2_steps} 个训练 step，{stage2_groups} 条 group trace。",
        f"- 最终评估：{report['num_eval_queries']} 个 query，对比 Original、Zero-shot rewrite、RL rewrite。",
        "",
        "## 训练过程结论",
        "",
        f"- reward_mean：阶段 1 平均 {num(reward.loc['Stage 1','mean'])}，末 10 step {num(reward.loc['Stage 1','last_10_mean'])}；阶段 2 平均 {num(reward.loc['Stage 2','mean'])}，末 10 step {num(reward.loc['Stage 2','last_10_mean'])}。",
        f"- main_reward_mean：阶段 1 平均 {num(main_reward.loc['Stage 1','mean'])}，阶段 2 平均 {num(main_reward.loc['Stage 2','mean'])}；阶段 2 主奖励整体更高。",
        f"- delta_mrr20_mean：阶段 1 平均 {num(delta_mrr.loc['Stage 1','mean'])}，阶段 2 平均 {num(delta_mrr.loc['Stage 2','mean'])}；阶段 2 的 MRR 增益更稳定地保持在正区间。",
        f"- recall_drop_ratio：阶段 1 平均 {pct(recall_drop.loc['Stage 1','mean'])}，阶段 2 平均 {pct(recall_drop.loc['Stage 2','mean'])}；阶段 2 的召回下降风险明显降低。",
        f"- keyword_preserve_mean：阶段 1 平均 {pct(keyword.loc['Stage 1','mean'])}，阶段 2 平均 {pct(keyword.loc['Stage 2','mean'])}；阶段 2 关键词保留更好。",
        f"- trainable_group_ratio：阶段 1 平均 {pct(trainable.loc['Stage 1','mean'])}，阶段 2 平均 {pct(trainable.loc['Stage 2','mean'])}；阶段 2 中可训练组比例下降，说明更多组 reward 变平或差异不足。",
        "",
        "## Group trace 结论",
        "",
    ]
    for _, row in trace_summary.iterrows():
        lines.extend(
            [
                f"- {row['stage']}：reward gap 达标率 {pct(row['reward_gap_met_rate'])}，平均 reward gap {num(row['mean_reward_gap_raw'])}，平均生成样本数 {num(row['mean_generated_sample_count'])}，平均额外采样 {num(row['mean_extra_sample_count'])}。",
                f"- {row['stage']}：threshold_reached {pct(row['stop_threshold_reached_rate'])}，max_group_size_reached {pct(row['stop_max_group_size_reached_rate'])}，平均唯一 final query 数 {num(row['mean_unique_final_query_count'])}。",
            ]
        )
    lines.extend(
        [
            "",
            "## 最终评估结论",
            "",
            f"- MRR@20：Original {num(eval_orig['mrr_mean'])}，Zero-shot {num(eval_zero['mrr_mean'])}，RL {num(eval_rl['mrr_mean'])}。RL 相比 Original 提升 {num(report['deltas']['rl_minus_original'])}，相比 Zero-shot 提升 {num(report['deltas']['rl_minus_zero'])}。",
            f"- Recall@20：Original {num(eval_orig['recall20_mean'])}，RL {num(eval_rl['recall20_mean'])}，RL 的平均 delta_recall20 为 {num(eval_rl['delta_recall20_mean'])}。",
            f"- Reward：Original {num(eval_orig['reward_mean'])}，Zero-shot {num(eval_zero['reward_mean'])}，RL {num(eval_rl['reward_mean'])}；RL reward 相比 Original 提升 {num(report['deltas']['rl_reward_minus_original'])}。",
            f"- 安全/格式惩罚：RL 的 recall_drop_penalty {num(eval_rl['recall_drop_penalty_mean'])}，overedit_penalty {num(eval_rl['overedit_penalty_mean'])}，bad_format_penalty {num(eval_rl['bad_format_penalty_mean'])}，unsafe_copy_penalty {num(eval_rl['unsafe_copy_penalty_mean'])}。",
            f"- 按 query 统计：RL MRR 优于 Zero-shot 的比例 {pct(rl_vs_zero_win)}，低于 Zero-shot 的比例 {pct(rl_vs_zero_loss)}，优于 Original 的比例 {pct(rl_vs_orig_win)}；RL Recall@20 有增益的比例 {pct(rl_recall_gain)}。",
            "",
            "## 输出图表",
            "",
            "- `combined_training_metrics.png`：两个阶段拼接的主要训练指标，浅蓝为阶段 1，浅橙为阶段 2。",
            "- `combined_training_diagnostics.png`：关键词保留、召回下降、可训练组、KL 等诊断指标。",
            "- `combined_group_trace_dynamics.png`：从 group trace 聚合出的采样与组内质量动态。",
            "- `stage_group_distributions.png`：两个阶段的 group 级分布箱线图。",
            "- `eval_comparison_summary.png`：最终评估的 Original / Zero-shot / RL 对比。",
        ]
    )
    (OUT / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    train, trace, spans = load_data()
    train_summary = summarize_train(train)
    trace_summary = summarize_trace(trace)
    eval_summary, per_qid, report = summarize_eval()

    train.to_csv(OUT / "combined_train_log.csv", index=False, encoding="utf-8-sig")
    trace_summary.to_csv(OUT / "trace_stage_summary.csv", index=False, encoding="utf-8-sig")
    train_summary.to_csv(OUT / "train_stage_metric_summary.csv", index=False, encoding="utf-8-sig")
    eval_summary.to_csv(OUT / "eval_summary.csv", index=False, encoding="utf-8-sig")
    per_qid.describe().to_csv(OUT / "eval_per_qid_delta_describe.csv", encoding="utf-8-sig")

    plot_training_overview(train, spans)
    plot_training_diagnostics(train, spans)
    plot_group_trace(trace, spans)
    plot_stage_distribution(trace)
    plot_eval(eval_summary, per_qid)
    write_report(train, trace, train_summary, trace_summary, eval_summary, per_qid, report)

    print(f"Wrote analysis outputs to: {OUT}")


if __name__ == "__main__":
    main()
