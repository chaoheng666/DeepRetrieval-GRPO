from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a train_log.jsonl and group_trace_log.jsonl pair into charts and summary files."
    )
    parser.add_argument("--train-log", required=True, help="Path to train_log.jsonl")
    parser.add_argument("--group-log", required=True, help="Path to group_trace_log.jsonl")
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs/log_summary",
        help="Directory where summary files and plots will be written.",
    )
    parser.add_argument(
        "--title",
        default="Training Log Summary",
        help="Title prefix used in the charts.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc
    return rows


def safe_mean(values: list[float]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return None
    return fmean(cleaned)


def safe_sum(values: list[float]) -> float:
    return float(sum(float(value) for value in values if value is not None))


def safe_ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def flatten_numeric_lists(rows: list[dict[str, Any]], key: str) -> list[float]:
    flattened: list[float] = []
    for row in rows:
        values = row.get(key) or []
        if isinstance(values, list):
            flattened.extend(float(value) for value in values)
    return flattened


def first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def set_fallback_metric(row: dict[str, Any], dest_key: str, *source_keys: str) -> None:
    if dest_key in row and row[dest_key] is not None:
        return
    fallback = first_present(row, *source_keys)
    if fallback is not None:
        row[dest_key] = fallback


def sanitize_name(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    return lowered.strip("_") or "summary"


def build_train_steps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train_steps: list[dict[str, Any]] = []
    ordered = sorted(rows, key=lambda row: (int(row.get("epoch", 0)), int(row.get("step", 0))))
    for idx, row in enumerate(ordered, start=1):
        sampled = row.get("sampled")
        valid_samples = row.get("valid_samples")
        entry = dict(row)
        set_fallback_metric(entry, "rewrite_mrr20_mean", "mrr_mean")
        set_fallback_metric(entry, "rewrite_recall20_mean", "recall_mean")
        set_fallback_metric(entry, "rewrite_recall50_mean", "recall_dense_mean")
        set_fallback_metric(entry, "main_reward_mean", "reward_mean")
        set_fallback_metric(entry, "flat_main_reward_group_ratio", "flat_reward_group_ratio")
        entry["global_step_idx"] = idx
        entry["valid_ratio"] = safe_ratio(valid_samples, sampled)
        train_steps.append(entry)
    return train_steps


def _count_unique_nonempty(values: list[Any]) -> int:
    seen = {str(value).strip() for value in values if str(value).strip()}
    return len(seen)


def build_group_steps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_step: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (int(row.get("epoch", 0)), int(row.get("step", 0)))
        rows_by_step[key].append(row)

    aggregated_steps: list[dict[str, Any]] = []
    ordered_keys = sorted(rows_by_step)
    for idx, key in enumerate(ordered_keys, start=1):
        epoch, step = key
        step_rows = rows_by_step[key]

        reward_values = flatten_numeric_lists(step_rows, "group_rewards")
        mrr_values = flatten_numeric_lists(step_rows, "group_mrr")
        recall_values = flatten_numeric_lists(step_rows, "group_recall")
        recall_dense_values = flatten_numeric_lists(step_rows, "group_recall_dense")
        orig_mrr_values = flatten_numeric_lists(step_rows, "group_orig_mrr")
        delta_mrr_values = flatten_numeric_lists(step_rows, "group_delta_mrr")
        main_reward_values = flatten_numeric_lists(step_rows, "group_main_rewards")
        anchor_bonus_values = flatten_numeric_lists(step_rows, "group_anchor_bonus")
        recall_drop_penalty_values = flatten_numeric_lists(step_rows, "group_recall_drop_penalties")
        overedit_values = flatten_numeric_lists(step_rows, "group_overedit_penalties")
        term_preserve_values = flatten_numeric_lists(step_rows, "group_term_preserve")
        keyword_preserve_values = flatten_numeric_lists(step_rows, "group_keyword_preserve")
        locked_term_preserve_values = flatten_numeric_lists(step_rows, "group_locked_term_preserve")
        length_score_values = flatten_numeric_lists(step_rows, "group_length_scores")
        clean_format_values = flatten_numeric_lists(step_rows, "group_clean_format_scores")
        bad_format_values = flatten_numeric_lists(step_rows, "group_bad_format_penalties")
        unsafe_copy_values = flatten_numeric_lists(step_rows, "group_unsafe_copy_penalties")

        if not unsafe_copy_values:
            unsafe_copy_values = flatten_numeric_lists(step_rows, "group_copy_penalties")

        row_reward_means: list[float] = []
        row_reward_maxes: list[float] = []
        row_unique_final_queries: list[float] = []
        row_fallback_ratios: list[float] = []
        row_flat_reward_flags: list[float] = []
        row_flat_mrr_flags: list[float] = []
        row_flat_main_reward_flags: list[float] = []
        row_best_reward_hit_best_mrr_flags: list[float] = []
        row_all_same_query_flags: list[float] = []

        for row in step_rows:
            group_rewards = row.get("group_rewards") or []
            group_mrr = row.get("group_mrr") or []
            group_main_rewards = row.get("group_main_rewards") or []
            final_queries = row.get("group_final_queries") or []
            fallback_flags = row.get("group_fallback_to_original") or []
            unique_count = _count_unique_nonempty(final_queries)

            if group_rewards:
                numeric_rewards = [float(value) for value in group_rewards]
                row_reward_means.append(fmean(numeric_rewards))
                row_reward_maxes.append(max(numeric_rewards))
                row_flat_reward_flags.append(1.0 if max(numeric_rewards) == min(numeric_rewards) else 0.0)
                if group_mrr:
                    numeric_mrr = [float(value) for value in group_mrr]
                    row_flat_mrr_flags.append(1.0 if max(numeric_mrr) == min(numeric_mrr) else 0.0)
                    best_reward_idx = max(range(len(numeric_rewards)), key=lambda idx: numeric_rewards[idx])
                    row_best_reward_hit_best_mrr_flags.append(
                        1.0 if numeric_mrr[best_reward_idx] == max(numeric_mrr) else 0.0
                    )
                if group_main_rewards:
                    numeric_main_reward = [float(value) for value in group_main_rewards]
                    row_flat_main_reward_flags.append(
                        1.0 if max(numeric_main_reward) == min(numeric_main_reward) else 0.0
                    )

            row_unique_final_queries.append(float(unique_count))
            row_all_same_query_flags.append(1.0 if unique_count <= 1 else 0.0)

            if fallback_flags:
                row_fallback_ratios.append(
                    fmean(1.0 if bool(value) else 0.0 for value in fallback_flags)
                )

        generated_counts = [float(row.get("generated_sample_count", 0.0)) for row in step_rows]
        extra_counts = [float(row.get("extra_sample_count", 0.0)) for row in step_rows]
        initial_counts = [float(row.get("initial_group_size", 0.0)) for row in step_rows]

        aggregated_entry = {
            "epoch": epoch,
            "step": step,
            "global_step_idx": idx,
            "qid_rows": len(step_rows),
            "trace_reward_mean": safe_mean(reward_values),
            "trace_mrr_mean": safe_mean(mrr_values),
            "trace_recall_mean": safe_mean(recall_values),
            "trace_recall_dense_mean": safe_mean(recall_dense_values),
            "trace_orig_mrr20_mean": safe_mean(orig_mrr_values),
            "trace_delta_mrr20_mean": safe_mean(delta_mrr_values),
            "trace_main_reward_mean": safe_mean(main_reward_values),
            "trace_anchor_bonus_mean": safe_mean(anchor_bonus_values),
            "trace_term_preserve_mean": safe_mean(term_preserve_values),
            "trace_keyword_preserve_mean": safe_mean(keyword_preserve_values),
            "trace_locked_term_preserve_mean": safe_mean(locked_term_preserve_values),
            "trace_length_score_mean": safe_mean(length_score_values),
            "trace_clean_format_mean": safe_mean(clean_format_values),
            "trace_recall_drop_penalty_mean": safe_mean(recall_drop_penalty_values),
            "trace_overedit_penalty_mean": safe_mean(overedit_values),
            "trace_bad_format_penalty_mean": safe_mean(bad_format_values),
            "trace_unsafe_copy_penalty_mean": safe_mean(unsafe_copy_values),
            "trace_row_reward_mean_avg": safe_mean(row_reward_means),
            "trace_row_reward_max_avg": safe_mean(row_reward_maxes),
            "trace_unique_final_query_mean": safe_mean(row_unique_final_queries),
            "trace_fallback_ratio": safe_mean(row_fallback_ratios),
            "trace_flat_reward_group_ratio": safe_mean(row_flat_reward_flags),
            "trace_flat_mrr20_group_ratio": safe_mean(row_flat_mrr_flags),
            "trace_flat_main_reward_group_ratio": safe_mean(row_flat_main_reward_flags),
            "trace_best_reward_hit_best_mrr20_ratio": safe_mean(row_best_reward_hit_best_mrr_flags),
            "trace_all_same_final_query_ratio": safe_mean(row_all_same_query_flags),
            "trace_anchor_hit_ratio": safe_mean([1.0 if value > 0.0 else 0.0 for value in anchor_bonus_values]),
            "trace_recall_drop_ratio": safe_mean(
                [1.0 if value > 0.0 else 0.0 for value in recall_drop_penalty_values]
            ),
            "collapsed_group_ratio": safe_mean([1.0 if row.get("collapsed_group") else 0.0 for row in step_rows]),
            "reward_gap_met_ratio": safe_mean([1.0 if row.get("reward_gap_met") else 0.0 for row in step_rows]),
            "max_group_size_hit_ratio": safe_mean(
                [
                    1.0 if row.get("reward_gap_stop_reason") == "max_group_size_reached" else 0.0
                    for row in step_rows
                ]
            ),
            "reward_gap_raw_mean": safe_mean([float(row.get("reward_gap_raw", 0.0)) for row in step_rows]),
            "reward_gap_threshold_mean": safe_mean(
                [float(row.get("reward_gap_threshold", 0.0)) for row in step_rows]
            ),
            "initial_group_size_mean": safe_mean(initial_counts),
            "generated_sample_count_mean": safe_mean(generated_counts),
            "extra_sample_count_mean": safe_mean(extra_counts),
            "extra_sample_ratio": safe_ratio(safe_sum(extra_counts), safe_sum(generated_counts)),
        }
        set_fallback_metric(
            aggregated_entry,
            "trace_flat_main_reward_group_ratio",
            "trace_flat_reward_group_ratio",
        )
        aggregated_steps.append(aggregated_entry)

    return aggregated_steps


def merge_steps(
    train_steps: list[dict[str, Any]],
    group_steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    group_map = {(int(row["epoch"]), int(row["step"])): row for row in group_steps}
    merged: list[dict[str, Any]] = []

    for train_row in train_steps:
        key = (int(train_row["epoch"]), int(train_row["step"]))
        group_row = group_map.get(key, {})
        merged_row = dict(train_row)
        for group_key, group_value in group_row.items():
            if group_key in {"epoch", "step"}:
                continue
            merged_row[group_key] = group_value
        merged.append(merged_row)

    return merged


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pearson_correlation(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mean_x = fmean(xs)
    mean_y = fmean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def paired_metric_correlation(
    merged_steps: list[dict[str, Any]],
    left_key: str,
    right_key: str,
) -> float | None:
    pairs: list[tuple[float, float]] = []
    for row in merged_steps:
        left = maybe_float(row.get(left_key))
        right = maybe_float(row.get(right_key))
        if left is None or right is None:
            continue
        pairs.append((left, right))
    return pearson_correlation(pairs)


def average_first_last(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    split = max(1, len(values) // 3)
    early = values[:split]
    late = values[-split:]
    return safe_mean(early), safe_mean(late)


def metric_series(rows: list[dict[str, Any]], key: str) -> list[float]:
    series: list[float] = []
    for row in rows:
        value = maybe_float(row.get(key))
        if value is not None:
            series.append(value)
    return series


def best_row(rows: list[dict[str, Any]], key: str, maximize: bool = True) -> dict[str, Any] | None:
    candidates = [row for row in rows if maybe_float(row.get(key)) is not None]
    if not candidates:
        return None
    chooser = max if maximize else min
    selected = chooser(candidates, key=lambda row: float(row[key]))
    return {
        "epoch": int(selected.get("epoch", 0)),
        "step": int(selected.get("step", 0)),
        "global_step_idx": int(selected.get("global_step_idx", 0)),
        key: float(selected[key]),
    }


def build_summary(
    train_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    merged_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    train_reward_series = metric_series(merged_steps, "reward_mean")
    trace_reward_series = metric_series(merged_steps, "trace_reward_mean")
    unique_query_series = metric_series(merged_steps, "trace_unique_final_query_mean")
    valid_ratio_series = metric_series(merged_steps, "valid_ratio")

    reward_early, reward_late = average_first_last(train_reward_series)
    trace_reward_early, trace_reward_late = average_first_last(trace_reward_series)
    diversity_early, diversity_late = average_first_last(unique_query_series)
    valid_early, valid_late = average_first_last(valid_ratio_series)

    summary = {
        "train_rows": len(train_rows),
        "group_rows": len(group_rows),
        "aligned_steps": len(merged_steps),
        "train_epochs": sorted({int(row.get("epoch", 0)) for row in train_rows}),
        "group_epochs": sorted({int(row.get("epoch", 0)) for row in group_rows}),
        "step_min": min((int(row.get("step", 0)) for row in merged_steps), default=None),
        "step_max": max((int(row.get("step", 0)) for row in merged_steps), default=None),
        "reward_mean_avg": safe_mean(train_reward_series),
        "trace_reward_mean_avg": safe_mean(trace_reward_series),
        "reward_alignment_corr": paired_metric_correlation(merged_steps, "reward_mean", "trace_reward_mean"),
        "mrr_alignment_corr": paired_metric_correlation(merged_steps, "mrr_mean", "trace_mrr_mean"),
        "recall_alignment_corr": paired_metric_correlation(merged_steps, "recall_mean", "trace_recall_mean"),
        "recall_dense_alignment_corr": paired_metric_correlation(
            merged_steps, "recall_dense_mean", "trace_recall_dense_mean"
        ),
        "term_preserve_alignment_corr": paired_metric_correlation(
            merged_steps, "term_preserve_mean", "trace_term_preserve_mean"
        ),
        "best_train_reward_mean": best_row(merged_steps, "reward_mean", maximize=True),
        "best_trace_reward_mean": best_row(merged_steps, "trace_reward_mean", maximize=True),
        "best_train_mrr_mean": best_row(merged_steps, "mrr_mean", maximize=True),
        "best_train_rewrite_mrr20_mean": best_row(merged_steps, "rewrite_mrr20_mean", maximize=True),
        "lowest_flat_main_reward_group_ratio": best_row(
            merged_steps, "flat_main_reward_group_ratio", maximize=False
        ),
        "best_trace_unique_final_query_mean": best_row(merged_steps, "trace_unique_final_query_mean", maximize=True),
        "highest_gap_met_ratio": best_row(merged_steps, "reward_gap_met_ratio", maximize=True),
        "lowest_collapsed_group_ratio": best_row(merged_steps, "collapsed_group_ratio", maximize=False),
        "loss_min": best_row(merged_steps, "loss", maximize=False),
        "averages": {
            "mrr_mean": safe_mean(metric_series(merged_steps, "mrr_mean")),
            "trace_mrr_mean": safe_mean(metric_series(merged_steps, "trace_mrr_mean")),
            "rewrite_mrr20_mean": safe_mean(metric_series(merged_steps, "rewrite_mrr20_mean")),
            "trace_orig_mrr20_mean": safe_mean(metric_series(merged_steps, "trace_orig_mrr20_mean")),
            "delta_mrr20_positive_ratio": safe_mean(metric_series(merged_steps, "delta_mrr20_positive_ratio")),
            "flat_main_reward_group_ratio": safe_mean(metric_series(merged_steps, "flat_main_reward_group_ratio")),
            "best_reward_hit_best_mrr20_ratio": safe_mean(
                metric_series(merged_steps, "best_reward_hit_best_mrr20_ratio")
            ),
            "anchor_bonus_mean": safe_mean(metric_series(merged_steps, "anchor_bonus_mean")),
            "trace_anchor_bonus_mean": safe_mean(metric_series(merged_steps, "trace_anchor_bonus_mean")),
            "recall_drop_penalty_mean": safe_mean(metric_series(merged_steps, "recall_drop_penalty_mean")),
            "trace_recall_drop_penalty_mean": safe_mean(
                metric_series(merged_steps, "trace_recall_drop_penalty_mean")
            ),
            "anchor_hit_ratio": safe_mean(metric_series(merged_steps, "anchor_hit_ratio")),
            "trace_anchor_hit_ratio": safe_mean(metric_series(merged_steps, "trace_anchor_hit_ratio")),
            "recall_drop_ratio": safe_mean(metric_series(merged_steps, "recall_drop_ratio")),
            "trace_recall_drop_ratio": safe_mean(metric_series(merged_steps, "trace_recall_drop_ratio")),
            "recall_mean": safe_mean(metric_series(merged_steps, "recall_mean")),
            "trace_recall_mean": safe_mean(metric_series(merged_steps, "trace_recall_mean")),
            "recall_dense_mean": safe_mean(metric_series(merged_steps, "recall_dense_mean")),
            "trace_recall_dense_mean": safe_mean(metric_series(merged_steps, "trace_recall_dense_mean")),
            "term_preserve_mean": safe_mean(metric_series(merged_steps, "term_preserve_mean")),
            "trace_term_preserve_mean": safe_mean(metric_series(merged_steps, "trace_term_preserve_mean")),
            "length_score_mean": safe_mean(metric_series(merged_steps, "length_score_mean")),
            "trace_length_score_mean": safe_mean(metric_series(merged_steps, "trace_length_score_mean")),
            "bad_format_penalty_mean": safe_mean(metric_series(merged_steps, "bad_format_penalty_mean")),
            "trace_bad_format_penalty_mean": safe_mean(metric_series(merged_steps, "trace_bad_format_penalty_mean")),
            "unsafe_copy_penalty_mean": safe_mean(metric_series(merged_steps, "unsafe_copy_penalty_mean")),
            "trace_unsafe_copy_penalty_mean": safe_mean(
                metric_series(merged_steps, "trace_unsafe_copy_penalty_mean")
            ),
            "unique_final_query_mean": safe_mean(metric_series(merged_steps, "unique_final_query_mean")),
            "trace_unique_final_query_mean": safe_mean(
                metric_series(merged_steps, "trace_unique_final_query_mean")
            ),
            "collapsed_group_ratio": safe_mean(metric_series(merged_steps, "collapsed_group_ratio")),
            "reward_gap_met_ratio": safe_mean(metric_series(merged_steps, "reward_gap_met_ratio")),
            "max_group_size_hit_ratio": safe_mean(metric_series(merged_steps, "max_group_size_hit_ratio")),
            "generated_sample_count_mean": safe_mean(metric_series(merged_steps, "generated_sample_count_mean")),
            "extra_sample_ratio": safe_mean(metric_series(merged_steps, "extra_sample_ratio")),
            "valid_ratio": safe_mean(metric_series(merged_steps, "valid_ratio")),
            "loss": safe_mean(metric_series(merged_steps, "loss")),
            "loss_pg": safe_mean(metric_series(merged_steps, "loss_pg")),
            "loss_kl": safe_mean(metric_series(merged_steps, "loss_kl")),
        },
        "trend_first_last_third": {
            "reward_mean_early": reward_early,
            "reward_mean_late": reward_late,
            "trace_reward_mean_early": trace_reward_early,
            "trace_reward_mean_late": trace_reward_late,
            "trace_unique_final_query_mean_early": diversity_early,
            "trace_unique_final_query_mean_late": diversity_late,
            "valid_ratio_early": valid_early,
            "valid_ratio_late": valid_late,
        },
    }
    return summary


def write_step_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_step_dashboard(merged_steps: list[dict[str, Any]], output_path: Path, title: str) -> None:
    x = [int(row.get("step", 0)) for row in merged_steps]

    fig, axes = plt.subplots(3, 2, figsize=(18, 14), dpi=200)
    axes = axes.flatten()

    axes[0].plot(x, metric_series(merged_steps, "reward_mean"), label="train reward_mean", linewidth=2.0)
    axes[0].plot(
        x,
        metric_series(merged_steps, "trace_reward_mean"),
        label="group trace reward_mean",
        linewidth=1.8,
        linestyle="--",
    )
    axes[0].plot(
        x,
        metric_series(merged_steps, "trace_row_reward_max_avg"),
        label="group trace mean max reward",
        linewidth=1.3,
        alpha=0.8,
    )
    axes[0].set_title("Reward")
    axes[0].set_xlabel("step")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(x, metric_series(merged_steps, "mrr_mean"), label="train mrr", linewidth=2.0)
    axes[1].plot(x, metric_series(merged_steps, "recall_mean"), label="train recall", linewidth=1.8)
    if metric_series(merged_steps, "recall_dense_mean"):
        axes[1].plot(
            x,
            metric_series(merged_steps, "recall_dense_mean"),
            label="train recall_dense",
            linewidth=1.6,
        )
    axes[1].plot(x, metric_series(merged_steps, "trace_mrr_mean"), label="trace mrr", linestyle="--", linewidth=1.6)
    axes[1].plot(
        x,
        metric_series(merged_steps, "trace_recall_mean"),
        label="trace recall",
        linestyle="--",
        linewidth=1.5,
    )
    if metric_series(merged_steps, "trace_recall_dense_mean"):
        axes[1].plot(
            x,
            metric_series(merged_steps, "trace_recall_dense_mean"),
            label="trace recall_dense",
            linestyle="--",
            linewidth=1.4,
        )
    axes[1].set_title("Retrieval Metrics")
    axes[1].set_xlabel("step")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, ncol=2)

    axes[2].plot(
        x,
        metric_series(merged_steps, "unique_final_query_mean"),
        label="train unique_final_query_mean",
        linewidth=2.0,
    )
    axes[2].plot(
        x,
        metric_series(merged_steps, "trace_unique_final_query_mean"),
        label="trace unique_final_query_mean",
        linewidth=1.8,
        linestyle="--",
    )
    axes[2].plot(
        x,
        metric_series(merged_steps, "collapsed_group_ratio"),
        label="collapsed_group_ratio",
        linewidth=1.5,
    )
    axes[2].plot(
        x,
        metric_series(merged_steps, "reward_gap_met_ratio"),
        label="reward_gap_met_ratio",
        linewidth=1.5,
    )
    axes[2].set_title("Diversity And Collapse")
    axes[2].set_xlabel("step")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)

    axes[3].plot(
        x,
        metric_series(merged_steps, "generated_sample_count_mean"),
        label="generated_sample_count_mean",
        linewidth=2.0,
    )
    axes[3].plot(
        x,
        metric_series(merged_steps, "initial_group_size_mean"),
        label="initial_group_size_mean",
        linewidth=1.6,
    )
    axes[3].plot(
        x,
        metric_series(merged_steps, "extra_sample_count_mean"),
        label="extra_sample_count_mean",
        linewidth=1.6,
    )
    axes[3].plot(
        x,
        metric_series(merged_steps, "extra_sample_ratio"),
        label="extra_sample_ratio",
        linewidth=1.4,
        linestyle="--",
    )
    axes[3].set_title("Sampling Behavior")
    axes[3].set_xlabel("step")
    axes[3].grid(alpha=0.25)
    axes[3].legend(fontsize=8)

    ax_loss = axes[4]
    ax_loss.plot(x, metric_series(merged_steps, "loss"), label="loss", linewidth=2.0)
    ax_loss.plot(x, metric_series(merged_steps, "loss_pg"), label="loss_pg", linewidth=1.7)
    ax_loss.plot(x, metric_series(merged_steps, "loss_kl"), label="loss_kl", linewidth=1.4)
    ax_loss.set_title("Loss And Valid Ratio")
    ax_loss.set_xlabel("step")
    ax_loss.grid(alpha=0.25)
    ax_valid = ax_loss.twinx()
    ax_valid.plot(
        x,
        metric_series(merged_steps, "valid_ratio"),
        label="valid_ratio",
        linewidth=1.5,
        color="black",
        linestyle="--",
    )
    loss_lines, loss_labels = ax_loss.get_legend_handles_labels()
    valid_lines, valid_labels = ax_valid.get_legend_handles_labels()
    ax_loss.legend(loss_lines + valid_lines, loss_labels + valid_labels, fontsize=8, loc="best")

    axes[5].plot(
        x,
        metric_series(merged_steps, "term_preserve_mean"),
        label="train term_preserve",
        linewidth=2.0,
    )
    axes[5].plot(
        x,
        metric_series(merged_steps, "trace_term_preserve_mean"),
        label="trace term_preserve",
        linewidth=1.8,
        linestyle="--",
    )
    axes[5].plot(
        x,
        metric_series(merged_steps, "length_score_mean"),
        label="train length_score",
        linewidth=1.6,
    )
    axes[5].plot(
        x,
        metric_series(merged_steps, "trace_length_score_mean"),
        label="trace length_score",
        linewidth=1.4,
        linestyle="--",
    )
    axes[5].plot(
        x,
        metric_series(merged_steps, "unsafe_copy_penalty_mean"),
        label="train unsafe_copy_penalty",
        linewidth=1.3,
        alpha=0.9,
    )
    if metric_series(merged_steps, "trace_unsafe_copy_penalty_mean"):
        axes[5].plot(
            x,
            metric_series(merged_steps, "trace_unsafe_copy_penalty_mean"),
            label="trace unsafe_copy_penalty",
            linewidth=1.2,
            linestyle="--",
            alpha=0.9,
        )
    axes[5].set_title("Preservation And Penalty")
    axes[5].set_xlabel("step")
    axes[5].grid(alpha=0.25)
    axes[5].legend(fontsize=8, ncol=2)

    fig.suptitle(title, fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_group_distributions(group_rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    reward_gap_values = [float(row.get("reward_gap_raw", 0.0)) for row in group_rows]
    generated_counts = [float(row.get("generated_sample_count", 0.0)) for row in group_rows]
    unique_counts = [_count_unique_nonempty(row.get("group_final_queries") or []) for row in group_rows]

    row_reward_means: list[float] = []
    for row in group_rows:
        rewards = row.get("group_rewards") or []
        if rewards:
            row_reward_means.append(fmean(float(value) for value in rewards))

    collapsed = sum(1 for row in group_rows if row.get("collapsed_group"))
    reward_gap_met = sum(1 for row in group_rows if row.get("reward_gap_met"))
    max_group_size_hit = sum(1 for row in group_rows if row.get("reward_gap_stop_reason") == "max_group_size_reached")
    other_stop = len(group_rows) - max_group_size_hit

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=200)
    axes = axes.flatten()

    axes[0].hist(row_reward_means, bins=24, color="#2E86AB", alpha=0.85, edgecolor="white")
    axes[0].set_title("Per-QID Mean Reward")
    axes[0].set_xlabel("mean reward")
    axes[0].set_ylabel("count")
    axes[0].grid(alpha=0.2)

    axes[1].hist(reward_gap_values, bins=24, color="#F18F01", alpha=0.85, edgecolor="white")
    axes[1].set_title("Reward Gap Raw")
    axes[1].set_xlabel("reward_gap_raw")
    axes[1].set_ylabel("count")
    axes[1].grid(alpha=0.2)

    axes[2].hist(unique_counts, bins=range(1, max(unique_counts, default=1) + 2), color="#6A4C93", alpha=0.85)
    axes[2].set_title("Unique Final Queries Per QID")
    axes[2].set_xlabel("unique final query count")
    axes[2].set_ylabel("count")
    axes[2].grid(alpha=0.2)

    categories = ["collapsed_group", "reward_gap_met", "max_group_size_hit", "other_stop_reason"]
    values = [collapsed, reward_gap_met, max_group_size_hit, other_stop]
    axes[3].bar(categories, values, color=["#C73E1D", "#4CAF50", "#577590", "#9E9E9E"])
    axes[3].set_title("Trace Outcome Counts")
    axes[3].set_ylabel("count")
    axes[3].grid(alpha=0.2, axis="y")
    axes[3].tick_params(axis="x", rotation=15)

    fig.suptitle(title, fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    train_log_path = Path(args.train_log).expanduser().resolve()
    group_log_path = Path(args.group_log).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(train_log_path)
    group_rows = read_jsonl(group_log_path)

    train_steps = build_train_steps(train_rows)
    group_steps = build_group_steps(group_rows)
    merged_steps = merge_steps(train_steps, group_steps)
    summary = build_summary(train_rows, group_rows, merged_steps)

    summary_json_path = output_dir / "summary.json"
    step_csv_path = output_dir / "step_metrics.csv"
    dashboard_path = output_dir / "step_dashboard.png"
    distribution_path = output_dir / "group_distributions.png"

    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_step_csv(merged_steps, step_csv_path)
    plot_step_dashboard(merged_steps, dashboard_path, args.title)
    plot_group_distributions(group_rows, distribution_path, args.title + " - Group Trace Distribution")

    print(f"train_log={train_log_path}")
    print(f"group_log={group_log_path}")
    print(f"output_dir={output_dir}")
    print(f"summary_json={summary_json_path}")
    print(f"step_csv={step_csv_path}")
    print(f"dashboard_png={dashboard_path}")
    print(f"distribution_png={distribution_path}")


if __name__ == "__main__":
    main()
