#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_TS="$(date '+%Y%m%d_%H%M%S')"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MODEL_NAME="${MODEL_NAME:-/root/autodl-tmp/hf_models/Qwen3-4B-Instruct-2507}"
LOG_DIR="${LOG_DIR:-log}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-train_and_eval_data_model_0423/artifacts_4b_top20_delta_curriculum_sweeps}"
RUN_ROOT="${RUN_ROOT:-${BASE_RUN_ROOT}/run_${RUN_TS}}"
CURRICULUM_METADATA_PATH="${CURRICULUM_METADATA_PATH:-${RUN_ROOT}/curriculum_query_metadata.jsonl}"

PHASE1_DIR="${RUN_ROOT}/phase1"
PHASE1_CHECKPOINT_DIR="${PHASE1_DIR}/checkpoints"
PHASE2_SWEEP_DIR="${RUN_ROOT}/phase2_sweep"
EVAL_DIR="${RUN_ROOT}/eval"
MANIFEST_PATH="${EVAL_DIR}/phase2_variant_manifest.jsonl"
SWEEP_SUMMARY_PATH="${EVAL_DIR}/phase2_sweep_summary.json"
SWEEP_LEADERBOARD_CSV="${EVAL_DIR}/phase2_sweep_leaderboard.csv"

PHASE1_ADAPTER_PATH="${PHASE1_ADAPTER_PATH:-${PHASE1_CHECKPOINT_PATH:-}}"
PHASE2_START_FROM="${PHASE2_START_FROM:-best}"
MAX_TRAIN_QUERIES="${MAX_TRAIN_QUERIES:-}"
MAX_VAL_QUERIES="${MAX_VAL_QUERIES:-}"
MAX_EVAL_QUERIES="${MAX_EVAL_QUERIES:-1000000}"
TARGET_DELTA_VS_ZERO="${TARGET_DELTA_VS_ZERO:-0.01}"

usage() {
  cat <<'EOF'
Usage:
  bash run_train_then_full_eval_4b_top20_delta_curriculum.sh [options]

Options:
  --phase1-adapter-path PATH     Optional checkpoint adapter to warm-start phase1.
  --phase1-checkpoint-path PATH  Alias of --phase1-adapter-path.
  --phase2-start-from best|latest
                                  Choose which phase1 checkpoint phase2 variants start from.
                                  Default: best
  --run-root PATH                Output root for this sweep run.
  --python-bin PATH              Python executable.
  --model-name PATH              Base model name/path.
  --max-train-queries N          Optional cap forwarded to phase1/phase2 training.
  --max-val-queries N            Optional validation cap forwarded to phase1/phase2 training.
  --max-eval-queries N           Optional eval cap for full compare. Default: 1000000
  --target-delta-vs-zero VALUE   Success target for RL minus zero-shot MRR. Default: 0.01
  -h, --help                     Show this help.

Env overrides are also supported:
  RUN_ROOT=/path/to/output
  PHASE1_ADAPTER_PATH=/path/to/checkpoints/latest
  PHASE1_CHECKPOINT_PATH=/path/to/checkpoints/latest
  PHASE2_START_FROM=best|latest
  MAX_TRAIN_QUERIES=512 MAX_VAL_QUERIES=128 MAX_EVAL_QUERIES=256
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase1-adapter-path|--phase1-checkpoint-path)
      [[ $# -ge 2 ]] || { echo "[error] $1 requires a path" >&2; exit 1; }
      PHASE1_ADAPTER_PATH="$2"
      shift 2
      ;;
    --phase2-start-from)
      [[ $# -ge 2 ]] || { echo "[error] --phase2-start-from requires best or latest" >&2; exit 1; }
      PHASE2_START_FROM="$2"
      shift 2
      ;;
    --run-root)
      [[ $# -ge 2 ]] || { echo "[error] --run-root requires a path" >&2; exit 1; }
      RUN_ROOT="$2"
      CURRICULUM_METADATA_PATH="${RUN_ROOT}/curriculum_query_metadata.jsonl"
      PHASE1_DIR="${RUN_ROOT}/phase1"
      PHASE1_CHECKPOINT_DIR="${PHASE1_DIR}/checkpoints"
      PHASE2_SWEEP_DIR="${RUN_ROOT}/phase2_sweep"
      EVAL_DIR="${RUN_ROOT}/eval"
      MANIFEST_PATH="${EVAL_DIR}/phase2_variant_manifest.jsonl"
      SWEEP_SUMMARY_PATH="${EVAL_DIR}/phase2_sweep_summary.json"
      SWEEP_LEADERBOARD_CSV="${EVAL_DIR}/phase2_sweep_leaderboard.csv"
      shift 2
      ;;
    --python-bin)
      [[ $# -ge 2 ]] || { echo "[error] --python-bin requires a path" >&2; exit 1; }
      PYTHON_BIN="$2"
      shift 2
      ;;
    --model-name)
      [[ $# -ge 2 ]] || { echo "[error] --model-name requires a path/name" >&2; exit 1; }
      MODEL_NAME="$2"
      shift 2
      ;;
    --max-train-queries)
      [[ $# -ge 2 ]] || { echo "[error] --max-train-queries requires a value" >&2; exit 1; }
      MAX_TRAIN_QUERIES="$2"
      shift 2
      ;;
    --max-val-queries)
      [[ $# -ge 2 ]] || { echo "[error] --max-val-queries requires a value" >&2; exit 1; }
      MAX_VAL_QUERIES="$2"
      shift 2
      ;;
    --max-eval-queries)
      [[ $# -ge 2 ]] || { echo "[error] --max-eval-queries requires a value" >&2; exit 1; }
      MAX_EVAL_QUERIES="$2"
      shift 2
      ;;
    --target-delta-vs-zero)
      [[ $# -ge 2 ]] || { echo "[error] --target-delta-vs-zero requires a value" >&2; exit 1; }
      TARGET_DELTA_VS_ZERO="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$PHASE2_START_FROM" in
  best)
    ;;
  latest|last|laste)
    PHASE2_START_FROM="latest"
    ;;
  *)
    echo "[error] PHASE2_START_FROM must be best or latest, got: $PHASE2_START_FROM" >&2
    exit 1
    ;;
esac

if [[ -n "$PHASE1_ADAPTER_PATH" && ! -d "$PHASE1_ADAPTER_PATH" ]]; then
  echo "[error] phase1 adapter checkpoint not found: $PHASE1_ADAPTER_PATH" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$RUN_ROOT" "$PHASE1_DIR" "$PHASE2_SWEEP_DIR" "$EVAL_DIR"

RUN_LOG_PATH="${LOG_DIR}/run_train_then_full_eval_4b_top20_delta_curriculum_${RUN_TS}.log"
if [[ -z "${RUN_LOG_REDIRECTED:-}" ]]; then
  export RUN_LOG_REDIRECTED=1
  exec > >(tee -a "$RUN_LOG_PATH") 2>&1
fi

: > "$MANIFEST_PATH"

echo "[log] command output is also saved to: $RUN_LOG_PATH"
echo "[env] python: $("$PYTHON_BIN" --version 2>&1)"
echo "[env] model: $MODEL_NAME"
echo "[env] run_root: $RUN_ROOT"
echo "[env] curriculum_metadata: $CURRICULUM_METADATA_PATH"
echo "[env] phase1_adapter_path: ${PHASE1_ADAPTER_PATH:-<fresh>}"
echo "[env] phase2_start_from: phase1 ${PHASE2_START_FROM}"
echo "[env] query_caps: train=${MAX_TRAIN_QUERIES:-<all>} val=${MAX_VAL_QUERIES:-<all>} eval=${MAX_EVAL_QUERIES}"
echo "[env] success_target_rl_minus_zero=${TARGET_DELTA_VS_ZERO}"

append_optional_query_caps() {
  local -n args_ref=$1
  if [[ -n "$MAX_TRAIN_QUERIES" ]]; then
    args_ref+=(--max-train-queries "$MAX_TRAIN_QUERIES")
  fi
  if [[ -n "$MAX_VAL_QUERIES" ]]; then
    args_ref+=(--max-val-queries "$MAX_VAL_QUERIES")
  fi
}

run_phase1() {
  local args=(
    bash run_train_phase1_top20_delta_curriculum.sh
    --run-root "$RUN_ROOT"
    --output-dir "$PHASE1_DIR"
    --metadata-path "$CURRICULUM_METADATA_PATH"
    --python-bin "$PYTHON_BIN"
    --model-name "$MODEL_NAME"
    --num-epochs 1
    --max-steps 300
    --eval-every-steps 20
    --learning-rate 1.0e-5
    --kl-beta 0.035
    --temperature 0.90
    --top-p 0.96
    --group-temperature-stride 0.14
    --group-top-p-stride 0.03
    --min-unique-final-queries 5
    --max-regen-rounds 3
    --reward-gap-threshold 0.04
    --reward-w-mrr 0.28
    --reward-w-recall 0.42
    --reward-w-recall-dense 0.20
    --reward-w-rank-bonus 0.10
    --recall-drop-lambda 2.00
    --anchor-bonus-value 0.02
  )

  append_optional_query_caps args
  if [[ -n "$PHASE1_ADAPTER_PATH" ]]; then
    args+=(--source-checkpoint "$PHASE1_ADAPTER_PATH")
  fi

  echo "[phase1] starting recall-first phase1 run"
  "${args[@]}"
}

run_phase2_variant() {
  local variant="$1"
  local lr="$2"
  local kl="$3"
  local temperature="$4"
  local top_p="$5"
  local group_temperature_stride="$6"
  local group_top_p_stride="$7"
  local min_unique="$8"
  local max_regen="$9"
  local reward_gap="${10}"
  local reward_w_mrr="${11}"
  local reward_w_recall="${12}"
  local reward_w_recall_dense="${13}"
  local reward_w_rank="${14}"
  local recall_drop_lambda="${15}"
  local anchor_bonus="${16}"

  local output_dir="${PHASE2_SWEEP_DIR}/${variant}"
  local checkpoint_dir="${output_dir}/checkpoints"
  local best_checkpoint="${checkpoint_dir}/best"
  local report_path="${EVAL_DIR}/eval_compare_report_${variant}.json"

  local args=(
    bash run_train_phase2_top20_delta_curriculum.sh
    --run-root "$RUN_ROOT"
    --output-dir "$output_dir"
    --metadata-path "$CURRICULUM_METADATA_PATH"
    --python-bin "$PYTHON_BIN"
    --model-name "$MODEL_NAME"
    --source-checkpoint "$PHASE2_INIT_ADAPTER_PATH"
    --num-epochs 1
    --max-steps 200
    --eval-every-steps 20
    --learning-rate "$lr"
    --kl-beta "$kl"
    --temperature "$temperature"
    --top-p "$top_p"
    --group-temperature-stride "$group_temperature_stride"
    --group-top-p-stride "$group_top_p_stride"
    --min-unique-final-queries "$min_unique"
    --max-regen-rounds "$max_regen"
    --reward-gap-threshold "$reward_gap"
    --reward-w-mrr "$reward_w_mrr"
    --reward-w-recall "$reward_w_recall"
    --reward-w-recall-dense "$reward_w_recall_dense"
    --reward-w-rank-bonus "$reward_w_rank"
    --recall-drop-lambda "$recall_drop_lambda"
    --anchor-bonus-value "$anchor_bonus"
  )

  append_optional_query_caps args

  echo "[phase2] variant=${variant} adapter=${PHASE2_INIT_ADAPTER_PATH}"
  "${args[@]}"

  if [[ ! -d "$best_checkpoint" ]]; then
    echo "[error] phase2 variant ${variant} best checkpoint not found: $best_checkpoint" >&2
    exit 1
  fi

  echo "[eval] variant=${variant} adapter=${best_checkpoint} query_batch_size=8"
  "$PYTHON_BIN" eval_compare.py \
    --rl-adapter-path "$best_checkpoint" \
    --model-name "$MODEL_NAME" \
    --strict-tokenizer-model-match \
    --max-eval-queries "$MAX_EVAL_QUERIES" \
    --max-new-tokens 10 \
    --temperature 0.80 \
    --top-p 0.92 \
    --query-batch-size 8 \
    --sample-print 20 \
    --report-path "$report_path"

  printf '{"variant":"%s","checkpoint_path":"%s","report_path":"%s"}\n' \
    "$variant" "$best_checkpoint" "$report_path" >> "$MANIFEST_PATH"
}

generate_sweep_summary() {
  "$PYTHON_BIN" - "$MANIFEST_PATH" "$SWEEP_SUMMARY_PATH" "$SWEEP_LEADERBOARD_CSV" "$TARGET_DELTA_VS_ZERO" <<'PY'
import csv
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
csv_path = Path(sys.argv[3])
target_delta = float(sys.argv[4])

entries = []
for raw in manifest_path.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    manifest_row = json.loads(raw)
    report = json.loads(Path(manifest_row["report_path"]).read_text(encoding="utf-8"))
    rl = report.get("rl", {})
    zero = report.get("zero_shot", {})
    original = report.get("original", {})
    deltas = report.get("deltas", {})

    rl_minus_zero = float(deltas.get("rl_minus_zero", float(rl.get("mrr", 0.0)) - float(zero.get("mrr", 0.0))))
    rl_minus_original = float(
        deltas.get("rl_minus_original", float(rl.get("mrr", 0.0)) - float(original.get("mrr", 0.0)))
    )
    rl_recall20 = float(rl.get("recall", rl.get("recall_mean", 0.0)))
    meets_target = rl_minus_zero >= target_delta

    entries.append(
        {
            "variant": manifest_row["variant"],
            "checkpoint_path": manifest_row["checkpoint_path"],
            "report_path": manifest_row["report_path"],
            "rl_minus_zero_mrr": rl_minus_zero,
            "rl_minus_original_mrr": rl_minus_original,
            "rl_recall20": rl_recall20,
            "meets_target": meets_target,
        }
    )

entries.sort(
    key=lambda item: (item["rl_minus_zero_mrr"], item["rl_minus_original_mrr"], item["rl_recall20"], item["variant"]),
    reverse=True,
)

winner = entries[0] if entries else None
summary = {
    "target_delta_vs_zero": target_delta,
    "leaderboard": entries,
    "winner": winner,
    "winner_meets_target": bool(winner and winner["meets_target"]),
}

summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "variant",
            "rl_minus_zero_mrr",
            "rl_minus_original_mrr",
            "rl_recall20",
            "meets_target",
            "checkpoint_path",
            "report_path",
        ],
    )
    writer.writeheader()
    for row in entries:
        writer.writerow(row)
PY
}

run_phase1

PHASE2_INIT_ADAPTER_PATH="${PHASE1_CHECKPOINT_DIR}/${PHASE2_START_FROM}"
if [[ ! -d "$PHASE2_INIT_ADAPTER_PATH" ]]; then
  echo "[error] phase1 ${PHASE2_START_FROM} checkpoint not found: $PHASE2_INIT_ADAPTER_PATH" >&2
  exit 1
fi

run_phase2_variant "mrr_strict" 6e-6 0.045 0.82 0.93 0.10 0.025 5 3 0.05 0.70 0.16 0.08 0.06 1.00 0.00
run_phase2_variant "mrr_balanced" 5e-6 0.050 0.84 0.94 0.10 0.025 5 3 0.04 0.62 0.20 0.10 0.08 1.20 0.00
run_phase2_variant "mrr_diverse" 7e-6 0.040 0.88 0.96 0.14 0.035 6 4 0.03 0.66 0.18 0.10 0.06 1.00 0.00

generate_sweep_summary

echo "[done] sweep summary: $SWEEP_SUMMARY_PATH"
echo "[done] sweep leaderboard csv: $SWEEP_LEADERBOARD_CSV"
"$PYTHON_BIN" - "$SWEEP_SUMMARY_PATH" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
winner = summary.get("winner")
if not winner:
    print("[result] no phase2 variants were recorded")
    raise SystemExit(0)

status = "met" if summary.get("winner_meets_target") else "not_met"
print(
    "[result] winner={variant} rl_minus_zero={delta:+.4f} rl_minus_original={orig:+.4f} "
    "rl_recall20={recall:.4f} target_status={status}".format(
        variant=winner["variant"],
        delta=float(winner["rl_minus_zero_mrr"]),
        orig=float(winner["rl_minus_original_mrr"]),
        recall=float(winner["rl_recall20"]),
        status=status,
    )
)
print(f"[result] checkpoint={winner['checkpoint_path']}")
print(f"[result] report={winner['report_path']}")
PY
