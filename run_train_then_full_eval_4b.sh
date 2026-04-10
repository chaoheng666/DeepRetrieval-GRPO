#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-train_and_eval_data_model}"
EXP_NAME="${EXP_NAME:-4b}"

TRAIN_DIR="${ARTIFACT_ROOT}/artifacts_${EXP_NAME}_train"
EVAL_DIR="${ARTIFACT_ROOT}/artifacts_${EXP_NAME}_eval"
TRAIN_CHECKPOINT_DIR="${TRAIN_DIR}/checkpoints"
TRAIN_LOG_PATH="${TRAIN_DIR}/train_log.jsonl"
TRAIN_TRACE_PATH="${TRAIN_DIR}/group_trace_log.jsonl"
EVAL_REPORT_PATH="${EVAL_DIR}/eval_compare_report_full.json"

mkdir -p "$TRAIN_DIR" "$EVAL_DIR"

echo "[1/3] Training 4B experiment..."
"$PYTHON_BIN" train.py \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --num-epochs 3 \
  --batch-size 4 \
  --group-size 8 \
  --max-new-tokens 20 \
  --temperature 0.7 \
  --top-p 0.9 \
  --save-dir "$TRAIN_CHECKPOINT_DIR" \
  --log-path "$TRAIN_LOG_PATH" \
  --group-trace-log-path "$TRAIN_TRACE_PATH"

BEST_ADAPTER="${TRAIN_CHECKPOINT_DIR}/best"
LATEST_ADAPTER="${TRAIN_CHECKPOINT_DIR}/latest"

if [[ -d "$BEST_ADAPTER" ]]; then
  ADAPTER_PATH="$BEST_ADAPTER"
elif [[ -d "$LATEST_ADAPTER" ]]; then
  ADAPTER_PATH="$LATEST_ADAPTER"
  echo "[warn] best adapter not found, fallback to latest adapter."
else
  echo "[error] No adapter found under ${TRAIN_CHECKPOINT_DIR} (expected best or latest)." >&2
  exit 1
fi

echo "[2/3] Running full evaluation for the 4B experiment..."
"$PYTHON_BIN" eval_compare.py \
  --rl-adapter-path "$ADAPTER_PATH" \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --strict-tokenizer-model-match \
  --max-eval-queries 1000000 \
  --sample-print 20 \
  --report-path "$EVAL_REPORT_PATH"

echo "Done. Report: $EVAL_REPORT_PATH"

echo "[3/3] Auto-committing all changes to git repository..."
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[error] Current directory is not a git repository." >&2
  exit 1
fi

git add -A
if git diff --cached --quiet; then
  echo "[3/3] No staged changes to commit. Skipped."
else
  COMMIT_MSG="chore: auto commit training and full eval artifacts $(date '+%Y-%m-%d %H:%M:%S')"
  git commit -m "$COMMIT_MSG"
  git push
  echo "[3/3] Commit created and pushed: $COMMIT_MSG"
fi
