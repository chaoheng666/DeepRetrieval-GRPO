#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[1/2] Training 3B experiment..."
"$PYTHON_BIN" train.py \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --num-epochs 3 \
  --batch-size 4 \
  --group-size 8 \
  --max-new-tokens 20 \
  --temperature 0.7 \
  --top-p 0.9 \
  --save-dir artifacts_3b_train/checkpoints \
  --log-path artifacts_3b_train/train_log.jsonl \
  --group-trace-log-path artifacts_3b_train/group_trace_log.jsonl

BEST_ADAPTER="artifacts_3b_train/checkpoints/best"
LATEST_ADAPTER="artifacts_3b_train/checkpoints/latest"

if [[ -d "$BEST_ADAPTER" ]]; then
  ADAPTER_PATH="$BEST_ADAPTER"
elif [[ -d "$LATEST_ADAPTER" ]]; then
  ADAPTER_PATH="$LATEST_ADAPTER"
  echo "[warn] best adapter not found, fallback to latest adapter."
else
  echo "[error] No adapter found under artifacts_3b/checkpoints (expected best or latest)." >&2
  exit 1
fi

echo "[2/2] Running full evaluation for the 3B experiment..."
"$PYTHON_BIN" eval_compare.py \
  --rl-adapter-path "$ADAPTER_PATH" \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --strict-tokenizer-model-match \
  --max-eval-queries 1000000 \
  --sample-print 20 \
  --report-path artifacts_3b_eval/eval_compare_report_full.json

echo "Done. Report: artifacts_3b_eval/eval_compare_report_full.json"

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
  echo "[3/3] Commit created: $COMMIT_MSG"
  git push
fi
