#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-train_and_eval_data_model}"
EXP_NAME="${EXP_NAME:-4b}"
VENV_DIR="${VENV_DIR:-.venv}"
LOG_DIR="${LOG_DIR:-log}"

TRAIN_DIR="${ARTIFACT_ROOT}/artifacts_${EXP_NAME}_train"
EVAL_DIR="${ARTIFACT_ROOT}/artifacts_${EXP_NAME}_eval"
TRAIN_CHECKPOINT_DIR="${TRAIN_DIR}/checkpoints"
TRAIN_LOG_PATH="${TRAIN_DIR}/train_log.jsonl"
TRAIN_TRACE_PATH="${TRAIN_DIR}/group_trace_log.jsonl"
EVAL_REPORT_PATH="${EVAL_DIR}/eval_compare_report_full.json"

mkdir -p "$TRAIN_DIR" "$EVAL_DIR" "$LOG_DIR"

RUN_TS="$(date '+%Y%m%d_%H%M%S')"
RUN_LOG_PATH="${LOG_DIR}/run_train_then_full_eval_4b_${RUN_TS}.log"

if [[ -z "${RUN_LOG_REDIRECTED:-}" ]]; then
  export RUN_LOG_REDIRECTED=1
  exec > >(tee -a "$RUN_LOG_PATH") 2>&1
fi

echo "[log] command output is also saved to: $RUN_LOG_PATH"

echo "[1/4] Preparing Python environment..."
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] Python command not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
PYTHON_BIN="${VENV_DIR}/bin/python"

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r requirements.txt

echo "[env] python: $("$PYTHON_BIN" --version 2>&1)"
echo "[env] pip: $("$PYTHON_BIN" -m pip --version)"

if command -v java >/dev/null 2>&1; then
  echo "[env] java: $(java -version 2>&1 | head -n 1)"
else
  echo "[warn] java not found. pyserini may fail without Java 21+."
fi

echo "[2/4] Training 4B experiment..."
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

echo "[3/4] Running full evaluation for the 4B experiment..."
"$PYTHON_BIN" eval_compare.py \
  --rl-adapter-path "$ADAPTER_PATH" \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --strict-tokenizer-model-match \
  --max-eval-queries 1000000 \
  --sample-print 20 \
  --report-path "$EVAL_REPORT_PATH"

echo "Done. Report: $EVAL_REPORT_PATH"

echo "[4/4] Auto-committing all changes to git repository..."
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[error] Current directory is not a git repository." >&2
  exit 1
fi

git add -A
if git diff --cached --quiet; then
  echo "[4/4] No staged changes to commit. Skipped."
else
  COMMIT_MSG="chore: auto commit training and full eval artifacts $(date '+%Y-%m-%d %H:%M:%S')"
  git commit -m "$COMMIT_MSG"
  git push
  echo "[4/4] Commit created and pushed: $COMMIT_MSG"
fi
