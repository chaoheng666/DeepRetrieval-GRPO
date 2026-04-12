#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-train_and_eval_data_model}"
EXP_NAME="${EXP_NAME:-4b}"
VENV_DIR="${VENV_DIR:-.venv}"
LOG_DIR="${LOG_DIR:-log}"
USE_VENV="${USE_VENV:-0}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
MODEL_NAME="${MODEL_NAME:-/root/autodl-tmp/hf_models/Qwen3-4B-Instruct-2507}"
AUTO_GIT_COMMIT="${AUTO_GIT_COMMIT:-1}"
AUTO_GIT_PUSH="${AUTO_GIT_PUSH:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
TRAIN_GROUP_SIZE="${TRAIN_GROUP_SIZE:-8}"
TRAIN_MAX_NEW_TOKENS="${TRAIN_MAX_NEW_TOKENS:-16}"
TRAIN_EVAL_EVERY_STEPS="${TRAIN_EVAL_EVERY_STEPS:-100}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-400}"
TRAIN_MAX_VAL_QUERIES="${TRAIN_MAX_VAL_QUERIES:-200}"

REWARD_MRR_K="${REWARD_MRR_K:-50}"
REWARD_RECALL_K="${REWARD_RECALL_K:-50}"
REWARD_W_MRR="${REWARD_W_MRR:-1.0}"
REWARD_W_RECALL="${REWARD_W_RECALL:-0.3}"
REWARD_W_COPY="${REWARD_W_COPY:-0.15}"
REWARD_W_FORMAT="${REWARD_W_FORMAT:-0.2}"
REWARD_COPY_TAU="${REWARD_COPY_TAU:-0.6}"
FORMAT_MAX_TOKENS="${FORMAT_MAX_TOKENS:-16}"
FORMAT_MIN_ENGLISH_RATIO="${FORMAT_MIN_ENGLISH_RATIO:-0.8}"
FORMAT_MAX_UNREADABLE_RATIO="${FORMAT_MAX_UNREADABLE_RATIO:-0.3}"

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

if [[ "$USE_VENV" == "1" ]]; then
  if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi

  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  PYTHON_BIN="${VENV_DIR}/bin/python"
fi

if [[ "$INSTALL_DEPS" == "1" ]]; then
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install -r requirements.txt
fi

echo "[env] python: $("$PYTHON_BIN" --version 2>&1)"
echo "[env] pip: $("$PYTHON_BIN" -m pip --version)"
echo "[env] executable: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "[env] torch cuda available: $("$PYTHON_BIN" -c 'import torch; print(torch.cuda.is_available())')"
is_likely_local_path=0
case "$MODEL_NAME" in
  /*|./*|../*|~/*)
    is_likely_local_path=1
    ;;
esac

if [[ "$is_likely_local_path" == "1" ]]; then
  if [[ ! -d "$MODEL_NAME" ]]; then
    echo "[error] MODEL_NAME local directory not found: $MODEL_NAME" >&2
    echo "[hint] Download model first, or set MODEL_NAME to an existing local directory/model id." >&2
    exit 1
  fi
  echo "[env] model source (local dir): $MODEL_NAME"
elif [[ -d "$MODEL_NAME" ]]; then
  echo "[env] model source (local dir): $MODEL_NAME"
else
  echo "[env] model source (model id): $MODEL_NAME"
fi

if command -v java >/dev/null 2>&1; then
  echo "[env] java: $(java -version 2>&1 | head -n 1)"
else
  echo "[warn] java not found. pyserini may fail without Java 21+."
fi

if [[ "$REQUIRE_CUDA" == "1" ]]; then
  "$PYTHON_BIN" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("[error] CUDA is unavailable in selected python env. Set REQUIRE_CUDA=0 to bypass.")
print(f"[env] cuda device count: {torch.cuda.device_count()}")
PY
fi

echo "[2/4] Training 4B experiment..."
"$PYTHON_BIN" train.py \
  --model-name "$MODEL_NAME" \
  --num-epochs 1 \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --group-size "$TRAIN_GROUP_SIZE" \
  --search-threads 8 \
  --max-new-tokens "$TRAIN_MAX_NEW_TOKENS" \
  --eval-every-steps "$TRAIN_EVAL_EVERY_STEPS" \
  --max-val-queries "$TRAIN_MAX_VAL_QUERIES" \
  --max-steps "$TRAIN_MAX_STEPS" \
  --reward-mrr-k "$REWARD_MRR_K" \
  --reward-recall-k "$REWARD_RECALL_K" \
  --reward-w-mrr "$REWARD_W_MRR" \
  --reward-w-recall "$REWARD_W_RECALL" \
  --reward-w-copy "$REWARD_W_COPY" \
  --reward-w-format "$REWARD_W_FORMAT" \
  --reward-copy-tau "$REWARD_COPY_TAU" \
  --format-max-tokens "$FORMAT_MAX_TOKENS" \
  --format-min-english-ratio "$FORMAT_MIN_ENGLISH_RATIO" \
  --format-max-unreadable-ratio "$FORMAT_MAX_UNREADABLE_RATIO" \
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
  --model-name "$MODEL_NAME" \
  --strict-tokenizer-model-match \
  --max-eval-queries 1000000 \
  --reward-mrr-k "$REWARD_MRR_K" \
  --reward-recall-k "$REWARD_RECALL_K" \
  --reward-w-mrr "$REWARD_W_MRR" \
  --reward-w-recall "$REWARD_W_RECALL" \
  --reward-w-copy "$REWARD_W_COPY" \
  --reward-w-format "$REWARD_W_FORMAT" \
  --reward-copy-tau "$REWARD_COPY_TAU" \
  --format-max-tokens "$FORMAT_MAX_TOKENS" \
  --format-min-english-ratio "$FORMAT_MIN_ENGLISH_RATIO" \
  --format-max-unreadable-ratio "$FORMAT_MAX_UNREADABLE_RATIO" \
  --sample-print 20 \
  --report-path "$EVAL_REPORT_PATH"

echo "Done. Report: $EVAL_REPORT_PATH"

echo "[4/4] Auto-committing all changes to git repository..."
if [[ "$AUTO_GIT_COMMIT" != "1" ]]; then
  echo "[4/4] AUTO_GIT_COMMIT=0, skipped."
else
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
    if [[ "$AUTO_GIT_PUSH" == "1" ]]; then
      if git push; then
        echo "[4/4] Commit created and pushed: $COMMIT_MSG"
      else
        echo "[warn] Commit created, but git push failed. Please push manually."
      fi
    else
      echo "[4/4] Commit created locally (AUTO_GIT_PUSH=0): $COMMIT_MSG"
    fi
  fi
fi
