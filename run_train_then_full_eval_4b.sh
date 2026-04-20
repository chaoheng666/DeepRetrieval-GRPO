#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-train_and_eval_data_model_0420}"
EXP_NAME="${EXP_NAME:-4b_conservative_mrr}"
VENV_DIR="${VENV_DIR:-.venv}"
LOG_DIR="${LOG_DIR:-log}"
USE_VENV="${USE_VENV:-0}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
MODEL_NAME="${MODEL_NAME:-/root/autodl-tmp/hf_models/Qwen3-4B-Instruct-2507}"
AUTO_GIT_COMMIT="${AUTO_GIT_COMMIT:-1}"
AUTO_GIT_PUSH="${AUTO_GIT_PUSH:-1}"
TRAIN_MAX_VAL_QUERIES="${TRAIN_MAX_VAL_QUERIES:-400}"
SEARCH_THREADS="${SEARCH_THREADS:-16}"
TRAIN_MAX_NEW_TOKENS="${TRAIN_MAX_NEW_TOKENS:-12}"
TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-0.85}"
TRAIN_TOP_P="${TRAIN_TOP_P:-0.95}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-$TRAIN_MAX_NEW_TOKENS}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-$TRAIN_TEMPERATURE}"
EVAL_TOP_P="${EVAL_TOP_P:-$TRAIN_TOP_P}"
EVAL_QUERY_BATCH_SIZE="${EVAL_QUERY_BATCH_SIZE:-8}"

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

echo "[preset] 4B tuned config: batch=24 group=10 max_group=14 train_decode=(${TRAIN_MAX_NEW_TOKENS},${TRAIN_TEMPERATURE},${TRAIN_TOP_P}) eval_decode=(${EVAL_MAX_NEW_TOKENS},${EVAL_TEMPERATURE},${EVAL_TOP_P}) eval_query_batch_size=${EVAL_QUERY_BATCH_SIZE} group_temperature_stride=0.07 group_top_p_stride=0.015 min_unique_final_queries=4 max_regen_rounds=2 reward_gap_threshold=0.08 gap_sampling_temperature_delta=0.15 reward=(mrr@10, recall@50, recall_dense@100) max_val_queries=${TRAIN_MAX_VAL_QUERIES} search_threads=${SEARCH_THREADS}"

echo "[2/4] Training 4B experiment..."
"$PYTHON_BIN" train.py \
  --model-name "$MODEL_NAME" \
  --num-epochs 1 \
  --batch-size 24 \
  --group-size 12 \
  --max-group-size 16 \
  --learning-rate 1.5e-5 \
  --clip-range 0.2 \
  --kl-beta 0.03 \
  --search-threads "$SEARCH_THREADS" \
  --max-new-tokens "$TRAIN_MAX_NEW_TOKENS" \
  --temperature "$TRAIN_TEMPERATURE" \
  --top-p "$TRAIN_TOP_P" \
  --eval-max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
  --eval-temperature "$EVAL_TEMPERATURE" \
  --eval-top-p "$EVAL_TOP_P" \
  --eval-query-batch-size "$EVAL_QUERY_BATCH_SIZE" \
  --group-temperature-stride 0.08 \
  --group-top-p-stride 0.015 \
  --min-unique-final-queries 4 \
  --max-regen-rounds 2 \
  --reward-gap-threshold 0.08 \
  --gap-sampling-temperature-delta 0.15 \
  --eval-every-steps 50 \
  --max-val-queries "$TRAIN_MAX_VAL_QUERIES" \
  --max-steps 500 \
  --reward-mrr-k 50 \
  --reward-recall-k 50 \
  --reward-recall-dense-k 100 \
  --reward-w-mrr 0.40 \
  --reward-w-recall 0.20 \
  --reward-w-recall-dense 0.15 \
  --reward-w-term-preserve 0.10 \
  --reward-w-length-score 0.08 \
  --reward-w-clean-format 0.07 \
  --reward-w-bad-format 0.15 \
  --reward-w-unsafe-copy 0.08 \
  --format-max-tokens 12 \
  --format-min-english-ratio 0.8 \
  --format-max-unreadable-ratio 0.25 \
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
  --search-threads "$SEARCH_THREADS" \
  --max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
  --temperature "$EVAL_TEMPERATURE" \
  --top-p "$EVAL_TOP_P" \
  --query-batch-size "$EVAL_QUERY_BATCH_SIZE" \
  --reward-mrr-k 50 \
  --reward-recall-k 50 \
  --reward-recall-dense-k 100 \
  --reward-w-mrr 0.40 \
  --reward-w-recall 0.20 \
  --reward-w-recall-dense 0.15 \
  --reward-w-term-preserve 0.10 \
  --reward-w-length-score 0.08 \
  --reward-w-clean-format 0.07 \
  --reward-w-bad-format 0.15 \
  --reward-w-unsafe-copy 0.08 \
  --format-max-tokens 12 \
  --format-min-english-ratio 0.8 \
  --format-max-unreadable-ratio 0.25 \
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
