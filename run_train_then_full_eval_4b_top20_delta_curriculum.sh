#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-train_and_eval_data_model_0420}"
EXP_NAME="${EXP_NAME:-4b_top20_delta_curriculum}"
LOG_DIR="${LOG_DIR:-log}"
MODEL_NAME="${MODEL_NAME:-/root/autodl-tmp/hf_models/Qwen3-4B-Instruct-2507}"
SEARCH_THREADS="${SEARCH_THREADS:-16}"
TRAIN_MAX_VAL_QUERIES="${TRAIN_MAX_VAL_QUERIES:-400}"
EVAL_QUERY_BATCH_SIZE="${EVAL_QUERY_BATCH_SIZE:-16}"
PHASE1_BATCH_SIZE="${PHASE1_BATCH_SIZE:-24}"
PHASE1_GROUP_SIZE="${PHASE1_GROUP_SIZE:-8}"
PHASE1_MAX_GROUP_SIZE="${PHASE1_MAX_GROUP_SIZE:-12}"
PHASE1_LR="${PHASE1_LR:-1.0e-5}"
PHASE1_KL_BETA="${PHASE1_KL_BETA:-0.045}"
PHASE1_TEMPERATURE="${PHASE1_TEMPERATURE:-0.75}"
PHASE1_TOP_P="${PHASE1_TOP_P:-0.92}"
PHASE1_MAX_NEW_TOKENS="${PHASE1_MAX_NEW_TOKENS:-10}"
PHASE1_EVAL_EVERY_STEPS="${PHASE1_EVAL_EVERY_STEPS:-20}"
PHASE1_MAX_STEPS="${PHASE1_MAX_STEPS:-80}"
PHASE2_BATCH_SIZE="${PHASE2_BATCH_SIZE:-24}"
PHASE2_GROUP_SIZE="${PHASE2_GROUP_SIZE:-8}"
PHASE2_MAX_GROUP_SIZE="${PHASE2_MAX_GROUP_SIZE:-12}"
PHASE2_LR="${PHASE2_LR:-8e-6}"
PHASE2_KL_BETA="${PHASE2_KL_BETA:-0.05}"
PHASE2_TEMPERATURE="${PHASE2_TEMPERATURE:-0.70}"
PHASE2_TOP_P="${PHASE2_TOP_P:-0.90}"
PHASE2_MAX_NEW_TOKENS="${PHASE2_MAX_NEW_TOKENS:-10}"
PHASE2_EVAL_EVERY_STEPS="${PHASE2_EVAL_EVERY_STEPS:-20}"
PHASE2_MAX_STEPS="${PHASE2_MAX_STEPS:-60}"
ACTOR_CHUNK_SIZE="${ACTOR_CHUNK_SIZE:-4}"
PROJECTION_CHUNK_SIZE="${PROJECTION_CHUNK_SIZE:-64}"
GROUP_TEMPERATURE_STRIDE="${GROUP_TEMPERATURE_STRIDE:-0.07}"
GROUP_TOP_P_STRIDE="${GROUP_TOP_P_STRIDE:-0.015}"
MIN_UNIQUE_FINAL_QUERIES="${MIN_UNIQUE_FINAL_QUERIES:-4}"
MAX_REGEN_ROUNDS="${MAX_REGEN_ROUNDS:-2}"
REWARD_GAP_THRESHOLD="${REWARD_GAP_THRESHOLD:-0.08}"
GAP_SAMPLING_TEMPERATURE_DELTA="${GAP_SAMPLING_TEMPERATURE_DELTA:-0.15}"
FORMAT_MAX_TOKENS="${FORMAT_MAX_TOKENS:-12}"
FORMAT_MIN_ENGLISH_RATIO="${FORMAT_MIN_ENGLISH_RATIO:-0.8}"
FORMAT_MAX_UNREADABLE_RATIO="${FORMAT_MAX_UNREADABLE_RATIO:-0.25}"

RUN_ROOT="${ARTIFACT_ROOT}/artifacts_${EXP_NAME}"
PHASE1_DIR="${RUN_ROOT}/phase1"
PHASE2_DIR="${RUN_ROOT}/phase2"
EVAL_DIR="${RUN_ROOT}/eval"
CURRICULUM_METADATA_PATH="${RUN_ROOT}/curriculum_query_metadata.jsonl"
PHASE1_CHECKPOINT_DIR="${PHASE1_DIR}/checkpoints"
PHASE2_CHECKPOINT_DIR="${PHASE2_DIR}/checkpoints"
PHASE1_LOG_PATH="${PHASE1_DIR}/train_log.jsonl"
PHASE2_LOG_PATH="${PHASE2_DIR}/train_log.jsonl"
PHASE1_TRACE_PATH="${PHASE1_DIR}/group_trace_log.jsonl"
PHASE2_TRACE_PATH="${PHASE2_DIR}/group_trace_log.jsonl"
EVAL_REPORT_PATH="${EVAL_DIR}/eval_compare_report_full.json"

mkdir -p "$PHASE1_DIR" "$PHASE2_DIR" "$EVAL_DIR" "$LOG_DIR"

RUN_TS="$(date '+%Y%m%d_%H%M%S')"
RUN_LOG_PATH="${LOG_DIR}/run_train_then_full_eval_4b_top20_delta_curriculum_${RUN_TS}.log"

if [[ -z "${RUN_LOG_REDIRECTED:-}" ]]; then
  export RUN_LOG_REDIRECTED=1
  exec > >(tee -a "$RUN_LOG_PATH") 2>&1
fi

echo "[log] command output is also saved to: $RUN_LOG_PATH"
echo "[env] python: $("$PYTHON_BIN" --version 2>&1)"
echo "[env] model: $MODEL_NAME"
echo "[env] search_threads: $SEARCH_THREADS"
echo "[env] curriculum_metadata: $CURRICULUM_METADATA_PATH"

COMMON_TRAIN_ARGS=(
  --model-name "$MODEL_NAME"
  --num-epochs 1
  --clip-range 0.2
  --ref-precision-mode 4bit
  --search-threads "$SEARCH_THREADS"
  --eval-query-batch-size "$EVAL_QUERY_BATCH_SIZE"
  --group-temperature-stride "$GROUP_TEMPERATURE_STRIDE"
  --group-top-p-stride "$GROUP_TOP_P_STRIDE"
  --min-unique-final-queries "$MIN_UNIQUE_FINAL_QUERIES"
  --max-regen-rounds "$MAX_REGEN_ROUNDS"
  --reward-gap-threshold "$REWARD_GAP_THRESHOLD"
  --gap-sampling-temperature-delta "$GAP_SAMPLING_TEMPERATURE_DELTA"
  --actor-chunk-size "$ACTOR_CHUNK_SIZE"
  --projection-chunk-size "$PROJECTION_CHUNK_SIZE"
  --max-val-queries "$TRAIN_MAX_VAL_QUERIES"
  --reward-mode top20_delta
  --reward-mrr-k 20
  --reward-recall-k 20
  --reward-recall-dense-k 50
  --reward-w-bad-format 0.18
  --reward-w-unsafe-copy 0.12
  --reward-w-overedit 0.10
  --overedit-tau 0.40
  --format-max-tokens "$FORMAT_MAX_TOKENS"
  --format-min-english-ratio "$FORMAT_MIN_ENGLISH_RATIO"
  --format-max-unreadable-ratio "$FORMAT_MAX_UNREADABLE_RATIO"
  --curriculum-enable
  --curriculum-metadata-path "$CURRICULUM_METADATA_PATH"
)

echo "[phase1] batch=${PHASE1_BATCH_SIZE} group=${PHASE1_GROUP_SIZE} max_group=${PHASE1_MAX_GROUP_SIZE} lr=${PHASE1_LR} kl=${PHASE1_KL_BETA} decode=(${PHASE1_MAX_NEW_TOKENS},${PHASE1_TEMPERATURE},${PHASE1_TOP_P}) reward=(0.55,0.20,0.15,0.10)"
"$PYTHON_BIN" train.py \
  "${COMMON_TRAIN_ARGS[@]}" \
  --curriculum-phase phase1 \
  --batch-size "$PHASE1_BATCH_SIZE" \
  --group-size "$PHASE1_GROUP_SIZE" \
  --max-group-size "$PHASE1_MAX_GROUP_SIZE" \
  --learning-rate "$PHASE1_LR" \
  --kl-beta "$PHASE1_KL_BETA" \
  --max-new-tokens "$PHASE1_MAX_NEW_TOKENS" \
  --temperature "$PHASE1_TEMPERATURE" \
  --top-p "$PHASE1_TOP_P" \
  --eval-max-new-tokens "$PHASE1_MAX_NEW_TOKENS" \
  --eval-temperature "$PHASE1_TEMPERATURE" \
  --eval-top-p "$PHASE1_TOP_P" \
  --eval-every-steps "$PHASE1_EVAL_EVERY_STEPS" \
  --max-steps "$PHASE1_MAX_STEPS" \
  --reward-w-mrr 0.55 \
  --reward-w-recall 0.20 \
  --reward-w-recall-dense 0.15 \
  --reward-w-rank-bonus 0.10 \
  --save-dir "$PHASE1_CHECKPOINT_DIR" \
  --log-path "$PHASE1_LOG_PATH" \
  --group-trace-log-path "$PHASE1_TRACE_PATH"

PHASE1_BEST="${PHASE1_CHECKPOINT_DIR}/best"
if [[ ! -d "$PHASE1_BEST" ]]; then
  echo "[error] phase1 best checkpoint not found: $PHASE1_BEST" >&2
  exit 1
fi

echo "[phase2] batch=${PHASE2_BATCH_SIZE} group=${PHASE2_GROUP_SIZE} max_group=${PHASE2_MAX_GROUP_SIZE} lr=${PHASE2_LR} kl=${PHASE2_KL_BETA} decode=(${PHASE2_MAX_NEW_TOKENS},${PHASE2_TEMPERATURE},${PHASE2_TOP_P}) reward=(0.65,0.15,0.10,0.10)"
"$PYTHON_BIN" train.py \
  "${COMMON_TRAIN_ARGS[@]}" \
  --curriculum-phase phase2 \
  --adapter-path "$PHASE1_BEST" \
  --batch-size "$PHASE2_BATCH_SIZE" \
  --group-size "$PHASE2_GROUP_SIZE" \
  --max-group-size "$PHASE2_MAX_GROUP_SIZE" \
  --learning-rate "$PHASE2_LR" \
  --kl-beta "$PHASE2_KL_BETA" \
  --max-new-tokens "$PHASE2_MAX_NEW_TOKENS" \
  --temperature "$PHASE2_TEMPERATURE" \
  --top-p "$PHASE2_TOP_P" \
  --eval-max-new-tokens "$PHASE2_MAX_NEW_TOKENS" \
  --eval-temperature "$PHASE2_TEMPERATURE" \
  --eval-top-p "$PHASE2_TOP_P" \
  --eval-every-steps "$PHASE2_EVAL_EVERY_STEPS" \
  --max-steps "$PHASE2_MAX_STEPS" \
  --reward-w-mrr 0.65 \
  --reward-w-recall 0.15 \
  --reward-w-recall-dense 0.10 \
  --reward-w-rank-bonus 0.10 \
  --save-dir "$PHASE2_CHECKPOINT_DIR" \
  --log-path "$PHASE2_LOG_PATH" \
  --group-trace-log-path "$PHASE2_TRACE_PATH"

PHASE2_BEST="${PHASE2_CHECKPOINT_DIR}/best"
if [[ ! -d "$PHASE2_BEST" ]]; then
  echo "[error] phase2 best checkpoint not found: $PHASE2_BEST" >&2
  exit 1
fi

echo "[eval] adapter=${PHASE2_BEST} query_batch_size=${EVAL_QUERY_BATCH_SIZE}"
"$PYTHON_BIN" eval_compare.py \
  --rl-adapter-path "$PHASE2_BEST" \
  --model-name "$MODEL_NAME" \
  --strict-tokenizer-model-match \
  --max-eval-queries 1000000 \
  --search-threads "$SEARCH_THREADS" \
  --max-new-tokens "$PHASE2_MAX_NEW_TOKENS" \
  --temperature "$PHASE2_TEMPERATURE" \
  --top-p "$PHASE2_TOP_P" \
  --query-batch-size "$EVAL_QUERY_BATCH_SIZE" \
  --reward-mode top20_delta \
  --reward-mrr-k 20 \
  --reward-recall-k 20 \
  --reward-recall-dense-k 50 \
  --reward-w-mrr 0.65 \
  --reward-w-recall 0.15 \
  --reward-w-recall-dense 0.10 \
  --reward-w-rank-bonus 0.10 \
  --reward-w-bad-format 0.18 \
  --reward-w-unsafe-copy 0.12 \
  --reward-w-overedit 0.10 \
  --overedit-tau 0.40 \
  --format-max-tokens "$FORMAT_MAX_TOKENS" \
  --format-min-english-ratio "$FORMAT_MIN_ENGLISH_RATIO" \
  --format-max-unreadable-ratio "$FORMAT_MAX_UNREADABLE_RATIO" \
  --sample-print 20 \
  --report-path "$EVAL_REPORT_PATH"

echo "[done] report: $EVAL_REPORT_PATH"
