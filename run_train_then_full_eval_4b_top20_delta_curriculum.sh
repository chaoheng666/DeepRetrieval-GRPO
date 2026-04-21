#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="/root/miniconda3/bin/python"
LOG_DIR="log"
RUN_ROOT="train_and_eval_data_model_0421/artifacts_4b_top20_delta_curriculum"
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
echo "[env] model: /root/autodl-tmp/hf_models/Qwen3-4B-Instruct-2507"
echo "[env] search_threads: 16"
echo "[env] curriculum_metadata: $CURRICULUM_METADATA_PATH"

COMMON_TRAIN_ARGS=(
  --model-name "/root/autodl-tmp/hf_models/Qwen3-4B-Instruct-2507"
  --clip-range 0.2
  --ref-precision-mode 4bit
  --search-threads 16
  --eval-query-batch-size 8
  --group-temperature-stride 0.08
  --group-top-p-stride 0.02
  --min-unique-final-queries 5
  --max-regen-rounds 3
  --reward-gap-threshold 0.12
  --gap-sampling-temperature-delta 0.18
  --actor-chunk-size 2
  --projection-chunk-size 64
  --max-val-queries 400
  --reward-mode top20_delta
  --reward-mrr-k 20
  --reward-recall-k 20
  --reward-recall-dense-k 50
  --reward-w-bad-format 0.18
  --reward-w-unsafe-copy 0.14
  --reward-w-overedit 0.08
  --overedit-tau 0.45
  --recall-drop-lambda 0.8
  --anchor-bonus-value 0.05
  --format-max-tokens 12
  --format-min-english-ratio 0.85
  --format-max-unreadable-ratio 0.20
  --curriculum-enable
  --curriculum-metadata-path "$CURRICULUM_METADATA_PATH"
)

echo "[phase1] epochs=2 batch=24 group=8 max_group=12 lr=1.0e-5 kl=0.040 decode=(10,0.82,0.93) reward=(0.40,0.28,0.22,0.10)"
"$PYTHON_BIN" train.py \
  "${COMMON_TRAIN_ARGS[@]}" \
  --curriculum-phase phase1 \
  --num-epochs 2 \
  --batch-size 24 \
  --group-size 8 \
  --max-group-size 12 \
  --learning-rate 1.0e-5 \
  --kl-beta 0.040 \
  --max-new-tokens 10 \
  --temperature 0.82 \
  --top-p 0.93 \
  --eval-max-new-tokens 10 \
  --eval-temperature 0.82 \
  --eval-top-p 0.93 \
  --eval-every-steps 20 \
  --max-steps 100 \
  --reward-w-mrr 0.40 \
  --reward-w-recall 0.28 \
  --reward-w-recall-dense 0.22 \
  --reward-w-rank-bonus 0.10 \
  --save-dir "$PHASE1_CHECKPOINT_DIR" \
  --log-path "$PHASE1_LOG_PATH" \
  --group-trace-log-path "$PHASE1_TRACE_PATH"

PHASE1_BEST="${PHASE1_CHECKPOINT_DIR}/best"
if [[ ! -d "$PHASE1_BEST" ]]; then
  echo "[error] phase1 best checkpoint not found: $PHASE1_BEST" >&2
  exit 1
fi

echo "[phase2] epochs=1 batch=24 group=8 max_group=12 lr=6.0e-6 kl=0.055 decode=(10,0.80,0.92) reward=(0.52,0.22,0.16,0.10)"
"$PYTHON_BIN" train.py \
  "${COMMON_TRAIN_ARGS[@]}" \
  --curriculum-phase phase2 \
  --adapter-path "$PHASE1_BEST" \
  --num-epochs 1 \
  --batch-size 24 \
  --group-size 8 \
  --max-group-size 12 \
  --learning-rate 6.0e-6 \
  --kl-beta 0.055 \
  --max-new-tokens 10 \
  --temperature 0.80 \
  --top-p 0.92 \
  --eval-max-new-tokens 10 \
  --eval-temperature 0.80 \
  --eval-top-p 0.92 \
  --eval-every-steps 20 \
  --max-steps 60 \
  --reward-w-mrr 0.52 \
  --reward-w-recall 0.22 \
  --reward-w-recall-dense 0.16 \
  --reward-w-rank-bonus 0.10 \
  --save-dir "$PHASE2_CHECKPOINT_DIR" \
  --log-path "$PHASE2_LOG_PATH" \
  --group-trace-log-path "$PHASE2_TRACE_PATH"

PHASE2_BEST="${PHASE2_CHECKPOINT_DIR}/best"
if [[ ! -d "$PHASE2_BEST" ]]; then
  echo "[error] phase2 best checkpoint not found: $PHASE2_BEST" >&2
  exit 1
fi

echo "[eval] adapter=${PHASE2_BEST} query_batch_size=8"
"$PYTHON_BIN" eval_compare.py \
  --rl-adapter-path "$PHASE2_BEST" \
  --model-name "/root/autodl-tmp/hf_models/Qwen3-4B-Instruct-2507" \
  --strict-tokenizer-model-match \
  --max-eval-queries 1000000 \
  --search-threads 16 \
  --max-new-tokens 10 \
  --temperature 0.80 \
  --top-p 0.92 \
  --query-batch-size 8 \
  --reward-mode top20_delta \
  --reward-mrr-k 20 \
  --reward-recall-k 20 \
  --reward-recall-dense-k 50 \
  --reward-w-mrr 0.52 \
  --reward-w-recall 0.22 \
  --reward-w-recall-dense 0.16 \
  --reward-w-rank-bonus 0.10 \
  --reward-w-bad-format 0.18 \
  --reward-w-unsafe-copy 0.14 \
  --reward-w-overedit 0.08 \
  --overedit-tau 0.45 \
  --recall-drop-lambda 0.8 \
  --anchor-bonus-value 0.05 \
  --format-max-tokens 12 \
  --format-min-english-ratio 0.85 \
  --format-max-unreadable-ratio 0.20 \
  --sample-print 20 \
  --report-path "$EVAL_REPORT_PATH"

echo "[done] report: $EVAL_REPORT_PATH"
