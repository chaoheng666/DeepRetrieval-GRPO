#!/usr/bin/env bash
set -euo pipefail

# 主入口：只跑 Top20 Delta Curriculum 主线。
# 流程固定为 phase1 训练 -> phase2 从 phase1 best adapter 热启动 -> full eval。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 共享常量：路径、模型和 curriculum metadata 都在这里统一定义。
PYTHON_BIN="/root/miniconda3/bin/python"
MODEL_NAME="/root/autodl-tmp/hf_models/Qwen3-4B-Instruct-2507"
LOG_DIR="log"
RUN_ROOT="train_and_eval_data_model_0421/artifacts_4b_top20_delta_curriculum"
CURRICULUM_METADATA_PATH="${RUN_ROOT}/curriculum_query_metadata.jsonl"

PHASE1_DIR="${RUN_ROOT}/phase1"
PHASE2_DIR="${RUN_ROOT}/phase2"
EVAL_DIR="${RUN_ROOT}/eval"
PHASE1_CHECKPOINT_DIR="${PHASE1_DIR}/checkpoints"
PHASE2_CHECKPOINT_DIR="${PHASE2_DIR}/checkpoints"
EVAL_REPORT_PATH="${EVAL_DIR}/eval_compare_report_full.json"

PHASE1_ADAPTER_PATH="${PHASE1_ADAPTER_PATH:-${PHASE1_CHECKPOINT_PATH:-}}"
PHASE2_START_FROM="${PHASE2_START_FROM:-best}"

usage() {
  cat <<'EOF'
Usage:
  bash run_train_then_full_eval_4b_top20_delta_curriculum.sh [options]

Options:
  --phase1-adapter-path PATH     Optional checkpoint adapter to warm-start phase1.
  --phase1-checkpoint-path PATH  Alias of --phase1-adapter-path.
  --phase2-start-from best|latest
                                  Choose which phase1 checkpoint phase2 starts from.
                                  Default: best
  -h, --help                      Show this help.

Env overrides are also supported:
  PHASE1_ADAPTER_PATH=/path/to/checkpoints/latest
  PHASE1_CHECKPOINT_PATH=/path/to/checkpoints/latest
  PHASE2_START_FROM=best|latest
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase1-adapter-path|--phase1-checkpoint-path)
      if [[ $# -lt 2 ]]; then
        echo "[error] $1 requires a path" >&2
        exit 1
      fi
      PHASE1_ADAPTER_PATH="$2"
      shift 2
      ;;
    --phase2-start-from)
      if [[ $# -lt 2 ]]; then
        echo "[error] --phase2-start-from requires best or latest" >&2
        exit 1
      fi
      PHASE2_START_FROM="$2"
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

mkdir -p "$LOG_DIR" "$PHASE1_DIR" "$PHASE2_DIR" "$EVAL_DIR"

# 整个 bash 的 stdout/stderr 都复制到 log，方便断线后复盘。
RUN_TS="$(date '+%Y%m%d_%H%M%S')"
RUN_LOG_PATH="${LOG_DIR}/run_train_then_full_eval_4b_top20_delta_curriculum_${RUN_TS}.log"
if [[ -z "${RUN_LOG_REDIRECTED:-}" ]]; then
  export RUN_LOG_REDIRECTED=1
  exec > >(tee -a "$RUN_LOG_PATH") 2>&1
fi

echo "[log] command output is also saved to: $RUN_LOG_PATH"
echo "[env] python: $("$PYTHON_BIN" --version 2>&1)"
echo "[env] model: $MODEL_NAME"
echo "[env] curriculum_metadata: $CURRICULUM_METADATA_PATH"
echo "[env] phase1_adapter_path: ${PHASE1_ADAPTER_PATH:-<fresh>}"
echo "[env] phase2_start_from: phase1 ${PHASE2_START_FROM}"

# 单个 phase 的训练封装。phase1 不传 adapter；phase2 传 phase1 best adapter。
run_train_phase() {
  local phase="$1"
  local adapter_path="$2"
  local epochs="$3"
  local lr="$4"
  local kl="$5"
  local temperature="$6"
  local top_p="$7"
  local max_steps="$8"
  local w_mrr="$9"
  local w_recall="${10}"
  local w_recall_dense="${11}"
  local save_dir="${12}"
  local log_path="${13}"
  local trace_path="${14}"

  local args=(
    # train.py 已经默认 top20_delta/4bit/curriculum，这里只传 phase 差异和路径。
    --model-name "$MODEL_NAME"
    --curriculum-phase "$phase"
    --curriculum-metadata-path "$CURRICULUM_METADATA_PATH"
    --num-epochs "$epochs"
    --batch-size 24
    --group-size 8
    --max-group-size 12
    --learning-rate "$lr"
    --kl-beta "$kl"
    --max-new-tokens 10
    --temperature "$temperature"
    --top-p "$top_p"
    --eval-max-new-tokens 10
    --eval-temperature "$temperature"
    --eval-top-p "$top_p"
    --eval-query-batch-size 8
    --eval-every-steps 20
    --max-steps "$max_steps"
    --reward-w-mrr "$w_mrr"
    --reward-w-recall "$w_recall"
    --reward-w-recall-dense "$w_recall_dense"
    --reward-w-rank-bonus 0.10
    --save-dir "$save_dir"
    --log-path "$log_path"
    --group-trace-log-path "$trace_path"
  )

  if [[ -n "$adapter_path" ]]; then
    args+=(--adapter-path "$adapter_path")
  fi

  "$PYTHON_BIN" train.py "${args[@]}"
}

# phase1：更偏 recall 和探索，先让模型学会产生有效 top20 rewrite。
echo "[phase1] epochs=2 batch=24 group=8 max_group=12 lr=1.0e-5 kl=0.040 decode=(10,0.82,0.93) reward=(0.38,0.30,0.22,0.10)"
run_train_phase \
  phase1 "$PHASE1_ADAPTER_PATH" 2 1.0e-5 0.040 0.82 0.93 100 0.38 0.30 0.22 \
  "$PHASE1_CHECKPOINT_DIR" "${PHASE1_DIR}/train_log.jsonl" "${PHASE1_DIR}/group_trace_log.jsonl"

# phase2 必须从 phase1 best 热启动；如果没有 best，说明 phase1 没有完成可用训练。
PHASE2_INIT_ADAPTER_PATH="${PHASE1_CHECKPOINT_DIR}/${PHASE2_START_FROM}"
if [[ ! -d "$PHASE2_INIT_ADAPTER_PATH" ]]; then
  echo "[error] phase1 ${PHASE2_START_FROM} checkpoint not found: $PHASE2_INIT_ADAPTER_PATH" >&2
  exit 1
fi

# phase2：降低学习率、略提高 KL，奖励权重切回正式 top20 delta 配方。
echo "[phase2] epochs=1 batch=24 group=8 max_group=12 lr=6.0e-6 kl=0.055 decode=(10,0.80,0.92) reward=(0.52,0.22,0.16,0.10)"
echo "[phase2] adapter=${PHASE2_INIT_ADAPTER_PATH}"
run_train_phase \
  phase2 "$PHASE2_INIT_ADAPTER_PATH" 1 6.0e-6 0.055 0.80 0.92 60 0.52 0.22 0.16 \
  "$PHASE2_CHECKPOINT_DIR" "${PHASE2_DIR}/train_log.jsonl" "${PHASE2_DIR}/group_trace_log.jsonl"

PHASE2_BEST="${PHASE2_CHECKPOINT_DIR}/best"
if [[ ! -d "$PHASE2_BEST" ]]; then
  echo "[error] phase2 best checkpoint not found: $PHASE2_BEST" >&2
  exit 1
fi

# full eval 只评估 phase2 best adapter，并输出 Original/Zero-shot/RL 对比报告。
echo "[eval] adapter=${PHASE2_BEST} query_batch_size=8"
"$PYTHON_BIN" eval_compare.py \
  --rl-adapter-path "$PHASE2_BEST" \
  --model-name "$MODEL_NAME" \
  --strict-tokenizer-model-match \
  --max-eval-queries 1000000 \
  --max-new-tokens 10 \
  --temperature 0.80 \
  --top-p 0.92 \
  --query-batch-size 8 \
  --sample-print 20 \
  --report-path "$EVAL_REPORT_PATH"

echo "[done] report: $EVAL_REPORT_PATH"
