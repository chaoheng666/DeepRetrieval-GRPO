#!/usr/bin/env bash
set -euo pipefail

# Independent phase2 trainer for the top20_delta curriculum run.
# It can start from any LoRA adapter checkpoint instead of being tied to phase1/best.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_TS="$(date '+%Y%m%d_%H%M%S')"

OUTPUT_DIR_WAS_SET="${OUTPUT_DIR+x}"
METADATA_PATH_WAS_SET="${CURRICULUM_METADATA_PATH+x}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MODEL_NAME="${MODEL_NAME:-/root/autodl-tmp/hf_models/Qwen3-4B-Instruct-2507}"
RUN_ROOT="${RUN_ROOT:-train_and_eval_data_model_0422/artifacts_4b_top20_delta_curriculum_independent}"
CURRICULUM_METADATA_PATH="${CURRICULUM_METADATA_PATH:-${RUN_ROOT}/curriculum_query_metadata.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/phase2_${RUN_TS}}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${ADAPTER_PATH:-}}"
LOG_DIR="${LOG_DIR:-log}"

NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-60}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-20}"
MAX_TRAIN_QUERIES="${MAX_TRAIN_QUERIES:-}"
MAX_VAL_QUERIES="${MAX_VAL_QUERIES:-}"

BATCH_SIZE="${BATCH_SIZE:-24}"
GROUP_SIZE="${GROUP_SIZE:-8}"
MAX_GROUP_SIZE="${MAX_GROUP_SIZE:-12}"
LEARNING_RATE="${LEARNING_RATE:-6.0e-6}"
KL_BETA="${KL_BETA:-0.055}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10}"
TEMPERATURE="${TEMPERATURE:-0.84}"
TOP_P="${TOP_P:-0.94}"
EVAL_QUERY_BATCH_SIZE="${EVAL_QUERY_BATCH_SIZE:-8}"

GROUP_TEMPERATURE_STRIDE="${GROUP_TEMPERATURE_STRIDE:-0.10}"
GROUP_TOP_P_STRIDE="${GROUP_TOP_P_STRIDE:-0.025}"
MIN_UNIQUE_FINAL_QUERIES="${MIN_UNIQUE_FINAL_QUERIES:-4}"
MAX_REGEN_ROUNDS="${MAX_REGEN_ROUNDS:-2}"
REWARD_GAP_THRESHOLD="${REWARD_GAP_THRESHOLD:-0.08}"

REWARD_W_MRR="${REWARD_W_MRR:-0.52}"
REWARD_W_RECALL="${REWARD_W_RECALL:-0.22}"
REWARD_W_RECALL_DENSE="${REWARD_W_RECALL_DENSE:-0.16}"
REWARD_W_RANK_BONUS="${REWARD_W_RANK_BONUS:-0.10}"
RECALL_DROP_LAMBDA="${RECALL_DROP_LAMBDA:-1.20}"
ANCHOR_BONUS_VALUE="${ANCHOR_BONUS_VALUE:-0.00}"

EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash run_train_phase2_top20_delta_curriculum.sh [options] [-- extra train.py args...]

Options:
  --source-checkpoint PATH   Optional LoRA adapter checkpoint to warm-start from.
  --adapter-path PATH        Alias of --source-checkpoint.
  --output-dir PATH          Output directory for this independent phase2 run.
  --run-root PATH            Experiment root. Default comes from RUN_ROOT env.
  --metadata-path PATH       Curriculum metadata jsonl path.
  --model-name PATH          Base model name/path.
  --python-bin PATH          Python executable.
  --num-epochs N             Default: 1
  --max-steps N              Default: 60
  --eval-every-steps N       Default: 20
  --max-train-queries N      Optional train-query cap passed to train.py.
  --max-val-queries N        Optional val-query cap passed to train.py.
  --learning-rate VALUE      Default: 6.0e-6
  --kl-beta VALUE            Default: 0.055
  --temperature VALUE        Default: 0.84
  --top-p VALUE              Default: 0.94
  --reward-w-mrr VALUE       Default: 0.52
  --reward-w-recall VALUE    Default: 0.22
  --reward-w-recall-dense V  Default: 0.16
  --reward-w-rank-bonus V    Default: 0.10
  --recall-drop-lambda V     Default: 1.20
  --reward-gap-threshold V   Default: 0.08
  -h, --help                 Show this help.

Common env overrides:
  SOURCE_CHECKPOINT, OUTPUT_DIR, RUN_ROOT, CURRICULUM_METADATA_PATH, MODEL_NAME
  NUM_EPOCHS, MAX_STEPS, LEARNING_RATE, KL_BETA, REWARD_W_MRR

Examples:
  bash run_train_phase2_top20_delta_curriculum.sh --source-checkpoint phase1_run/checkpoints/best
  bash run_train_phase2_top20_delta_curriculum.sh --source-checkpoint phase2_old/checkpoints/best --max-steps 120 --learning-rate 4e-6
  bash run_train_phase2_top20_delta_curriculum.sh -- --seed 123 --search-threads 24
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-checkpoint|--adapter-path)
      [[ $# -ge 2 ]] || { echo "[error] $1 requires a path" >&2; exit 1; }
      SOURCE_CHECKPOINT="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "[error] --output-dir requires a path" >&2; exit 1; }
      OUTPUT_DIR="$2"
      OUTPUT_DIR_WAS_SET=x
      shift 2
      ;;
    --run-root)
      [[ $# -ge 2 ]] || { echo "[error] --run-root requires a path" >&2; exit 1; }
      RUN_ROOT="$2"
      if [[ -z "$METADATA_PATH_WAS_SET" ]]; then
        CURRICULUM_METADATA_PATH="${RUN_ROOT}/curriculum_query_metadata.jsonl"
      fi
      if [[ -z "$OUTPUT_DIR_WAS_SET" ]]; then
        OUTPUT_DIR="${RUN_ROOT}/phase2_${RUN_TS}"
      fi
      shift 2
      ;;
    --metadata-path)
      [[ $# -ge 2 ]] || { echo "[error] --metadata-path requires a path" >&2; exit 1; }
      CURRICULUM_METADATA_PATH="$2"
      METADATA_PATH_WAS_SET=x
      shift 2
      ;;
    --model-name)
      [[ $# -ge 2 ]] || { echo "[error] --model-name requires a path/name" >&2; exit 1; }
      MODEL_NAME="$2"
      shift 2
      ;;
    --python-bin)
      [[ $# -ge 2 ]] || { echo "[error] --python-bin requires a path" >&2; exit 1; }
      PYTHON_BIN="$2"
      shift 2
      ;;
    --num-epochs)
      [[ $# -ge 2 ]] || { echo "[error] --num-epochs requires a value" >&2; exit 1; }
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --max-steps)
      [[ $# -ge 2 ]] || { echo "[error] --max-steps requires a value" >&2; exit 1; }
      MAX_STEPS="$2"
      shift 2
      ;;
    --eval-every-steps)
      [[ $# -ge 2 ]] || { echo "[error] --eval-every-steps requires a value" >&2; exit 1; }
      EVAL_EVERY_STEPS="$2"
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
    --learning-rate)
      [[ $# -ge 2 ]] || { echo "[error] --learning-rate requires a value" >&2; exit 1; }
      LEARNING_RATE="$2"
      shift 2
      ;;
    --kl-beta)
      [[ $# -ge 2 ]] || { echo "[error] --kl-beta requires a value" >&2; exit 1; }
      KL_BETA="$2"
      shift 2
      ;;
    --temperature)
      [[ $# -ge 2 ]] || { echo "[error] --temperature requires a value" >&2; exit 1; }
      TEMPERATURE="$2"
      shift 2
      ;;
    --top-p)
      [[ $# -ge 2 ]] || { echo "[error] --top-p requires a value" >&2; exit 1; }
      TOP_P="$2"
      shift 2
      ;;
    --reward-w-mrr)
      [[ $# -ge 2 ]] || { echo "[error] --reward-w-mrr requires a value" >&2; exit 1; }
      REWARD_W_MRR="$2"
      shift 2
      ;;
    --reward-w-recall)
      [[ $# -ge 2 ]] || { echo "[error] --reward-w-recall requires a value" >&2; exit 1; }
      REWARD_W_RECALL="$2"
      shift 2
      ;;
    --reward-w-recall-dense)
      [[ $# -ge 2 ]] || { echo "[error] --reward-w-recall-dense requires a value" >&2; exit 1; }
      REWARD_W_RECALL_DENSE="$2"
      shift 2
      ;;
    --reward-w-rank-bonus)
      [[ $# -ge 2 ]] || { echo "[error] --reward-w-rank-bonus requires a value" >&2; exit 1; }
      REWARD_W_RANK_BONUS="$2"
      shift 2
      ;;
    --recall-drop-lambda)
      [[ $# -ge 2 ]] || { echo "[error] --recall-drop-lambda requires a value" >&2; exit 1; }
      RECALL_DROP_LAMBDA="$2"
      shift 2
      ;;
    --reward-gap-threshold)
      [[ $# -ge 2 ]] || { echo "[error] --reward-gap-threshold requires a value" >&2; exit 1; }
      REWARD_GAP_THRESHOLD="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
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

if [[ -n "$SOURCE_CHECKPOINT" && ! -d "$SOURCE_CHECKPOINT" ]]; then
  echo "[error] source checkpoint not found: $SOURCE_CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
RUN_LOG_PATH="${LOG_DIR}/run_train_phase2_top20_delta_curriculum_${RUN_TS}.log"
if [[ -z "${RUN_LOG_REDIRECTED:-}" ]]; then
  export RUN_LOG_REDIRECTED=1
  exec > >(tee -a "$RUN_LOG_PATH") 2>&1
fi

SAVE_DIR="${OUTPUT_DIR}/checkpoints"
TRAIN_LOG_PATH="${OUTPUT_DIR}/train_log.jsonl"
TRACE_LOG_PATH="${OUTPUT_DIR}/group_trace_log.jsonl"

args=(
  --model-name "$MODEL_NAME"
  --curriculum-phase phase2
  --curriculum-metadata-path "$CURRICULUM_METADATA_PATH"
  --num-epochs "$NUM_EPOCHS"
  --batch-size "$BATCH_SIZE"
  --group-size "$GROUP_SIZE"
  --max-group-size "$MAX_GROUP_SIZE"
  --learning-rate "$LEARNING_RATE"
  --kl-beta "$KL_BETA"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --group-temperature-stride "$GROUP_TEMPERATURE_STRIDE"
  --group-top-p-stride "$GROUP_TOP_P_STRIDE"
  --min-unique-final-queries "$MIN_UNIQUE_FINAL_QUERIES"
  --max-regen-rounds "$MAX_REGEN_ROUNDS"
  --reward-gap-threshold "$REWARD_GAP_THRESHOLD"
  --eval-max-new-tokens "$MAX_NEW_TOKENS"
  --eval-temperature "$TEMPERATURE"
  --eval-top-p "$TOP_P"
  --eval-query-batch-size "$EVAL_QUERY_BATCH_SIZE"
  --eval-every-steps "$EVAL_EVERY_STEPS"
  --max-steps "$MAX_STEPS"
  --reward-w-mrr "$REWARD_W_MRR"
  --reward-w-recall "$REWARD_W_RECALL"
  --reward-w-recall-dense "$REWARD_W_RECALL_DENSE"
  --reward-w-rank-bonus "$REWARD_W_RANK_BONUS"
  --recall-drop-lambda "$RECALL_DROP_LAMBDA"
  --anchor-bonus-value "$ANCHOR_BONUS_VALUE"
  --save-dir "$SAVE_DIR"
  --log-path "$TRAIN_LOG_PATH"
  --group-trace-log-path "$TRACE_LOG_PATH"
)

if [[ -n "$MAX_TRAIN_QUERIES" ]]; then
  args+=(--max-train-queries "$MAX_TRAIN_QUERIES")
fi
if [[ -n "$MAX_VAL_QUERIES" ]]; then
  args+=(--max-val-queries "$MAX_VAL_QUERIES")
fi
if [[ -n "$SOURCE_CHECKPOINT" ]]; then
  args+=(--adapter-path "$SOURCE_CHECKPOINT")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  args+=("${EXTRA_ARGS[@]}")
fi

echo "[log] command output is also saved to: $RUN_LOG_PATH"
echo "[phase2] output_dir=$OUTPUT_DIR"
echo "[phase2] source_checkpoint=${SOURCE_CHECKPOINT:-<fresh>}"
echo "[phase2] metadata=$CURRICULUM_METADATA_PATH"
echo "[phase2] epochs=$NUM_EPOCHS max_steps=$MAX_STEPS lr=$LEARNING_RATE kl=$KL_BETA reward_mrr=$REWARD_W_MRR"

"$PYTHON_BIN" train.py "${args[@]}"

echo "[done] phase2 output: $OUTPUT_DIR"
echo "[done] best checkpoint: ${SAVE_DIR}/best"
echo "[done] latest checkpoint: ${SAVE_DIR}/latest"
