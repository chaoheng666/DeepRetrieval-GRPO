#!/usr/bin/env bash
set -euo pipefail

# Lightweight validation for the independent phase training wrappers.
# This only checks syntax and CLI surface. It does not start model training.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PHASE1_SCRIPT="./run_train_phase1_top20_delta_curriculum.sh"
PHASE2_SCRIPT="./run_train_phase2_top20_delta_curriculum.sh"

required_files=(
  "$PHASE1_SCRIPT"
  "$PHASE2_SCRIPT"
  "./train.py"
)

required_flags=(
  "--source-checkpoint"
  "--adapter-path"
  "--output-dir"
  "--max-steps"
  "--learning-rate"
  "--reward-w-mrr"
  "--reward-gap-threshold"
)

echo "[check] required files"
for file in "${required_files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "[error] missing file: $file" >&2
    exit 1
  fi
  echo "  ok: $file"
done

echo "[check] bash syntax"
bash -n "$PHASE1_SCRIPT"
echo "  ok: $PHASE1_SCRIPT"
bash -n "$PHASE2_SCRIPT"
echo "  ok: $PHASE2_SCRIPT"

echo "[check] wrapper help flags"
for script in "$PHASE1_SCRIPT" "$PHASE2_SCRIPT"; do
  help_text="$(bash "$script" --help)"
  for flag in "${required_flags[@]}"; do
    if ! grep -q -- "$flag" <<<"$help_text"; then
      echo "[error] $script help is missing flag: $flag" >&2
      exit 1
    fi
  done
  echo "  ok: $script"
done

echo "[check] train.py accepts wrapper-forwarded arguments"
train_required_flags=(
  "--adapter-path"
  "--curriculum-phase"
  "--curriculum-metadata-path"
  "--save-dir"
  "--log-path"
  "--group-trace-log-path"
  "--reward-w-mrr"
  "--reward-gap-threshold"
)

for flag in "${train_required_flags[@]}"; do
  if ! grep -q -- "$flag" ./train.py; then
    echo "[error] train.py source is missing flag: $flag" >&2
    exit 1
  fi
done
echo "  ok: train.py"

echo "[done] independent phase scripts look valid."
