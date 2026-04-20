# 训练命令整理（统一到 `train_and_eval_data_model`）

## 目录规范

所有实验产物都放在同一个根目录下，并且 `train` / `eval` 分开：

```text
train_and_eval_data_model/
  artifacts_<exp_name>_train/
    checkpoints/
    train_log.jsonl
    group_trace_log.jsonl
  artifacts_<exp_name>_eval/
    eval_compare_report.json
```

示例：

- `artifacts_0p8b_train` + `artifacts_0p8b_eval`
- `artifacts_4b_train` + `artifacts_4b_eval`

---

## 统一变量（PowerShell）

```powershell
$ROOT = "train_and_eval_data_model"
$EXP = "0p8b"
$TRAIN_DIR = "$ROOT/artifacts_${EXP}_train"
$EVAL_DIR = "$ROOT/artifacts_${EXP}_eval"
```

## 统一变量（Bash）

```bash
ROOT="train_and_eval_data_model"
EXP="0p8b"
TRAIN_DIR="${ROOT}/artifacts_${EXP}_train"
EVAL_DIR="${ROOT}/artifacts_${EXP}_eval"
```

---

## 训练命令（0.8B 示例）

```powershell
python train.py `
  --low-mem-mode `
  --save-dir "$TRAIN_DIR/checkpoints" `
  --log-path "$TRAIN_DIR/train_log.jsonl" `
  --group-trace-log-path "$TRAIN_DIR/group_trace_log.jsonl"
```

---

## 训练命令（4B 主实验）

```powershell
$EXP = "4b"
$TRAIN_DIR = "$ROOT/artifacts_${EXP}_train"
$EVAL_DIR = "$ROOT/artifacts_${EXP}_eval"

python train.py `
  --model-name Qwen/Qwen3-4B-Instruct-2507 `
  --num-epochs 1 `
  --batch-size 24 `
  --group-size 10 `
  --max-group-size 14 `
  --learning-rate 1.5e-5 `
  --clip-range 0.2 `
  --kl-beta 0.03 `
  --max-new-tokens 12 `
  --temperature 0.85 `
  --top-p 0.95 `
  --group-temperature-stride 0.07 `
  --group-top-p-stride 0.015 `
  --min-unique-final-queries 4 `
  --max-regen-rounds 2 `
  --reward-gap-threshold 0.08 `
  --gap-sampling-temperature-delta 0.15 `
  --reward-mrr-k 10 `
  --reward-recall-k 50 `
  --reward-recall-dense-k 100 `
  --reward-w-mrr 0.40 `
  --reward-w-recall 0.20 `
  --reward-w-recall-dense 0.15 `
  --reward-w-term-preserve 0.10 `
  --reward-w-length-score 0.08 `
  --reward-w-clean-format 0.07 `
  --reward-w-bad-format 0.15 `
  --reward-w-unsafe-copy 0.08 `
  --save-dir "$TRAIN_DIR/checkpoints" `
  --log-path "$TRAIN_DIR/train_log.jsonl" `
  --group-trace-log-path "$TRAIN_DIR/group_trace_log.jsonl"
```

---

## 评估命令（与 train 分开）

优先用 `best`，不存在时回退 `latest`：

```powershell
$ADAPTER = "$TRAIN_DIR/checkpoints/best"
if (-not (Test-Path $ADAPTER)) { $ADAPTER = "$TRAIN_DIR/checkpoints/latest" }

python eval_compare.py `
  --rl-adapter-path $ADAPTER `
  --model-name Qwen/Qwen3-4B-Instruct-2507 `
  --strict-tokenizer-model-match `
  --max-eval-queries 1000000 `
  --sample-print 20 `
  --report-path "$EVAL_DIR/eval_compare_report_full.json"
```

---

## 一键脚本（已整理）

- Linux / macOS: `./run_train_then_full_eval_4b.sh`
- Windows PowerShell: `.\run_train_then_full_eval_4b.ps1`

这两个脚本默认都会写入：

- `train_and_eval_data_model/artifacts_4b_train`
- `train_and_eval_data_model/artifacts_4b_eval`

可选覆盖：

- `ARTIFACT_ROOT`：修改根目录
- `EXP_NAME`：修改实验名（默认 `4b`）
- `PYTHON_BIN`：修改 Python 命令（如 `python3`）
