# 训练命令速查（0.5B / 3B / 7B）

本文档整理了当前项目最常用的三档训练命令，并统一了输出目录结构，方便直接对比实验结果。

## 通用说明

- 日志文件：
  - `--log-path`：训练/评估指标日志（step 级）
  - `--group-trace-log-path`：每个 query 一条 group 采样明细日志（用于排查 4bit 乱码）
- 模型精度：
  - Actor 默认开启 4bit（除非传 `--disable-4bit`）
  - Ref 默认 `ref_precision_mode=auto`（先全精度，OOM 自动回退 4bit，且优先走 GPU 路径）
- 若只做快速联调，建议先跑 0.5B；确认链路稳定后再跑 3B/7B。

---

## 0.5B（低显存快速验证）

> 适合先验证训练链路、日志、奖励函数是否正常。

```powershell
python train.py `
  --low-mem-mode `
  --max-steps 50 `
  --max-train-queries 128 `
  --max-val-queries 64 `
  --save-dir artifacts_0p5b/checkpoints `
  --log-path artifacts_0p5b/train_log.jsonl `
  --group-trace-log-path artifacts_0p5b/group_trace_log.jsonl
```

---

## 3B（默认主实验）

> 推荐作为主实验配置，平衡效果与资源占用。

```powershell
python train.py `
  --model-name Qwen/Qwen3-4B-Instruct-2507 `
  --num-epochs 3 `
  --batch-size 4 `
  --group-size 8 `
  --max-new-tokens 20 `
  --temperature 0.7 `
  --top-p 0.9 `
  --save-dir artifacts_3b/checkpoints `
  --log-path artifacts_3b/train_log.jsonl `
  --group-trace-log-path artifacts_3b/group_trace_log.jsonl
```

---

## 7B（大模型实验）

> 默认仍走 4bit，建议在显存较充足机器上运行；先用保守 batch/group 起步。

```powershell
python train.py `
  --model-name Qwen/Qwen2.5-7B-Instruct `
  --num-epochs 1 `
  --batch-size 1 `
  --group-size 2 `
  --max-new-tokens 20 `
  --temperature 0.9 `
  --top-p 0.95 `
  --save-dir artifacts_7b/checkpoints `
  --log-path artifacts_7b/train_log.jsonl `
  --group-trace-log-path artifacts_7b/group_trace_log.jsonl
```

---

## 可选：关闭 4bit 对照实验

> 用于对照“4bit 开/关”输出质量。当前实现中不会再把 ref 强制放到 CPU。

```powershell
python train.py `
  --model-name Qwen/Qwen2.5-3B-Instruct `
  --disable-4bit `
  --batch-size 1 `
  --group-size 2 `
  --save-dir artifacts_3b_fp/checkpoints `
  --log-path artifacts_3b_fp/train_log.jsonl `
  --group-trace-log-path artifacts_3b_fp/group_trace_log.jsonl
```

---

## 建议的排查顺序（4bit 乱码问题）

1. 先跑 0.5B（低显存）确认 `group_trace_log.jsonl` 是否持续输出、字段是否完整。  
2. 再跑 3B（4bit 开启）观察 `group_raw_responses` 的可读性和 `unreadable_ratio_mean`。  
3. 最后跑 3B 关闭 4bit 对照，比较两组 trace 中的乱码比例与奖励差异。  

