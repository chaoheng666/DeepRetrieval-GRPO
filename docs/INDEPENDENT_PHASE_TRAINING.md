# 独立阶段训练脚本

这两个脚本把原来耦合在一起的 phase1 -> phase2 流程拆开了。每个阶段都可以：

- 从 base model 新建 LoRA 训练；
- 用 `--source-checkpoint` 从任意已有 LoRA adapter checkpoint 继续训练；
- 写入带时间戳的新输出目录，避免污染旧日志；
- 用 `--` 把额外参数原样透传给 `train.py`。

## 阶段 1

从头训练：

```bash
bash run_train_phase1_top20_delta_curriculum.sh
```

从已有 checkpoint 继续阶段 1：

```bash
bash run_train_phase1_top20_delta_curriculum.sh \
  --source-checkpoint train_and_eval_data_model_0422/artifacts_4b_top20_delta_curriculum/phase1/checkpoints/best \
  --max-steps 160 \
  --learning-rate 8e-6
```

## 阶段 2

从阶段 1 的 best 开始阶段 2：

```bash
bash run_train_phase2_top20_delta_curriculum.sh \
  --source-checkpoint train_and_eval_data_model_0422/artifacts_4b_top20_delta_curriculum/phase1/checkpoints/best
```

继续已有阶段 2：

```bash
bash run_train_phase2_top20_delta_curriculum.sh \
  --source-checkpoint train_and_eval_data_model_0422/artifacts_4b_top20_delta_curriculum/phase2/checkpoints/best \
  --max-steps 120 \
  --learning-rate 4e-6 \
  --eval-every-steps 10
```

## 常用输出

脚本默认输出到：

```text
train_and_eval_data_model_0422/artifacts_4b_top20_delta_curriculum_independent/
```

每次运行会生成：

```text
phase1_<timestamp>/
  checkpoints/best
  checkpoints/latest
  train_log.jsonl
  group_trace_log.jsonl

phase2_<timestamp>/
  checkpoints/best
  checkpoints/latest
  train_log.jsonl
  group_trace_log.jsonl
```

指定固定输出目录：

```bash
bash run_train_phase2_top20_delta_curriculum.sh \
  --source-checkpoint old_run/checkpoints/best \
  --output-dir train_and_eval_data_model_0422/my_phase2_continue
```

## 额外参数透传

`--` 后面的参数会直接传给 `train.py`：

```bash
bash run_train_phase2_top20_delta_curriculum.sh \
  --source-checkpoint old_run/checkpoints/best \
  --max-steps 120 \
  -- \
  --seed 123 \
  --search-threads 24
```
