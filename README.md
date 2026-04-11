# DeepRetrieval-GRPO（中文运行手册）

本项目是一个轻量级、可离线交付的查询重写强化学习工程：

- 基础模型：`Qwen/Qwen2.5-3B-Instruct`
- 算法：纯 PyTorch 手写 GRPO（不依赖 TRL 等 RL 框架）
- 检索与数据：严格使用 Pyserini 预编译接口
  - `get_topics/get_qrels('msmarco-passage-dev-subset')`
  - `LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')`

目标是让模型把模糊用户查询重写为更精准的搜索查询，并通过 MRR@10 衡量检索收益。

---

## 1. 项目结构

```text
.
|-- app_config.py                # 全局配置（训练、模型、奖励、路径）
|-- train.py                     # 训练入口（GRPO 主循环）
|-- eval_compare.py              # 对比评测（Original / Zero-shot / RL）
|-- requirements.txt             # Python 依赖
|-- core/
|   |-- model_wrapper.py         # QLoRA模型加载 + 生成 + logprob计算
|   |-- grpo_engine.py           # GRPO核心（采样、优势归一化、PPO-clip、KL）
|   `-- reward_func.py           # 奖励函数（MRR@10 + 文本惩罚）
|-- data/
|   `-- loader.py                # Pyserini topics/qrels 加载与数据切分
`-- tests/
    `-- test_grpo_math.py        # 最小数学/奖励单元测试
```

---

## 2. 运行环境要求

## 2.1 硬件建议

- GPU：建议 24GB 显存（如 RTX 3090 / 4090）
- 显存不足时可先用小批量和小样本跑通流程

## 2.2 软件版本建议

- Python：`3.11`（强烈建议，Pyserini 兼容性最好）
- CUDA：建议与你的 PyTorch 版本匹配
- Java：建议 `21`（Pyserini/Anserini 通常要求较新 Java）
- 操作系统：Windows / Linux 均可

## 2.3 环境自检命令

```bash
python --version
java -version
nvidia-smi
```

---

## 3. 环境配置（推荐两种方式）

## 3.1 方式 A：Conda（推荐）

```bash
conda create -n dr-grpo python=3.11 -y
conda activate dr-grpo
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3.2 方式 B：venv

Linux / macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 4. 首次启动前说明（很重要）

首次运行时，Pyserini 会自动下载/缓存：

- 预编译索引（`msmarco-v1-passage`）
- topics/qrels 资源

因此第一次启动可能较慢，属于正常现象。

---

## 5. 快速开始（最短路径）

执行顺序建议：

1. 安装依赖  
2. 跑一次小规模训练（确认链路通）  
3. 跑对比评测  

## 5.1 小规模训练（快速验证）

Linux/macOS:

```bash
python train.py \
  --num-epochs 1 \
  --batch-size 1 \
  --group-size 2 \
  --max-train-queries 64 \
  --max-val-queries 32 \
  --eval-every-steps 10 \
  --max-steps 20
```

Windows PowerShell（换行符用反引号）：

```powershell
python train.py `
  --num-epochs 1 `
  --batch-size 1 `
  --group-size 2 `
  --max-train-queries 64 `
  --max-val-queries 32 `
  --eval-every-steps 10 `
  --max-steps 20
```

## 5.2 评测对比

```bash
python eval_compare.py --rl-adapter-path train_and_eval_data_model/artifacts_default_train/checkpoints/best
```

---

## 6. 训练命令详解

`train.py` 会自动完成：

- 加载 Pyserini 数据并做 80/20 切分
- 加载 QLoRA actor + 冻结 reference model
- 执行 GRPO 更新
- 周期性验证并保存 best/latest adapter

## 6.1 默认训练

```bash
python train.py
```

## 6.2 常用自定义参数示例

```bash
python train.py \
  --num-epochs 1 \
  --batch-size 2 \
  --group-size 4 \
  --learning-rate 2e-5 \
  --clip-range 0.2 \
  --kl-beta 0.02 \
  --max-new-tokens 24 \
  --temperature 1.0 \
  --top-p 0.95 \
  --eval-every-steps 20 \
  --max-train-queries 2000 \
  --max-val-queries 400
```

## 6.3 从已有 Adapter 热启动

```bash
python train.py --adapter-path train_and_eval_data_model/artifacts_default_train/checkpoints/best
```

## 6.4 自定义输出路径

```bash
python train.py \
  --save-dir runs/exp1/checkpoints \
  --log-path runs/exp1/train_log.jsonl
```

## 6.5 关键参数说明（训练）

- `--group-size`：GRPO 每条 query 采样候选数 K  
- `--clip-range`：PPO clip epsilon  
- `--kl-beta`：KL 惩罚系数  
- `--max-new-tokens`：每条重写最大生成长度  
- `--max-steps`：总步数上限（调试时常用）  
- `--max-train-queries/--max-val-queries`：限制样本数，缩短验证时间  

---

## 7. 评测命令详解

`eval_compare.py` 在同一验证集上比较三种策略：

- Original（原始 query）
- Zero-shot（基础模型提示重写）
- RL（加载 GRPO 训练后的 adapter）

## 7.1 基本用法

```bash
python eval_compare.py --rl-adapter-path train_and_eval_data_model/artifacts_default_train/checkpoints/best
```

低显存训练产物推荐这样评估（更快）：

```bash
python eval_compare.py --rl-adapter-path train_and_eval_data_model/artifacts_lowmem_train/checkpoints/best --low-mem-mode
```

## 7.2 输出更多样例 + 自定义报告路径

```bash
python eval_compare.py \
  --rl-adapter-path train_and_eval_data_model/artifacts_default_train/checkpoints/best \
  --sample-print 20 \
  --report-path train_and_eval_data_model/artifacts_default_eval/eval_compare_report_exp1.json
```

## 7.3 限制评估样本数（快速）

```bash
python eval_compare.py \
  --rl-adapter-path train_and_eval_data_model/artifacts_default_train/checkpoints/best \
  --max-eval-queries 100
```

## 7.4 常用参数说明（评测）

- `--rl-adapter-path`：必填，RL adapter 目录  
- `--model-name`：覆盖评测基础模型（adapter 必须与该模型匹配）  
- `--low-mem-mode`：评测低显存预设（0.5B + slim 索引 + 较小样本）  
- `--sample-print`：打印样例数量  
- `--max-eval-queries`：评估样本数上限  
- `--report-path`：评测结果 JSON 输出路径  

---

## 8. 输出文件与目录说明

默认输出如下：

- `train_and_eval_data_model/artifacts_default_train/checkpoints/best/`：验证集 MRR 最优 adapter
- `train_and_eval_data_model/artifacts_default_train/checkpoints/latest/`：最近一次保存 adapter
- `train_and_eval_data_model/artifacts_default_train/train_log.jsonl`：训练与评估过程日志
- `train_and_eval_data_model/artifacts_default_eval/eval_compare_report.json`：三路对比评测报告

`train_log.jsonl` 每行一条 JSON，可用于后续画图分析（loss、mrr、reward 等）。

---

## 9. 配置文件说明（app_config.py）

你可以直接修改 `app_config.py` 的默认值，也可以通过 CLI 覆盖。

建议原则：

1. 先用默认配置跑通链路  
2. 再小步调整 `group-size / kl-beta / learning-rate`  
3. 最后再放大数据量和训练步数  

---

## 10. 常见问题（FAQ）

## 10.1 `pyserini` 安装失败

优先检查：

1. Python 是否为 3.11  
2. Java 是否安装并可用（`java -version`）  
3. pip 是否最新（`python -m pip install --upgrade pip`）  

## 10.1.1 `UnsupportedClassVersionError`（你遇到的这个）

报错特征（示例）：

```text
... compiled by a more recent version ...
... only recognizes class file versions up to 62.0
```

这表示：运行时 Java 太低（62=Java 18），而当前 Pyserini 需要 Java 21（65）或更高。

本项目已在 `data/loader.py` 里加入自动修复逻辑：

1. 若 `JAVA_HOME` 太低，但 PATH 上 `java` 足够新，会自动切换到新 JDK  
2. 若两者都不满足，会给出明确错误说明

你也可以手动修复（Windows PowerShell）：

```powershell
$env:JAVA_HOME = "D:\JDK"        # 改成你自己的 JDK21+ 路径
$env:Path = "$env:JAVA_HOME\\bin;$env:Path"
java -version
python train.py --low-mem-mode
```

## 10.2 首次运行很慢

这是正常现象，通常是 Pyserini 在下载预编译索引与评测资源缓存。

## 10.2.1 预编译索引下载中断后报尺寸不匹配

报错示例：

```text
... does not match expected file size! Expecting ..., got ...
```

原因：
- 下载被中断，缓存目录里留下了损坏的 `.tar.gz` 半包。

本项目已在 `core/reward_func.py` 内加入自动恢复：

1. 检测到该错误时自动删除坏包  
2. 自动重试一次下载

若你想手动清理，可删除：

```text
C:\Users\<你的用户名>\.cache\pyserini\indexes\
```

然后重跑训练命令即可。

## 10.3 显存不足（OOM）

依次尝试：

1. 减小 `--batch-size`
2. 减小 `--group-size`
3. 减小 `--max-new-tokens`
4. 先设置更小的 `--max-train-queries`
5. 降低并发评估频率（增大 `--eval-every-steps`）

## 10.4 训练日志里出现 NaN

建议排查：

1. 学习率是否过大（适当降低）
2. `kl-beta` 是否过小（策略漂移过大）
3. 检查输入数据与生成文本是否异常

---

## 11. 开发与静态检查命令

仅做语法检查：

```bash
python -m compileall app_config.py core data train.py eval_compare.py tests/test_grpo_math.py
```

运行最小单元测试：

```bash
python -m unittest tests/test_grpo_math.py
```

---

## 12. 推荐复现流程（实战）

```bash
# 1) 环境准备
python -m pip install -r requirements.txt

# 2) 快速链路验证
python train.py --max-train-queries 64 --max-val-queries 32 --max-steps 20

# 3) 正式训练（按需调参）
python train.py --max-train-queries 2000 --max-val-queries 400

# 4) 对比评测
python eval_compare.py --rl-adapter-path train_and_eval_data_model/artifacts_default_train/checkpoints/best

python eval_compare.py --rl-adapter-path train_and_eval_data_model/artifacts_lowmem_train/checkpoints/best --low-mem-mode --strict-tokenizer-model-match

```

---

## 14. 低显存模式（`--low-mem-mode`）

如果你的机器显存较小（例如 6GB），可以直接使用：

```bash
python train.py --low-mem-mode
```

该模式会自动应用一组保守参数（用于跑通链路，不是最终效果配置）：

1. 模型切换为 `Qwen/Qwen2.5-0.5B-Instruct`
2. `batch_size=1`
3. `group_size=2`（保证 GRPO 有组内优势信号）
4. `max_new_tokens=16`
5. `max_train_queries=64`
6. `max_val_queries=32`
7. `max_steps=20`
8. 奖励计算切换到 `MRR@10 + Recall@50`
9. 启用 `CopyPenalty` 与 `FormatPenalty` 约束，避免纯 MRR 稀疏导致无学习信号
10. 检索索引切换为 `msmarco-v1-passage-slim`（下载体积更小）
11. 输出目录改为 `train_and_eval_data_model/artifacts_lowmem_train/`

另外：
- 若检测到 CUDA 不可用，会自动切换为 CPU 兼容加载（很慢，但可用于排障和链路验证）。

你仍然可以在低显存模式下覆盖参数，例如：

```bash
python train.py --low-mem-mode --max-steps 50 --max-train-queries 128 --reward-mrr-k 10 --reward-recall-k 50
```

Windows PowerShell 示例：

```powershell
python train.py --low-mem-mode --max-steps 50 --max-train-queries 128 --reward-mrr-k 10 --reward-recall-k 50
```


python train.py `
  --num-epochs 3 `
  --batch-size 4 `
  --group-size 8 `
  --learning-rate 1.5e-5 `
  --clip-range 0.2 `
  --kl-beta 0.03 `
  --max-new-tokens 32 `
  --temperature 0.9 `
  --top-p 0.95 `
  --reward-mrr-k 10 `
  --reward-recall-k 50 `
  --reward-w-mrr 1.0 `
  --reward-w-recall 0.3 `
  --reward-w-copy 0.15 `
  --reward-w-format 0.2 `
  --eval-every-steps 100 `
  --save-dir train_and_eval_data_model/artifacts_3b_heavy_train/checkpoints `
  --log-path train_and_eval_data_model/artifacts_3b_heavy_train/train_log.jsonl

默认就是Qwen/Qwen2.5-3B-Instruct
