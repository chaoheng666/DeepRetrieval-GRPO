# DeepRetrieval-GRPO 项目全量总结（截至 2026-04-13）

## 0. 文档说明

- 文档目的：把当前仓库的**项目思路、处理过程、当前效果、数据集与工程状态**一次性汇总成可交付的技术说明。
- 统计口径：全部基于仓库内现有代码与产物文件（不依赖外部猜测）。
- 关键结论先行：
  - 项目已形成完整闭环：`数据加载 -> GRPO训练 -> 三路评测 -> Prompt扫表评估`。
  - 目前多组实验中，**RL 相比 Zero-shot 有稳定提升**，但多数情况下仍**未超过 Original 原始检索基线**。
  - 奖励函数与评测口径已经发生版本演进（legacy -> v1），不同阶段结果需要按口径解读，不能简单横向对比。

---

## 1. 项目目标与核心思路

### 1.1 目标

项目目标是让大模型把用户查询重写成更利于 BM25 检索的查询，并以检索指标（MRR/Recall）作为强化学习奖励信号，最终提升检索质量。

### 1.2 技术路线

- 基础模型：Qwen 系列（0.5B / 0.8B / 3B / 4B 路径都已在工程中体现）。
- 训练算法：手写 GRPO（Group Relative Policy Optimization），不依赖 TRL。
- 检索后端：Pyserini 预编译索引（可回退本地 BM25 流程）。
- 训练方式：QLoRA（4-bit）+ LoRA 参数高效微调。

### 1.3 为什么是 GRPO

- 不需要单独 Critic（相比 PPO 可减显存压力）。
- 通过组内采样与相对优势归一化，直接在生成式任务中优化策略。
- 更适合“查询改写”这种序列级奖励（检索后反馈）的任务形态。

---

## 2. 仓库结构与模块职责

## 2.1 主工程（根目录）

- `app_config.py`：统一配置中心（数据、模型、训练、奖励、提示词）。
- `train.py`：GRPO 训练入口（含低显存模式、日志、checkpoint）。
- `eval_compare.py`：三路评测（Original / Zero-shot / RL）。
- `core/model_wrapper.py`：模型加载、生成、token 级 logprob 计算。
- `core/grpo_engine.py`：GRPO 核心训练循环与损失计算。
- `core/reward_func.py`：奖励函数（当前为 Reward v1）。
- `data/loader.py`：Pyserini 数据读取、Java 环境修复、数据切分。
- `prompt_eval/*`：提示词批量评估模块。
- `train_and_eval_data_model/*`：训练与评估产物目录。

## 2.2 data-local 子工程（最小闭环）

该目录是一个独立的“本地最小检索评估环境”：

- `scripts/prepare_data.py`：从 HF 数据集准备 `corpus/queries/qrels`。
- `scripts/build_index.py`：优先 Pyserini 建索引，失败回退 rank_bm25。
- `scripts/eval_mrr.py`：执行 MRR 评测并输出报告。
- `retrieval_eval/core.py`：最小 BM25 评测核心逻辑。

这个子目录用于快速验证 reward 闭环，不直接等同于主工程 RL 训练结果。

---

## 3. 配置体系（当前默认）

## 3.1 数据配置（DataConfig）

- `topic_name = msmarco-passage-dev-subset`
- `prebuilt_index = msmarco-v1-passage`
- `train_ratio = 0.8`
- `seed = 42`
- `max_train_queries = 20000`
- `max_val_queries = 4000`

## 3.2 模型配置（ModelConfig）

- 默认基础模型：`Qwen/Qwen2.5-3B-Instruct`
- 默认 `load_in_4bit = true`（QLoRA）
- LoRA 参数：`r=16`, `alpha=32`, `dropout=0.05`
- LoRA 目标层：`q/k/v/o/gate/up/down_proj`

## 3.3 训练配置（TrainConfig）

- `num_epochs = 1`
- `batch_size = 8`
- `group_size = 8`
- `learning_rate = 2e-5`
- `clip_range = 0.2`
- `kl_beta = 0.005`
- `max_new_tokens = 24`
- `eval_every_steps = 20`

## 3.4 奖励配置（RewardConfig，当前 v1）

- 检索项：`mrr_k=50`, `recall_k=50`
- 权重：
  - `w_mrr = 1.0`
  - `w_recall = 0.3`
  - `w_copy = 0.15`
  - `w_format = 0.2`
- 约束阈值：
  - `copy_tau = 0.6`
  - `format_max_tokens = 16`
  - `format_min_english_ratio = 0.8`
  - `format_max_unreadable_ratio = 0.3`

奖励公式（v1）：

```text
total = w_mrr * mrr + w_recall * recall - w_copy * copy_penalty - w_format * format_penalty
```

---

## 4. 端到端处理流程（主工程）

## 4.1 数据加载阶段

1. `data/loader.py` 先检查 Java 版本（要求 >= 21）。
2. 若 `JAVA_HOME` 过低但 PATH 上 java 合格，会自动修复 `JAVA_HOME`。
3. 通过 Pyserini 加载 topics + qrels。
4. 仅保留有正样本 qrels 的 query。
5. 按 `train_ratio` 切分 train/val，并可做 max 样本截断。

## 4.2 训练阶段（train.py + GRPOEngine）

1. 读取默认配置，按需应用 `--low-mem-mode`。
2. CLI 覆盖配置；并做运行时安全校正（例如 group_size < 2 会自动修正）。
3. 构建 ModelWrapper：
   - actor：可训练 LoRA 模型
   - ref：冻结参考模型（用于 KL 约束）
4. GRPO 训练循环：
   - 每个 query 采样 `group_size` 条重写
   - 每条重写计算 reward（检索 + 约束）
   - 组内标准化得到 advantage
   - 按 PPO-clipped objective + KL 项计算损失
   - 反向传播、梯度裁剪、优化器更新
5. 日志写入：
   - `train_log.jsonl`：step 级训练/评估指标
   - `group_trace_log.jsonl`：每个 query 的组采样细节
6. checkpoint 策略：
   - 每次评估后保存 `latest`
   - 验证 mrr 提升时更新 `best`

## 4.3 评估阶段（eval_compare.py）

统一在同一验证集对比三路：

- Original：原始 query
- Zero-shot：基础模型重写
- RL：加载 LoRA adapter 重写

输出到 `eval_compare_report.json`，包含指标与 delta。

## 4.4 Prompt Sweep 阶段（prompt_eval）

- 在固定随机样本上评估多种 prompt。
- 当前 `prompt_eval/prompt_bank.py` 可生成 50 个 prompt 候选。
- 当前实跑报告（seed42_n100）实际执行了 16 个 prompt（由 `--max-prompts 16` 控制）。
- 结果写入：
  - `prompt_eval_report.json`
  - `prompt_eval_history.json`

---

## 5. 数据集与数据资产

## 5.1 主工程数据（Pyserini）

- 数据集：`msmarco-passage-dev-subset`
- 检索索引：`msmarco-v1-passage`（低显存模式可切 `msmarco-v1-passage-slim`）
- 在 prompt_eval 报告中可见：
  - 验证池大小 `available_val_queries = 1396`
  - 样本抽样 `sample_size = 100`（seed=42）

## 5.2 data-local 数据资产（本地最小闭环）

`data-local/data/stats.json` 显示：

- `num_docs = 40,704`
- `num_queries = 5,000`
- `num_qrels_queries = 4,851`
- `num_empty_qrels_queries = 149`
- `avg_relevant_docs_per_qrel_query = 1.1167`
- `qrels_doc_coverage = 1.0`
- 来源：`ms_marco/v1.1`, split=`train[:5000]`, seed=42

该子工程主要用于数据准备、索引与 reward 逻辑验证。

---

## 6. 当前实验效果总览（核心）

以下均来自仓库已有 JSON 报告。

## 6.1 三路评测结果（主工程）

| 产物 | 模型 | 指标口径 | Original MRR | Zero-shot MRR | RL MRR | RL-Orig | RL-Zero |
|---|---|---|---:|---:|---:|---:|---:|
| `artifacts_0p5b_eval` | Qwen2.5-0.5B-Instruct | legacy | 0.241157 | 0.170716 | 0.209022 | -0.032135 | +0.038306 |
| `artifacts_0p8b_eval` | Qwen3.5-0.8B | legacy | 0.241157 | 0.221987 | 0.238657 | -0.002500 | +0.016670 |
| `artifacts_default_eval` | Qwen3.5-0.8B | v1 | 0.241157 | 0.085370 | 0.111352 | -0.129805 | +0.025982 |

补充（`artifacts_default_eval` 同时记录 Recall）：

- Original Recall@50 = 0.62
- Zero-shot Recall@50 = 0.25
- RL Recall@50 = 0.36

结论：

- RL 对 Zero-shot 有正向增益（所有报告均 `RL > Zero-shot`）。
- 但当前记录下 RL 仍未超过 Original 基线。
- 0.8B 的 legacy 结果最接近 Original（仅 -0.0025）。

## 6.2 Prompt Sweep 结果（`prompt_eval/artifacts/seed42_n100`）

- 运行时间：`2026-04-12T18:22:49Z`
- 评估模型：`Qwen/Qwen3.5-0.8B`（无 LoRA）
- 样本量：100 query
- prompt 数量：16
- baseline（Original）：
  - MRR = 0.217312
  - Recall = 0.63

最佳 prompt：`entity_anchor_alias_bridge`

- MRR = 0.210096
- Delta MRR = -0.007216（仍低于 Original）

统计结论：

- 16/16 prompt 全部未超过 Original（`prompts_beating_baseline_mrr = 0`）
- 所有 prompt 的 delta_mrr 区间：
  - max = -0.007216
  - min = -0.113581
  - mean = -0.054944

history 显示两次大样本 sweep：

1. 20 prompts：best delta -0.004887（`compact_6_14_recall_push`）
2. 16 prompts：best delta -0.007216（`entity_anchor_alias_bridge`）

## 6.3 训练日志轨迹观察

### A) `artifacts_0p8b_train/train_log.jsonl`

- 训练 step：1~50（50条 train）
- eval 点：step 10/20/30/40/50（5条 eval）
- eval `mrr_mean` 基本恒定：0.002232
- eval `reward_mean` 约 -0.0203 到 -0.0195
- 但 train 末尾 `mrr_mean=1.0, reward_mean=1.3`

现象：训练批次内高分与验证集表现之间存在明显落差。

### B) `artifacts_0p5b-train/train_log.jsonl`

- eval `mrr_mean` 全程 0.0（step 10~50）
- train 末尾同样出现高 train 指标（`mrr_mean=1.0`）

现象：0.5B 在该阶段基本未在验证集产生有效泛化收益。

### C) `train_log (2).jsonl`（历史长跑）

- 总 500 条：496 train + 4 eval（step 100/200/300/400）
- eval mrr_mean 全 0.0
- eval reward_mean 为负（-0.115 到 -0.18）
- format_penalty_mean 偏高（0.575~0.9）

该日志提示：历史上存在“格式约束违规导致奖励被持续惩罚”的阶段。

## 6.4 data-local 最小闭环评估结果

`data-local/runs/eval_report.json`：

- backend = `rank_bm25`
- sample_size = 100
- topk = 10
- avg_reward = 0.441742
- nonzero_ratio = 0.81
- 阈值校验通过（`passed=true`）

含义：本地最小检索评测链路本身是可工作的。

---

## 7. 奖励与评测口径演进（非常关键）

仓库产物可见两种奖励口径并存：

## 7.1 legacy（旧口径）

在早期评估报告中可见（如 `artifacts_0p5b_eval`, `artifacts_0p8b_eval`）：

- 使用 `topk / mrr_weight / overlap_weight / penalty_*`
- 主要形态：`MRR + overlap - penalty`

## 7.2 Reward v1（当前代码）

当前 `core/reward_func.py` 为：

- `MRR@k + Recall@k - CopyPenalty - FormatPenalty`
- 并含严格格式约束：单行、英文比例、长度、可读性、解释性文本检测等。

影响：

- 不同时间点的实验结果**不应直接横向比较绝对值**。
- 需要在同一奖励口径下做 A/B 才有可解释性。

---

## 8. Prompt 与输出质量现状

从 prompt_eval 样例可见当前主要问题：

- 模型经常输出 `<think>...</think>`、`Search Query:`、`Rewritten Query:` 等模板残留。
- 甚至出现代码块符号、多行文本、解释性文本。
- 这会同时影响：
  - 检索召回（关键词污染）
  - 格式惩罚（v1 下直接扣分）

另外，当前 `Rewarder.score()` 默认直接用 `rewritten_query.strip()` 检索，`clean_rewritten_query()` 并未在主打分路径中启用，这会放大模板残留对检索的负面影响。

---

## 9. 工程健壮性与测试状态

## 9.1 主工程测试

- 命令：`python -m unittest tests/test_grpo_math.py`
- 结果：22 个测试全部通过。
- 覆盖点包括：
  - advantage 标准化
  - PPO clip 数学
  - MRR/Recall 计算
  - reward 组合公式
  - 格式惩罚规则
  - query 清洗与 Rewarder 行为一致性

## 9.2 data-local 测试

- 在 `data-local` 目录执行：`python -m unittest discover -s tests -p "test_*.py"`
- 结果：6 个测试通过。

注意：若在仓库根目录直接跑 data-local 测试，会因模块路径导致导入失败，需要切到 `data-local` 目录运行。

---

## 10. 已具备能力与当前短板

## 10.1 已具备能力

1. 完整的训练闭环与评估闭环已打通。
2. 支持 4-bit + LoRA 的轻量训练路径。
3. 支持低显存模式（快速 smoke）。
4. 支持 prompt 批量评估与历史记录。
5. 支持自动化脚本一键训练+评估（含 4B 配置路径）。

## 10.2 当前短板

1. 当前公开产物中，RL 尚未超过 Original 基线。
2. 输出格式污染（模板残留、思维链残留）显著影响检索质量。
3. 奖励口径发生演进，历史结果存在可比性断层。
4. train 高分与 eval 低分之间存在明显脱节，泛化稳定性不足。

---

## 11. 建议的后续推进方向（按优先级）

1. 统一评估口径
- 固定 Reward v1 参数与评测脚本，重新跑一版可比实验矩阵（0.5B/0.8B/3B/4B）。

2. 强化输出清洗链路
- 在 reward 检索前引入 `clean_rewritten_query`（或增加后处理层），先消除模板污染。

3. 做最小变量对照实验
- 固定模型与数据，只替换 prompt / 解码参数 / reward 权重，逐项验证增益来源。

4. 对齐训练与验证分布
- 检查 group 内采样与验证推理解码策略差异（temperature/top_p）对泛化的影响。

5. 建立统一实验看板
- 从 `train_log.jsonl` + `eval_compare_report.json` 自动汇总生成同口径 dashboard，避免手工对账。

---

## 12. 可复现命令清单（当前仓库）

## 12.1 主训练

```bash
python train.py
```

## 12.2 低显存 smoke

```bash
python train.py --low-mem-mode
```

## 12.3 三路评测

```bash
python eval_compare.py --rl-adapter-path <checkpoint_dir>
```

## 12.4 Prompt Sweep

```bash
python prompt_eval/run_prompt_eval.py --sample-size 100
```

## 12.5 data-local 最小闭环

```bash
cd data-local
python scripts/prepare_data.py --dataset ms_marco --split "train[:5000]" --seed 42
python scripts/build_index.py --input data --index index --backend auto
python scripts/eval_mrr.py --queries data/queries.jsonl --qrels data/qrels.json --corpus data/corpus.jsonl --index index --topk 10 --sample-size 100 --seed 42
```

---

## 13. 关键产物索引（便于查阅）

- 主说明文档：`README.md`
- 训练脚本：`train.py`
- 三路评测：`eval_compare.py`
- 奖励实现：`core/reward_func.py`
- GRPO 实现：`core/grpo_engine.py`
- 模型封装：`core/model_wrapper.py`
- Prompt sweep：`prompt_eval/run_prompt_eval.py`
- Prompt 候选：`prompt_eval/prompt_bank.py`
- Prompt 报告：`prompt_eval/artifacts/seed42_n100/prompt_eval_report.json`
- 主评估报告：
  - `train_and_eval_data_model/artifacts_0p5b_eval/eval_compare_report.json`
  - `train_and_eval_data_model/artifacts_0p8b_eval/eval_compare_report.json`
  - `train_and_eval_data_model/artifacts_default_eval/eval_compare_report.json`
- 历史训练日志：`train_log (2).jsonl`
- data-local 统计：`data-local/data/stats.json`
- data-local 评估：`data-local/runs/eval_report.json`

---

## 14. 一句话总结

这个项目已经是一个可运行、可扩展、可评估的“生成式检索强化学习工程骨架”，当前阶段的主要矛盾不在于链路通不通，而在于**如何把 RL 的训练收益稳定转化为对 Original 基线的可复现超越**。
