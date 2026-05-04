# DeepRetrieval-GRPO 项目总结

- 主实验流水线：`run_train_then_full_eval_4b_top20_delta_curriculum.sh`
- 指定实验结果：`train_and_eval_data_model_0423/artifacts_4b_top20_delta_curriculum_sweeps/run_20260423_161608/eval`

## 1. 项目目标

DeepRetrieval-GRPO 的核心目标是训练一个面向稀疏检索的 query rewrite 模型：给定用户原始查询，模型输出一条更适合 BM25 检索的英文检索式，使相关文档尽可能进入 top-20，并尽量提升首个相关文档的排名。

项目聚焦的不是问答生成，而是检索前的查询改写。模型最终只需要输出一行搜索 query，不输出解释、推理过程、标签、Markdown 或多候选项。这个目标决定了整个系统的奖励函数、prompt、训练约束和评估方式都围绕检索指标展开。

本阶段主线任务可以概括为：

1. 使用 Qwen3-4B-Instruct-2507 作为基座模型，构建 LoRA query rewriter。
2. 使用 Pyserini / Lucene BM25 在 MS MARCO passage dev subset 上直接计算检索收益。
3. 用手写 GRPO clipped objective 做强化学习更新，不依赖 TRL 等高层 RL 框架。
4. 引入 top20_delta 奖励，把目标从泛化的“改写得像搜索词”收束到 MRR@20、Recall@20、Recall@50 的实际检索收益。
5. 引入 curriculum 课程采样，把训练样本按原始 query 的检索难度分桶，避免大量无学习信号或过强样本冲淡训练。
6. 通过两阶段训练和三个 Phase2 变体 sweep，比较不同 reward 权重与采样策略的效果。
7. 在完整验证集上对 Original、Zero-shot、RL adapter 三路策略做统一评估。

## 2. 项目做了什么

项目实现了一条从数据加载、课程元数据构建、GRPO 训练、checkpoint 管理、变体扫描，到最终全量评估和 leaderboard 汇总的闭环。

### 2.1 数据与检索基线

系统使用 `msmarco-passage-dev-subset` 的 topics / qrels 作为监督信号，检索器使用 Pyserini 预构建的 `msmarco-v1-passage` Lucene BM25 索引。每个 query 的 reward 不是来自人工标注的“好改写文本”，而是来自改写后 query 在 BM25 检索结果中的真实表现。

数据切分采用固定随机种子和 train / validation split。训练阶段使用 train split 采样，阶段内周期评估使用 validation 子集；最终 `eval_compare.py` 则在同一切分逻辑下，对完整验证查询进行三路比较。本次 full eval 的评估查询数为 1396 条。

### 2.2 模型与参数高效训练

模型主线为 `/root/autodl-tmp/hf_models/Qwen3-4B-Instruct-2507`，即 Qwen3-4B-Instruct-2507 本地基座。训练采用 4bit 量化加载 actor / reference，并在 actor 上挂载 LoRA adapter。

LoRA 目标模块覆盖注意力和 MLP 关键投影层：

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

默认 LoRA 配置为 `r=16`、`alpha=32`、`dropout=0.05`。这种设计使项目可以在 4B 模型上完成强化学习微调，同时保留 frozen reference model 用于 KL 约束，降低策略漂移风险。

### 2.3 Prompt 约束

项目使用面向 BM25 top-20 的 lexical rewrite prompt。prompt 的核心约束是：

- 输出必须是单行英文检索 query。
- 不允许输出解释、答案、标签、XML、Markdown、`<think>`、多候选项或推理过程。
- 查询长度倾向于 3 到 11 个有效词。
- 保留实体、稀有技术词、缩写、数字、年份、版本号、单位和否定词。
- 如果原 query 已经很强，则只做小幅安全改写，避免大幅语义漂移。
- 优先使用可能在 MS MARCO passage 文本中逐字出现的词。

这个 prompt 与 reward 的目标一致：不是让模型变得“更会回答”，而是让模型变得“更会写 BM25 检索式”。

## 3. 如何做的

本项目的主要方法是“检索指标驱动的 GRPO query rewrite”。每轮训练不是直接学习参考答案，而是让模型为同一个 query 生成一组候选改写，使用 BM25 检索结果给每个候选打分，然后用组内相对优势更新 actor。

### 3.1 GRPO 训练核心

每个训练 step 的关键过程如下：

1. 从课程采样后的训练 query 中取 batch。
2. 对每个 query 生成一组候选 rewrite。
3. 对原始模型输出进行清洗、截断和 guardrail 兜底，得到最终送入检索器的 query。
4. 使用 BM25 检索并根据 qrels 计算 MRR@20、Recall@20、Recall@50 和 rank bonus。
5. 组合 top20_delta reward，同时扣除格式错误、过度改写、copy、recall 下降等惩罚。
6. 在同一 query 的候选组内标准化 reward，得到 advantage。
7. 重新计算当前 actor 的 logprob，并与采样时 old logprob 构成 PPO clipped surrogate objective。
8. 使用 frozen reference model 计算 KL 项，把策略更新限制在可控范围内。
9. 只更新 LoRA adapter，并周期性在 validation split 上评估，保存 best / latest checkpoint。

这种训练方式的好处是可以直接优化检索收益，不需要事先构造“标准改写文本”。模型学到的是相对于同组候选而言哪个 query 更能提升检索，而不是简单模仿某种固定写法。

### 3.2 top20_delta 奖励设计

本阶段的奖励围绕 top-20 检索窗口设计，主指标为：

- `MRR@20`：首个相关文档越靠前越好，是主优化方向。
- `Recall@20`：top-20 中召回相关文档的比例，防止只优化单个命中。
- `Recall@50`：辅助召回指标，增强对候选相关文档覆盖的关注。
- `rank_bonus`：对首个相关文档出现在更靠前位置给予额外奖励。

reward 使用 delta 思路，即不仅看 rewrite 的绝对检索表现，也看它相对原始 query 的变化。这样可以避免模型把本来已经很好的 query 改坏，也可以鼓励模型在原 query 排名较差但仍有可召回空间时做有价值的改写。

同时加入以下保护项：

- `recall_drop_penalty`：当 rewrite 的 recall 低于原 query 时惩罚，避免为提升 MRR 牺牲召回。
- `overedit_penalty`：源 query 的关键内容保留不足时惩罚，限制语义漂移。
- `bad_format_penalty`：惩罚多行、解释文字、标签污染、不可读字符、过长输出等。
- `unsafe_copy_penalty`：当 query 明显需要改写但模型直接复制时惩罚。
- `anchor_bonus`：当 rewrite 同时不低于原 query 的 MRR 和 recall 时给予稳定奖励。

最终 reward 既推动 MRR，也维护 recall、格式和语义安全。

### 3.3 Curriculum 课程采样

项目为 train split 中每条 query 构建课程元数据，记录原始 query 的：

- `orig_mrr20`
- `orig_recall20`
- `orig_recall100`
- `orig_best_hit_rank`
- bucket 类型

课程分桶逻辑如下：

- A 桶：原 query 在 Recall@100 中能找到相关文档，但 MRR@20 不好。这类样本最适合学习 top-20 排名改写。
- B 桶：原 query 几乎找不到相关文档，但文本质量仍可用。这类样本用于扩大探索。
- C 桶：原 query 已经较强。这类样本少量保留，防止模型只学会激进改写。
- DROP：缺少有效学习信号或文本质量不适合训练的样本，不进入主线训练。

两阶段训练采用不同采样权重：

| 阶段 | A 桶 | B 桶 | C 桶 | 设计意图 |
| --- | ---: | ---: | ---: | --- |
| Phase1 | 80% | 10% | 10% | recall-first，优先学习有明确改写空间的 query |
| Phase2 | 65% | 25% | 10% | 在已有 adapter 基础上增加探索和泛化压力 |

这套 curriculum 的意义在于，把训练注意力集中到“原 query 有可召回信号但 top-20 排名不足”的样本上，同时保留少量失败样本和强样本，让模型不过度依赖单一类型 query。

## 4. 本次主实验流水线

`run_train_then_full_eval_4b_top20_delta_curriculum.sh` 是本次总结依据的主实验脚本。它把 Phase1、Phase2 sweep 和 full eval 串成完整实验。

流水线结构如下：

1. 创建统一 run root、eval 目录和 sweep manifest。
2. 运行 Phase1 recall-first 训练，得到 `phase1/checkpoints/best` 和 `phase1/checkpoints/latest`。
3. 从 Phase1 的 best checkpoint 进入 Phase2。
4. 依次训练三个 Phase2 变体：`mrr_strict`、`mrr_balanced`、`mrr_diverse`。
5. 每个 Phase2 变体训练完成后，用其 best checkpoint 做 `eval_compare.py` 三路对比。
6. 写入 `phase2_variant_manifest.jsonl`。
7. 汇总 `phase2_sweep_summary.json` 和 `phase2_sweep_leaderboard.csv`。
8. 按 `RL - Zero-shot MRR` 排序选出 winner。

本次 run 的根目录为：

`train_and_eval_data_model_0423/artifacts_4b_top20_delta_curriculum_sweeps/run_20260423_161608`

### 4.1 Phase1 配置取向

Phase1 的目标是 recall-first。它更重视召回和稳定改写，参数取向包括：

| 项目 | 值 |
| --- | ---: |
| max steps | 300 |
| 实际训练到 | 180 |
| eval 间隔 | 20 steps |
| learning rate | `1.0e-5` |
| KL beta | `0.035` |
| temperature / top-p | `0.90 / 0.96` |
| group temperature stride | `0.14` |
| group top-p stride | `0.03` |
| reward gap threshold | `0.04` |
| reward 权重 | MRR 0.28 / Recall@20 0.42 / Recall@50 0.20 / Rank 0.10 |
| recall drop lambda | `2.00` |
| anchor bonus | `0.02` |

Phase1 实际在 step 180 触发 `significant_drop_stop`。训练日志中 best eval 出现在 step 100：

| 指标 | Phase1 best eval |
| --- | ---: |
| MRR@20 | 0.1959 |
| Recall@20 | 0.4742 |
| Recall@50 | 0.5900 |

Phase1 的角色不是最终交付模型，而是为 Phase2 提供一个相对稳定、具备检索改写能力的初始化 adapter。

### 4.2 Phase2 sweep 设计

Phase2 从 Phase1 best checkpoint 出发，分别训练三个变体。三个变体的核心差异是 reward 权重、KL 强度、采样温度和探索力度。

| 变体 | 学习率 | KL beta | temperature / top-p | reward 权重 MRR / R20 / R50 / Rank | 设计取向 |
| --- | ---: | ---: | ---: | ---: | --- |
| `mrr_strict` | `6e-6` | `0.045` | `0.82 / 0.93` | `0.70 / 0.16 / 0.08 / 0.06` | 最强 MRR 导向，较克制探索 |
| `mrr_balanced` | `5e-6` | `0.050` | `0.84 / 0.94` | `0.62 / 0.20 / 0.10 / 0.08` | MRR 与 recall 平衡 |
| `mrr_diverse` | `7e-6` | `0.040` | `0.88 / 0.96` | `0.66 / 0.18 / 0.10 / 0.06` | 更强探索和多样性 |

三个变体均配置 `max_steps=200`，每 20 step 做一次阶段内 eval。训练日志显示：

| 变体 | 实际训练步数 | eval 次数 | 阶段内 best step | 阶段内 best MRR@20 | 阶段内 best Recall@20 | 阶段内 best Recall@50 | 停止状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `mrr_strict` | 200 | 10 | 140 | 0.1973 | 0.4842 | 0.5963 | within_tolerance |
| `mrr_balanced` | 200 | 10 | 140 | 0.1976 | 0.4817 | 0.5888 | within_tolerance |
| `mrr_diverse` | 200 | 10 | 140 | 0.1982 | 0.4729 | 0.5800 | significant_drop_stop |

阶段内 eval 只反映训练过程中的 validation 监控；最终结论以 full eval 的三路对比为准。

## 5. Full Eval 评估方式

本次指定 eval 目录包含三个 report：

- `eval_compare_report_mrr_strict.json`
- `eval_compare_report_mrr_balanced.json`
- `eval_compare_report_mrr_diverse.json`

每个 report 都在同一批 1396 条验证 query 上比较：

1. `Original`：原始 query 直接进入 BM25。
2. `Zero-shot`：基座模型不加载 RL adapter，仅按 prompt 改写。
3. `RL`：加载对应 Phase2 best LoRA adapter 后改写。

评估时使用：

- `MRR@20`
- `Recall@20`
- `Recall@50`
- reward 拆解均值
- 每个 qid 的原始 query、zero-shot rewrite、RL rewrite 和三路 reward 明细

最终 sweep leaderboard 以 `RL MRR - Zero-shot MRR` 排序，并检查是否达到目标阈值 `+0.01`。

## 6. 实验结果

### 6.1 总体结论

本次实验的核心结论是：

1. 三个 Phase2 RL adapter 相比 Original 都取得了稳定提升。
2. 三个 Phase2 RL adapter 相比各自 Zero-shot 也都取得了正向提升。
3. `mrr_strict` 是 sweep leaderboard 的 winner，因为它的 `RL - Zero-shot MRR` 最大。
4. `mrr_diverse` 的 RL 绝对 MRR@20 和 Recall@20 最高，但它对应的 Zero-shot 基线也最高，因此相对 Zero-shot 的增量不是最大。
5. 预设成功标准 `RL - Zero-shot MRR >= +0.01` 尚未达到，最佳结果为 `+0.00685`。

这说明当前训练已经学到了有效的检索改写能力，但距离脚本设定的强成功目标还有差距。

### 6.2 Leaderboard

`phase2_sweep_summary.json` 给出的 leaderboard 如下：

| 排名 | 变体 | RL - Zero-shot MRR | RL - Original MRR | RL Recall@20 | 是否达到 +0.01 目标 |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | `mrr_strict` | +0.00685 | +0.00910 | 0.49654 | 否 |
| 2 | `mrr_balanced` | +0.00504 | +0.00876 | 0.49833 | 否 |
| 3 | `mrr_diverse` | +0.00350 | +0.00930 | 0.50084 | 否 |

winner 为 `mrr_strict`，其 best checkpoint 为：

`train_and_eval_data_model_0423/artifacts_4b_top20_delta_curriculum_sweeps/run_20260423_161608/phase2_sweep/mrr_strict/checkpoints/best`

### 6.3 三路 MRR@20 对比

| 变体 | Original MRR@20 | Zero-shot MRR@20 | RL MRR@20 | Zero-shot - Original | RL - Original | RL - Zero-shot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mrr_strict` | 0.20224 | 0.20449 | 0.21134 | +0.00225 | +0.00910 | +0.00685 |
| `mrr_balanced` | 0.20224 | 0.20595 | 0.21100 | +0.00371 | +0.00876 | +0.00504 |
| `mrr_diverse` | 0.20224 | 0.20804 | 0.21154 | +0.00581 | +0.00930 | +0.00350 |

从绝对 MRR 看，三个 RL adapter 非常接近：

- `mrr_diverse`：0.21154，绝对值最高。
- `mrr_strict`：0.21134，与最高值只差约 0.00020。
- `mrr_balanced`：0.21100，同样保持在 0.211 左右。

从相对 Zero-shot 增量看，`mrr_strict` 最好。这符合它的设计：更高 MRR 权重、更低探索温度、更严格的优化方向，使 RL adapter 相比同配置 zero-shot 的提升最大。

### 6.4 Recall 对比

| 变体 | Original Recall@20 | Zero-shot Recall@20 | RL Recall@20 | Original Recall@50 | Zero-shot Recall@50 | RL Recall@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mrr_strict` | 0.48651 | 0.48949 | 0.49654 | 0.59921 | 0.60805 | 0.61605 |
| `mrr_balanced` | 0.48651 | 0.48508 | 0.49833 | 0.59921 | 0.60387 | 0.61748 |
| `mrr_diverse` | 0.48651 | 0.49081 | 0.50084 | 0.59921 | 0.60817 | 0.61139 |

Recall 结果说明，RL 并不是只把个别相关文档推到更前，同时也整体改善了召回：

- `mrr_strict` 相比 Original：Recall@20 +0.01003，Recall@50 +0.01683。
- `mrr_balanced` 相比 Original：Recall@20 +0.01182，Recall@50 +0.01827。
- `mrr_diverse` 相比 Original：Recall@20 +0.01433，Recall@50 +0.01218。

其中 `mrr_diverse` 的 Recall@20 最高，体现了更强探索配置带来的召回收益；`mrr_balanced` 的 Recall@50 最高，体现了 balanced 权重对更宽召回窗口的友好性；`mrr_strict` 虽然 recall 不是最高，但在 MRR 增量上表现最强。

### 6.5 Reward 与稳定性指标

| 变体 | Original reward_mean | Zero-shot reward_mean | RL reward_mean | RL main_reward_mean | keyword_preserve | bad_format_penalty | unsafe_copy_penalty | recall_drop_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mrr_strict` | -0.06884 | 0.00182 | 0.03371 | 0.03997 | 0.92419 | 0.00000 | 0.00072 | 0.01719 |
| `mrr_balanced` | -0.06884 | -0.00023 | 0.03198 | 0.04059 | 0.91197 | 0.00000 | 0.00287 | 0.02006 |
| `mrr_diverse` | -0.06884 | 0.00660 | 0.03652 | 0.04032 | 0.93759 | 0.00000 | 0.00000 | 0.01361 |

这些指标说明：

- RL 三个变体都显著改善了 reward_mean。
- 三个变体的 `bad_format_penalty` 都为 0，说明最终输出基本满足单行搜索 query 的格式约束。
- `keyword_preserve` 均在 0.91 以上，说明强化学习没有靠大幅语义漂移获得检索收益。
- `recall_drop_ratio` 控制在约 1.36% 到 2.01%，说明 recall 下降的样本比例较低。
- `mrr_diverse` 在 keyword preservation 和 unsafe copy penalty 上最好，但相对 Zero-shot 的 MRR 增量较小。

### 6.6 Per-query 改善分布

full eval 中大量 query 的 MRR 保持不变，这是稀疏检索 top-k 指标的常见现象：许多改写不会改变首个相关文档是否进入 top-20，或原 query / rewrite 命中情况完全一致。因此除了均值，也需要看 per-query 改善分布。

| 变体 | RL MRR > Zero | RL MRR = Zero | RL MRR < Zero | RL MRR > Original | RL MRR = Original | RL MRR < Original |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mrr_strict` | 136 | 1138 | 122 | 198 | 1118 | 80 |
| `mrr_balanced` | 128 | 1172 | 96 | 213 | 1098 | 85 |
| `mrr_diverse` | 153 | 1113 | 130 | 191 | 1136 | 69 |

从这个分布可以看到：

- `mrr_strict` 相比 Zero-shot 的提升样本数略高于下降样本数，并且提升幅度在均值上更强。
- `mrr_balanced` 相比 Original 的正向样本数最多，为 213 条。
- `mrr_diverse` 相比 Original 的下降样本数最少，为 69 条，同时绝对 MRR 和 Recall@20 最高。

这说明三个变体各有侧重：`mrr_strict` 更适合作为 leaderboard winner，`mrr_diverse` 更像稳定泛化候选，`mrr_balanced` 则在 Original 对比上有较多正向样本。

## 7. 项目取得的成果

### 7.1 方法层成果

项目证明了一个可行方向：可以不依赖人工 query rewrite 标注，直接用检索器和 qrels 构造 reward，对大语言模型进行检索目标驱动的强化学习。

本阶段不是简单 prompt engineering，而是完成了以下方法组合：

- BM25 检索结果直接反馈到 RL reward。
- 以 MRR@20 为主目标，Recall@20 / Recall@50 为辅助目标。
- 使用 delta reward，使模型关注“比原 query 更好”而不是单纯追求绝对分数。
- 使用 curriculum 分桶，让训练样本具有更强学习信号。
- 用组内多候选采样和 advantage 标准化降低 query 难度差异带来的 reward 尺度问题。
- 用 KL 约束和 LoRA 微调保持模型稳定。
- 用 format / overedit / recall-drop / unsafe-copy 惩罚控制生成质量。

### 7.2 工程层成果

项目已经形成了比较完整的工程闭环：

- `train.py` 负责主训练逻辑、配置覆盖、课程采样、周期评估和 checkpoint 保存。
- `core/grpo_engine.py` 实现多候选 rollout、reward 打分、advantage 标准化、PPO clipped objective 和 KL 更新。
- `core/reward_func.py` 实现 BM25 检索指标、top20_delta reward、格式惩罚、语义保留和兜底逻辑。
- `core/model_wrapper.py` 封装 4bit 模型加载、LoRA adapter、reference model、批量生成和 logprob 计算。
- `data/curriculum.py` 实现原 query 难度扫描、bucket 构建、DROP 过滤和 phase-aware 采样。
- `eval_compare.py` 实现 Original / Zero-shot / RL 三路对比，并输出完整 per-qid report。
- `run_train_then_full_eval_4b_top20_delta_curriculum.sh` 实现两阶段训练、三变体 sweep、full eval 和 leaderboard 汇总。

这套结构的价值是实验可追踪：每个阶段都有独立 checkpoint、训练日志、group trace、full eval report、manifest 和 leaderboard。

### 7.3 实验层成果

本次 `run_20260423_161608` 实验取得了明确正收益：

- 最佳 `RL - Zero-shot MRR@20`：`+0.00685`。
- 最佳 `RL - Original MRR@20`：`+0.00930`。
- 最佳 RL MRR@20：`0.21154`。
- 最佳 RL Recall@20：`0.50084`。
- 最佳 RL Recall@50：`0.61748`。
- 三个 Phase2 adapter 全部超过 Original。
- 三个 Phase2 adapter 全部超过各自 Zero-shot。
- 输出格式控制良好，full eval 中 RL 的 `bad_format_penalty_mean` 为 0。
- 关键词保留稳定，RL 的 `keyword_preserve_mean` 均超过 0.91。

这些结果说明，项目已经从“能跑通 RL query rewrite”推进到了“能在完整验证集上稳定超过原 query 和 zero-shot prompt”的阶段。

## 8. 结果解读

### 8.1 为什么 `mrr_strict` 赢

`mrr_strict` 的 reward 中 MRR 权重最高，为 0.70，同时温度较低、KL 适中、anchor bonus 为 0。这让训练更集中地优化首个相关文档的前排命中，而不是追求更大范围的探索。

从 full eval 看，它不是绝对 MRR 最高，但相对自己的 Zero-shot 基线提升最大。脚本的 winner 排序标准是 `RL - Zero-shot MRR`，所以 `mrr_strict` 成为最终 winner。

### 8.2 为什么 `mrr_diverse` 绝对值最高却不是 winner

`mrr_diverse` 的温度和 top-p 更高，组内采样跨度更大，B 桶探索也更多。这种配置让它获得了最高的 RL MRR@20 和 Recall@20：

- RL MRR@20：0.21154
- RL Recall@20：0.50084

但它的 Zero-shot 基线也更高，Zero-shot MRR@20 达到 0.20804。因此它的 `RL - Zero-shot` 只有 +0.00350，低于 `mrr_strict`。

这说明 `mrr_diverse` 的 prompt / decoding 配置本身已经带来较强 zero-shot 表现，RL adapter 的边际增益被压缩。


## 9. 当前局限

1. 相比 Zero-shot 的提升尚未达到 +0.01 目标。
2. per-query 中 MRR 不变的样本仍占多数，说明模型有效影响检索排名的覆盖面有限。
3. Phase2 的阶段内 eval 和 full eval 结论并不完全一致，说明 400 条阶段评估样本与 1396 条 full eval 之间仍存在采样差异。
4. `mrr_diverse` 在训练后期触发 significant drop，说明高探索配置仍有稳定性风险。
5. 当前主要依赖 BM25 sparse lexical 检索，尚未验证 dense / hybrid 检索场景下的迁移效果。
6. 目前 leaderboard 只基于单个 run，尚未形成多 seed 统计显著性结论。

## 10. 后续方向

后续优化可以围绕以下方向展开：

1. 提高 `RL - Zero-shot` 增量：重点分析 RL 低于 Zero-shot 的样本，定位是否由过度改写、实体丢失、词形选择或 recall 下降导致。
2. 扩大正向覆盖率：针对大量 MRR 不变样本，增加能改变 top-20 排名的候选生成多样性，但继续保留格式和语义约束。
3. 调整 Phase2 目标：可以在 `mrr_strict` 的基础上小幅提高 recall 权重，吸收 `mrr_diverse` 的召回优势。
4. 改进课程采样：继续细分 A 桶，例如区分“相关文档在 21-50 位”和“相关文档在 51-100 位”的样本。
5. 强化负样本诊断：对 RL 明显低于 Original / Zero-shot 的 query 做类别归因，形成 targeted penalty。
6. 做多 seed 验证：使用相同脚本结构重复不同 seed，确认 MRR 提升是否稳定。
7. 增加人工可读性抽样：结合 per-qid report 观察 RL rewrite 是否符合检索式直觉，避免只看指标。

## 11. 阶段性结论

DeepRetrieval-GRPO 当前已经完成了一套完整的检索强化学习训练体系。它把 BM25 检索指标、MS MARCO qrels、Qwen3-4B LoRA、GRPO 组内相对优势、top20_delta reward、课程采样和 full eval sweep 连接成一个可验证的闭环。

从 `run_20260423_161608` 的结果看，项目已经取得实质性成果：RL adapter 在 1396 条验证 query 上稳定超过 Original 和 Zero-shot，最佳变体 `mrr_strict` 相比 Zero-shot 提升 MRR@20 `+0.00685`，相比 Original 提升 `+0.00910`。虽然尚未达到脚本设定的 `+0.01` 强目标，但方向明确、收益真实、工程链路完整，已经具备继续做更细粒度调参、错误分析和多 seed 验证的基础。
