# DeepRetrieval-GRPO 训练思路与实验总结（2026-04-21）

## 1. 这份文档要解决什么问题

这份文档的目标不是再复述一遍命令，而是把当前仓库里已经形成的训练闭环讲清楚：

- 现在到底在训练什么。
- 奖励函数为什么这样设计。
- 训练时做了哪些防塌缩、防脏输出、防过度改写的处理。
- 当前有哪些主实验线。
- 各条实验线的关键超参数是什么。
- 仓库里已经有哪些结果，哪些还只是脚本和 smoke run。

这份总结主要基于以下代码和产物：

- `app_config.py`
- `train.py`
- `core/grpo_engine.py`
- `core/reward_func.py`
- `core/model_wrapper.py`
- `data/loader.py`
- `data/curriculum.py`
- `eval_compare.py`
- `run_train_then_full_eval_4b.sh`
- `run_train_then_full_eval_4b_top20_delta_curriculum.sh`
- `analysis_outputs/download_logs_20260420/summary.json`
- `train_and_eval_data_model/*`

## 2. 一句话概括当前训练主线

当前项目的核心任务是：

- 把用户原始 query 改写成更适合 `Lucene BM25 + MS MARCO passage` 的单行英文检索 query。
- 用真实检索结果产生奖励，而不是用语言模型自评。
- 用 `Qwen + QLoRA + 手写 GRPO` 做参数高效强化学习微调。

换句话说，这不是在训练“更会说话”的改写模型，而是在训练“更会给 BM25 喂 query”的改写模型。

## 3. 当前总训练思路

## 3.1 总目标

总目标可以拆成两层：

- 主目标：提升 rewrite 后 query 的检索质量，核心看 `MRR` / `Recall`。
- 约束目标：不要把 query 改坏，尤其不要丢掉数字、缩写、否定词、命名实体，也不要输出脏格式、解释文本、多行内容或无意义重复。

所以整个训练不是“尽量多改”，而是“在尽量不破坏原意和关键约束的前提下，做小而有效的 lexical rewrite”。

## 3.2 当前有两条主要实验线

### A. `legacy / conservative_mrr` 主线

这条线是当前最稳妥的主实验思路：

- 直接优化 rewrite 本身的绝对检索表现。
- 奖励里同时加入 `MRR / Recall / Recall_dense` 和保守约束项。
- 偏向“保守增益”：宁可小改，也尽量不要过度改写。
- 4B 主实验脚本 `run_train_then_full_eval_4b.sh` 走的就是这条线。

### B. `top20_delta + curriculum` 新方向

这条线是更“针对检索排名”的新实验：

- 不再奖励 rewrite 的绝对分数，而是奖励“相对原 query 的提升”。
- 重点优化 `MRR@20 / Recall@20 / Recall@50 / rank bonus`。
- 加入 curriculum，把训练 query 分阶段采样。
- 两阶段训练，phase2 从 phase1 的 best adapter 热启动。
- 目前仓库里只有 smoke 产物，完整 4B 结果还没提交到仓库。

## 4. 端到端训练流程

## 4.1 数据来源

训练和评估统一使用 Pyserini 预编译资源：

- topic：`msmarco-passage-dev-subset`
- index：`msmarco-v1-passage`

数据处理规则：

- 只保留有正相关文档的 qid。
- 固定 `seed=42` 做 train/val 切分。
- 默认 `train_ratio=0.8`。
- 可以通过 `max_train_queries` / `max_val_queries` 截断样本，用于 smoke 或控制训练成本。

## 4.2 训练前的 query 过滤逻辑

非 curriculum 模式下，训练前还会过滤一部分“已经像检索 query”的样本：

- token 数在 `2~8` 之间；
- 不含问号；
- 不含典型疑问词。

也就是说，如果原始 query 已经很像紧凑 BM25 query，就先不拿它做训练，以免模型学到“什么都别改”。

## 4.3 模型结构

训练时模型被拆成两部分：

- actor：可训练策略模型。
- ref：冻结参考模型，只用于 KL 约束。

实现方式：

- actor 默认走 `4bit QLoRA`。
- ref 可以走 `auto / full / 4bit` 三种模式。
- 在 24GB 左右显存上，如果是 4bit actor + auto ref，运行时会自动偏向更保守的 `ref_precision_mode=4bit`，优先保住稳定性。

## 4.4 LoRA 配置

当前默认 LoRA 结构：

- `r = 16`
- `alpha = 32`
- `dropout = 0.05`
- 目标模块：
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`

这是一个比较标准的 QLoRA causal LM 配置，重点改注意力投影和 MLP 投影层。

## 4.5 每个训练 step 在做什么

每个 mini-batch 里的每条 query，训练流程如下：

1. 构造 prompt。
2. actor 对同一条 query 采样一组 rewrite。
3. 对生成结果做清洗和稳定化。
4. 用 Pyserini 检索，计算每个 rewrite 的 reward。
5. 在组内对 reward 做标准化，得到相对 advantage。
6. 用 PPO clipped objective + KL 项更新 actor 的 LoRA 参数。
7. 周期性在验证集上跑整体验证。
8. 保存 `latest`，并按验证指标更新 `best`。

## 5. 采样与防塌缩设计

这部分是当前训练代码最关键的工程细节之一。

## 5.1 组采样不是固定死的

训练不是简单地“每个 query 采 8 条就结束”，而是分两层做多样性控制：

- 初始组采样：先采 `group_size` 条。
- 自适应补采样：如果当前组里 reward spread 不够大，就继续补样本，直到：
  - `reward_gap_raw >= reward_gap_threshold`
  - 或者达到 `max_group_size`

这意味着：

- GRPO 看到的不只是一个固定大小 group。
- 如果当前 query 很难拉开候选差异，系统会主动补充更多样本。
- 目标是让组内 reward 有足够对比度，advantage 才更有用。

## 5.2 组内 decode 参数会轻微错开

同一组样本不是完全相同 decode 参数，而是会按索引做轻微 stride：

- `group_temperature_stride`
- `group_top_p_stride`

这相当于故意给同一条 query 加一点解码扰动，提升组内多样性。

## 5.3 重采样机制

如果一组 rewrite 太“塌”，会触发额外重采样：

- 唯一 final query 数低于 `min_unique_final_queries`
- 输出标带签污染、多行、脏格式
- 与原 query 完全相同

这时会对部分槽位做更高温度的 regenerate，替换掉：

- duplicate
- polluted
- exact-copy
- fallback-to-original

这一步的目的不是直接提高 reward，而是先保证 group 有训练价值。

## 5.4 什么情况会被视为“组塌缩”

如果一个 group 最终只有一个唯一 final query，就认为这个 group collapsed。

collapsed group：

- 仍会被记日志；
- 但不会进入有效 backward 更新路径。

这是为了避免“所有候选都一样”时，组内 advantage 全部失去意义。

## 6. 输出清洗与 guardrail

## 6.1 当前代码里的 rewrite 清洗逻辑

生成文本会经过如下处理：

- 截断 stop marker。
- 尝试从多行文本里提取真正的 query 行。
- 过滤 `rewritten query:`、`search query:`、`better bm25 query:` 等模板污染。
- 删除多余引号、编号、前缀和明显 explanation。

当前代码里真正会直接 fallback 到原 query 的条件很保守：

- 仅当清洗后变成空字符串时，才会触发 `fallback_to_original`。

也就是说：

- 多行、带标签、带 explanation 的输出主要靠清洗和惩罚处理；
- 不是所有不理想输出都会自动 fallback。

## 6.2 当前约束重点

当前 reward 设计里，约束主要围绕下面几类风险：

- 关键术语丢失
- query 太长或太短
- 输出脏格式
- 对不成熟原 query 做危险的原样复制
- 在 `top20_delta` 模式下过度改写

## 6.3 锁定词保护

当前明确重点保护的内容有：

- 数字
- 大写缩写
- 否定词

相关指标：

- `keyword_preserve`
- `locked_term_preserve`
- `term_preserve = 0.5 * keyword_preserve + 0.5 * locked_term_preserve`

这意味着模型即使想换词，也最好别把数字、版本、缩写、否定条件弄丢。

## 7. 奖励函数设计

## 7.1 `legacy` 模式：绝对奖励

当前默认代码配置是 `reward_mode = legacy`。

公式如下：

```text
total =
  w_mrr * mrr
  + w_recall * recall
  + w_recall_dense * recall_dense
  + w_term_preserve * term_preserve
  + w_length_score * length_score
  + w_clean_format * clean_format
  - w_bad_format * bad_format_penalty
  - w_unsafe_copy * unsafe_copy_penalty
```

当前默认权重：

```text
w_mrr = 0.40
w_recall = 0.20
w_recall_dense = 0.15
w_term_preserve = 0.10
w_length_score = 0.08
w_clean_format = 0.07
w_bad_format = 0.15
w_unsafe_copy = 0.08
```

这条线的理念是：

- 检索收益是主导；
- 但为了稳定，仍显式奖励“保留必要词”和“干净输出”。

## 7.2 `top20_delta` 模式：相对提升奖励

`top20_delta` 不再奖励 rewrite 的绝对值，而是奖励它相对原 query 的增量：

```text
main_reward =
  w_mrr * (mrr - orig_mrr)
  + w_recall * (recall - orig_recall)
  + w_recall_dense * (recall_dense - orig_recall_aux)
  + w_rank_bonus * (rank_bonus - orig_rank_bonus)

total =
  main_reward
  - w_bad_format * bad_format_penalty
  - w_unsafe_copy * unsafe_copy_penalty
  - w_overedit * overedit_penalty
```

这条线的理念是：

- 原 query 本来就可能很强。
- 训练目标不该是“让模型输出高分 query”，而是“比原 query 更好”。
- 所以最重要的是 `delta_mrr`、`delta_recall` 和 top-rank 改善。

## 7.3 排名 bonus

`rank_bonus` 会对命中更靠前的 relevant doc 给额外奖励：

- rank=1 -> 1.0
- rank<=3 -> 0.8
- rank<=5 -> 0.5
- rank<=10 -> 0.3
- rank<=20 -> 0.15
- rank<=50 -> 0.05

这个 bonus 尤其适合 `top20_delta`，因为它更强调“把相关文档顶到更前面”。

## 7.4 长度分数

长度分数不是简单 hard cutoff，而是分段函数：

- 太短接近 0
- 在理想区间给满分
- 太长再逐渐降回 0

当前默认区间：

- `min_terms = 1`
- `ideal_min = 4`
- `ideal_max = 12`
- `max_terms = 20`

## 7.5 格式惩罚

坏格式惩罚会综合考虑：

- 是否多行
- 是否出现 explanation / template pollution
- token 数是否超阈值
- 英文字母比例是否过低
- 不可读字符比例是否过高

这个设计的目标是让 reward 直接对“输出是否像合格检索 query”负责。

## 7.6 危险复制惩罚

`unsafe_copy_penalty` 只在一种情况下触发：

- 原 query 本身还不 retrieval-ready；
- rewrite 和原 query 完全一样。

也就是说，如果原 query 已经足够像检索 query，保留它本身并不会被罚。

## 7.7 过度改写惩罚

`top20_delta` 模式额外引入：

```text
overedit_penalty = max(0, overedit_tau - keyword_preserve)
```

默认：

- `overedit_tau = 0.40`

这个项的含义很直接：

- 允许改写；
- 但如果关键词保留率掉得太厉害，就说明 rewrite 过头了。

## 8. GRPO / PPO 更新细节

## 8.1 advantage 的来源

当前 advantage 不是来自 critic，而是同一 group 内 reward 标准化：

```text
advantage_i = (reward_i - group_mean) / group_std
```

这就是当前手写 GRPO 的核心。

## 8.2 PPO 损失

每条 sample 都会重新计算：

- `logprob_new`：当前 actor 对响应 token 的 logprob
- `logprob_old`：rollout 时旧策略的 logprob
- `logprob_ref`：冻结 ref 对同一响应的 logprob

损失：

```text
loss_pg = -mean(min(ratio * adv, clipped_ratio * adv))
loss_kl = kl_beta * mean(logprob_new - logprob_ref)
loss = loss_pg + loss_kl
```

## 8.3 内存优化细节

当前代码里为了把训练压进更小显存，做了几件事：

- actor 走 4bit QLoRA
- ref 支持 4bit / auto fallback
- actor logprob 重算按 `actor_chunk_size` 分块
- `lm_head` 投影按 `projection_chunk_size` 分块
- 梯度累加是“即时 backward”，避免整 batch graph 一直留在显存里

## 8.4 梯度更新

当前逻辑是：

- 先对所有 valid samples 累积梯度；
- 再按 `1 / valid_samples` 把梯度缩回平均值；
- 做 `clip_grad_norm_`；
- 然后 `optimizer.step()`。

## 9. 验证、checkpoint 和 early stop

## 9.1 验证怎么做

验证集评估时：

- actor 切到 eval 模式；
- 按配置生成 rewrite；
- 同样做清洗和稳定化；
- 用和训练一致的 rewarder 重新打分；
- 聚合成 step 级验证指标。

## 9.2 `best` checkpoint 按什么保存

每次 eval 都会保存 `latest`。

`best` 的更新依据是：

- `current_eval_mrr = eval_metrics["rewrite_mrr20_mean"]`

这里有一个历史命名问题要特别注意：

- 字段名叫 `rewrite_mrr20_mean`
- 但它实际承载的是当前 reward 配置下的 `mrr_mean`
- 如果当前 `reward_mrr_k = 50`，这个字段名仍然不会变成 `rewrite_mrr50_mean`

所以以后读日志时，不要被字段名里的 `20` 误导。

## 9.3 early stop 条件

当前 early stop 是一个经验性组合规则，满足任一项就停止：

- `rewrite_mrr20_mean` 连续不提升
- `flat_main_reward_group_ratio` 持续变坏
- `kl_dominance_ratio` 持续上升，同时 `delta_mrr20_positive_ratio` 持续下降

当前默认：

- `early_stop_patience = 3`

## 10. 日志与分析产物

训练会写两类 JSONL：

- `train_log.jsonl`
  - step 级训练指标
  - eval 指标
  - loss / reward / mrr / recall / preserve / gap 等聚合值
- `group_trace_log.jsonl`
  - 每条 query 的组采样细节
  - 原始输出、cleaned query、final query
  - 每个样本的 reward / mrr / recall / penalties
  - reward gap、是否 collapsed、补采样温度等

此外还有一个离线分析脚本：

- `summarize_training_logs.py`

这个脚本会把两类日志对齐，输出：

- step 级 CSV
- 曲线图
- summary.json

## 11. 当前代码默认值

这一节说的是“代码默认值”，不是“主实验脚本值”。

## 11.1 数据默认值

| 项目 | 默认值 |
|---|---:|
| `topic_name` | `msmarco-passage-dev-subset` |
| `prebuilt_index` | `msmarco-v1-passage` |
| `train_ratio` | `0.8` |
| `seed` | `42` |
| `max_train_queries` | `20000` |
| `max_val_queries` | `4000` |

## 11.2 模型默认值

| 项目 | 默认值 |
|---|---:|
| `model_name` | `Qwen/Qwen2.5-3B-Instruct` |
| `load_in_4bit` | `true` |
| `bnb_4bit_quant_type` | `nf4` |
| `bnb_4bit_compute_dtype` | `float16` |
| `bnb_4bit_use_double_quant` | `true` |
| `projection_chunk_size` | `64` |
| `ref_precision_mode` | `auto` |
| `lora_r` | `16` |
| `lora_alpha` | `32` |
| `lora_dropout` | `0.05` |

## 11.3 训练默认值

| 项目 | 默认值 |
|---|---:|
| `num_epochs` | `1` |
| `batch_size` | `8` |
| `group_size` | `10` |
| `max_group_size` | `14` |
| `learning_rate` | `1.5e-5` |
| `clip_range` | `0.2` |
| `kl_beta` | `0.03` |
| `grad_clip_norm` | `1.0` |
| `max_new_tokens` | `12` |
| `temperature` | `0.85` |
| `top_p` | `0.95` |
| `group_temperature_stride` | `0.07` |
| `group_top_p_stride` | `0.015` |
| `min_unique_final_queries` | `4` |
| `max_regen_rounds` | `2` |
| `regen_temperature_delta` | `0.15` |
| `reward_gap_threshold` | `0.08` |
| `gap_sampling_temperature_delta` | `0.15` |
| `actor_chunk_size` | `4` |
| `eval_every_steps` | `20` |
| `early_stop_patience` | `3` |
| `curriculum_enable` | `false` |

## 11.4 reward 默认值

| 项目 | 默认值 |
|---|---:|
| `reward_mode` | `legacy` |
| `mrr_k` | `50` |
| `recall_k` | `50` |
| `recall_dense_k` | `100` |
| `search_threads` | `8` |
| `w_mrr` | `0.40` |
| `w_recall` | `0.20` |
| `w_recall_dense` | `0.15` |
| `w_term_preserve` | `0.10` |
| `w_length_score` | `0.08` |
| `w_clean_format` | `0.07` |
| `w_bad_format` | `0.15` |
| `w_unsafe_copy` | `0.08` |
| `w_rank_bonus` | `0.10` |
| `w_overedit` | `0.10` |
| `overedit_tau` | `0.40` |
| `format_max_tokens` | `16` |
| `format_min_english_ratio` | `0.80` |
| `format_max_unreadable_ratio` | `0.30` |
| `bad_format_cap` | `1.5` |

## 11.5 prompt 默认值

当前默认 prompt 是一套偏“BM25 lexical rewrite”的单行英文 query 模板：

- `prompt_id = p24_diverse_lexical`
- 训练目标写死为改写检索 query，而不是解释问题
- 只允许输出最终 query
- 推荐长度 `3~11` 词
- 强调保留实体、数字、缩写、否定
- 默认 eval decode 是保守的：
  - `max_new_tokens = 16`
  - `temperature = 0.0`
  - `top_p = 1.0`
  - `stop_on = "\n"`

如果 reward mode 切到 `top20_delta`，而且还在用 stock prompt id，系统会自动把 prompt 文案改成“优化 top-20 检索”的版本。

## 12. 主实验脚本：`4b_conservative_mrr`

这一节说的是 `run_train_then_full_eval_4b.sh` / `.ps1` 覆盖出来的真正主实验参数。

## 12.1 这条脚本的定位

定位是：

- 直接上 4B 模型
- 用相对保守的 reward 和 decode
- 尽量先做出“比 zero-shot 明显更稳，但不过度激进”的 RL 版本

这也是目前最像“正式实验”的一条脚本线。

## 12.2 Bash 脚本默认值

| 项目 | 默认值 |
|---|---:|
| `ARTIFACT_ROOT` | `train_and_eval_data_model_0420` |
| `EXP_NAME` | `4b_conservative_mrr` |
| `MODEL_NAME` | `/root/autodl-tmp/hf_models/Qwen3-4B-Instruct-2507` |
| `SEARCH_THREADS` | `16` |
| `TRAIN_BATCH_SIZE` | `16` |
| `TRAIN_GROUP_SIZE` | `8` |
| `TRAIN_MAX_GROUP_SIZE` | `12` |
| `ACTOR_CHUNK_SIZE` | `4` |
| `PROJECTION_CHUNK_SIZE` | `64` |
| `TRAIN_MAX_NEW_TOKENS` | `12` |
| `TRAIN_TEMPERATURE` | `0.85` |
| `TRAIN_TOP_P` | `0.95` |
| `EVAL_QUERY_BATCH_SIZE` | `8` |
| `TRAIN_GROUP_TEMPERATURE_STRIDE` | `0.07` |
| `TRAIN_GROUP_TOP_P_STRIDE` | `0.015` |
| `TRAIN_MIN_UNIQUE_FINAL_QUERIES` | `4` |
| `TRAIN_MAX_REGEN_ROUNDS` | `2` |
| `TRAIN_REWARD_GAP_THRESHOLD` | `0.08` |
| `TRAIN_GAP_SAMPLING_TEMPERATURE_DELTA` | `0.15` |
| `TRAIN_EVAL_EVERY_STEPS` | `50` |
| `TRAIN_MAX_STEPS` | `500` |
| `TRAIN_MAX_VAL_QUERIES` | `400` |
| `ref_precision_mode` | `4bit` |

注意：

- Bash 默认模型来源是本地目录。
- PowerShell 默认模型来源是 Hugging Face model id：`Qwen/Qwen3-4B-Instruct-2507`。

## 12.3 这条脚本的 reward 配置

| 项目 | 默认值 |
|---|---:|
| `reward_mode` | `legacy` |
| `reward_mrr_k` | `50` |
| `reward_recall_k` | `50` |
| `reward_recall_dense_k` | `100` |
| `reward_w_mrr` | `0.40` |
| `reward_w_recall` | `0.20` |
| `reward_w_recall_dense` | `0.15` |
| `reward_w_term_preserve` | `0.10` |
| `reward_w_length_score` | `0.08` |
| `reward_w_clean_format` | `0.07` |
| `reward_w_bad_format` | `0.15` |
| `reward_w_unsafe_copy` | `0.08` |
| `format_max_tokens` | `12` |
| `format_min_english_ratio` | `0.8` |
| `format_max_unreadable_ratio` | `0.25` |

这比代码默认值更严格：

- 格式阈值从 `16` 收紧到 `12`
- 不可读字符比例也更严格

这说明主实验脚本明显在往“更保守、更像干净检索 query”的方向收。

## 12.4 这条脚本的训练判断

这条线的实验哲学很明确：

- 先保证 rewrite 输出稳定、短、小心、可控。
- 先争取 `RL > Zero-shot`，再去逼近或超过 `Original`。
- 不急着追求很激进的 query 变形。

## 12.5 一个容易忽略的脚本细节

`run_train_then_full_eval_4b.sh` 和 `.ps1` 默认都带：

- `AUTO_GIT_COMMIT=1`
- `AUTO_GIT_PUSH=1`

也就是说，脚本默认会把训练和评估产物自动 `git add -A`、commit，甚至 push。

这不是训练算法的一部分，但确实是当前实验工作流的一部分，后续正式跑实验前最好有意识地确认是否保留这个行为。

## 13. 新实验脚本：`top20_delta + curriculum`

## 13.1 这条线的目标

这条线不是单纯“让 reward 更高”，而是想解决一个核心问题：

- 原 query 本身经常已经不差；
- 如果还用绝对 reward，模型很容易学到保守复制；
- 更合理的目标是“只奖励相对原 query 的真实提升”。

所以这条线把目标收窄成：

- 优先拉 `top-20` 排名
- 只奖励相对原 query 的改进
- 通过 curriculum 让模型先学“还有救的 query”，再学更难的 query

## 13.2 共同参数

`run_train_then_full_eval_4b_top20_delta_curriculum.sh` 两个 phase 共享：

| 项目 | 默认值 |
|---|---:|
| `ARTIFACT_ROOT` | `train_and_eval_data_model_0420` |
| `EXP_NAME` | `4b_top20_delta_curriculum` |
| `MODEL_NAME` | `/root/autodl-tmp/hf_models/Qwen3-4B-Instruct-2507` |
| `SEARCH_THREADS` | `16` |
| `EVAL_QUERY_BATCH_SIZE` | `8` |
| `ACTOR_CHUNK_SIZE` | `2` |
| `PROJECTION_CHUNK_SIZE` | `64` |
| `GROUP_TEMPERATURE_STRIDE` | `0.07` |
| `GROUP_TOP_P_STRIDE` | `0.015` |
| `MIN_UNIQUE_FINAL_QUERIES` | `4` |
| `MAX_REGEN_ROUNDS` | `2` |
| `REWARD_GAP_THRESHOLD` | `0.08` |
| `GAP_SAMPLING_TEMPERATURE_DELTA` | `0.15` |
| `TRAIN_MAX_VAL_QUERIES` | `400` |
| `reward_mode` | `top20_delta` |
| `reward_mrr_k` | `20` |
| `reward_recall_k` | `20` |
| `reward_recall_dense_k` | `50` |
| `reward_w_bad_format` | `0.18` |
| `reward_w_unsafe_copy` | `0.12` |
| `reward_w_overedit` | `0.10` |
| `overedit_tau` | `0.40` |
| `format_max_tokens` | `12` |
| `format_min_english_ratio` | `0.8` |
| `format_max_unreadable_ratio` | `0.25` |

## 13.3 phase1 参数

| 项目 | 默认值 |
|---|---:|
| `batch_size` | `24` |
| `group_size` | `8` |
| `max_group_size` | `12` |
| `learning_rate` | `1.0e-5` |
| `kl_beta` | `0.045` |
| `temperature` | `0.75` |
| `top_p` | `0.92` |
| `max_new_tokens` | `10` |
| `eval_every_steps` | `20` |
| `max_steps` | `80` |
| `reward_w_mrr` | `0.55` |
| `reward_w_recall` | `0.20` |
| `reward_w_recall_dense` | `0.15` |
| `reward_w_rank_bonus` | `0.10` |

phase1 的倾向是：

- 更看重 `delta_mrr`
- 仍保留 recall 和 rank bonus
- 解码稍微保守
- 学习率略大一些，先把“能学到的相对提升”快速学起来

## 13.4 phase2 参数

| 项目 | 默认值 |
|---|---:|
| `adapter_path` | `phase1/checkpoints/best` |
| `batch_size` | `24` |
| `group_size` | `8` |
| `max_group_size` | `12` |
| `learning_rate` | `8e-6` |
| `kl_beta` | `0.05` |
| `temperature` | `0.70` |
| `top_p` | `0.90` |
| `max_new_tokens` | `10` |
| `eval_every_steps` | `20` |
| `max_steps` | `60` |
| `reward_w_mrr` | `0.65` |
| `reward_w_recall` | `0.15` |
| `reward_w_recall_dense` | `0.10` |
| `reward_w_rank_bonus` | `0.10` |

phase2 的倾向是：

- 更强地把优化重点推到 `delta_mrr`
- 学习率更低
- 继续 warm-start 微调

一句话说，phase2 比 phase1 更像“收口调优”。

## 13.5 curriculum 分桶规则

当前把训练 query 分成四类：

- `A`
  - `orig_recall100 > 0`
  - `orig_mrr20 < 0.35`
  - 含义：能搜到相关文档，但排位还不够好，最适合优化 top rank
- `B`
  - `orig_recall100 == 0`
  - 但 query 看起来已经比较像可用检索 query
  - 含义：本身格式还行，但没搜到相关文档，是 harder negative
- `C`
  - `orig_mrr20 >= 0.35`
  - 含义：原 query 已经挺强
- `DROP`
  - 不满足以上条件
  - 直接不进 curriculum 训练

## 13.6 phase 混合比例

| phase | A | B | C |
|---|---:|---:|---:|
| `phase1` | `0.80` | `0.05` | `0.15` |
| `phase2` | `0.60` | `0.20` | `0.20` |

这背后的思路是：

- `phase1` 先主攻最有提升空间的 A 类；
- `phase2` 再逐步加入更多 B/C 类，让模型适应更复杂、更强的原 query。

## 14. 低显存 smoke 模式

当前 `--low-mem-mode` 不是主实验，但它很有用，因为它定义了“最小可跑通配置”。

覆盖项如下：

| 项目 | 值 |
|---|---:|
| `model_name` | `Qwen/Qwen3.5-0.8B` |
| `load_in_4bit` | `true` |
| `lora_r` | `8` |
| `lora_alpha` | `16` |
| `batch_size` | `1` |
| `group_size` | `8` |
| `max_group_size` | `24` |
| `max_new_tokens` | `20` |
| `temperature` | `0.1` |
| `top_p` | `0.95` |
| `reward_gap_threshold` | `0.10` |
| `actor_chunk_size` | `1` |
| `projection_chunk_size` | `32` |
| `eval_every_steps` | `10` |
| `max_steps` | `50` |
| `max_train_queries` | `64` |
| `max_val_queries` | `32` |
| `prebuilt_index` | `msmarco-v1-passage-slim` |

这套配置的定位不是追结果，而是：

- 先确认链路和代码没坏；
- 在小显存上尽快 smoke test；
- 快速验证 reward、日志和 checkpoint 是否正常。

## 15. 仓库内已有实验结果

## 15.1 已有三路评测结果

目前仓库里能直接看到的 eval report 有三组：

| 产物 | 模型 | 说明 | Original | Zero-shot | RL | RL-Orig | RL-Zero |
|---|---|---|---:|---:|---:|---:|---:|
| `artifacts_0p5b_eval` | `Qwen/Qwen2.5-0.5B-Instruct` | 历史 legacy 结果 | `0.2412` | `0.1707` | `0.2090` | `-0.0321` | `+0.0383` |
| `artifacts_0p8b_eval` | `Qwen/Qwen3.5-0.8B` | 历史 legacy 结果 | `0.2412` | `0.2220` | `0.2387` | `-0.0025` | `+0.0167` |
| `artifacts_default_eval` | `Qwen/Qwen3.5-0.8B` | 更早一版 reward schema | `0.2412` | `0.0854` | `0.1114` | `-0.1298` | `+0.0260` |

当前最值得记住的结论：

- RL 相比 zero-shot 是稳定正增益。
- 但在仓库已提交结果里，大多数设置仍未超过 original 基线。
- 离 original 最近的是 `0.8B legacy`，只差 `-0.0025`。

## 15.2 对这些历史结果怎么解读

这些结果能说明趋势，但不能简单横向硬比：

- 仓库里不同时间点的 reward schema 发生过演进；
- 有些旧 report 还是更早的字段结构；
- 当前代码已经比旧产物多了：
  - reward gap 自适应补采样
  - regen
  - `top20_delta`
  - curriculum
  - 更完整的 group trace 指标

所以最合理的用法是：

- 把这些报告当作“旧实验里 RL 普遍优于 zero-shot，但还没稳稳超过 original”的证据；
- 不要把它们当作当前代码最终结论。

## 15.3 最近一份训练日志摘要说明了什么

`analysis_outputs/download_logs_20260420/summary.json` 给出的最近一轮日志摘要里，有几个信号很重要：

- `reward_mean_avg = 0.5038`
- `mrr_mean_avg = 0.1707`
- `term_preserve_mean = 0.9364`
- `unique_final_query_mean = 6.2248`
- `collapsed_group_ratio = 0.0339`
- `reward_gap_met_ratio = 0.4535`
- `generated_sample_count_mean = 14.2616`
- `extra_sample_ratio = 0.1575`
- `valid_ratio` 从前段的 `0.9030` 提升到后段的 `1.0`

我对这组数字的解读是：

- 组内多样性控制确实在工作，平均唯一 final query 超过 6。
- 塌缩组比例已经不高。
- reward gap 自适应补采样经常在工作，说明固定 group 还不够，需要额外探索。
- 后期 valid ratio 到 1.0，说明训练稳定性在变好。

但同时也能看到：

- 后段 `reward_mean` 没明显抬升；
- 说明当前训练更像是在把过程“训稳”，而不是已经找到一个强提升方向。

## 15.4 `top20_delta` 当前仓库状态

仓库里目前看到的是：

- `artifacts_top20_delta_smoke/phase1`
- `artifacts_top20_delta_smoke/phase2`
- `curriculum_query_metadata.jsonl`

它更像 smoke 验证而不是正式结果：

- curriculum metadata 只有 `8` 条
- phase1 / phase2 都只看到非常小规模 step 记录
- eval 样本也很小

所以现阶段只能说明：

- `top20_delta + curriculum` 整条链路能跑通；
- 还不能据此判断它是否优于 conservative 主线。

## 15.5 当前缺的关键结果

当前仓库里最缺的是两类正式结果：

- 4B `conservative_mrr` 的完整训练和 full eval 报告
- 4B `top20_delta + curriculum` 的完整训练和 full eval 报告

也就是说：

- 现在脚本已经准备好了；
- 但仓库里还没有把“4B 正式结论”落成最终产物。

## 16. 对当前训练策略的整体判断

## 16.1 已经证明有效的部分

目前我认为已经被代码和产物证明“有价值”的设计有：

- 用真实检索指标直接做 reward
- 保留词约束和格式惩罚
- group 内相对 advantage，而不是单样本硬打分
- reward gap 驱动的补采样
- group regen 防塌缩
- QLoRA + ref KL 约束
- 训练日志和 group trace 双日志体系

这些设计至少说明：

- 训练不是盲目跑；
- 现在这套系统已经能比较细地观察“模型为什么没提升”。

## 16.2 现在最大的核心矛盾

当前最大的矛盾不是“训不动”，而是：

- RL 能稳步优于 zero-shot；
- 但还不够稳地超过 original。

这说明系统正在学到“比 base model prompt rewrite 更像检索 query”的东西，
但还没稳定学到“比用户原 query 更强”的东西。

## 16.3 为什么会出现这个矛盾

从当前代码和日志看，最可能的原因有几类：

- 原 query 本来就不弱，提升空间有限。
- 如果奖励太保守，模型容易学成小修小补。
- 如果奖励太激进，又容易过改、丢词或触发 unsafe copy / overedit 风险。
- query 改写任务的有效动作空间很小，很多“看起来不同”的改写在检索上没本质差别。

这也是为什么项目后来会往 `top20_delta + curriculum` 方向走：

- 不是因为 legacy 思路错了；
- 而是因为 legacy 更容易学成“保守、稳定、接近原 query”的策略。

## 17. 我建议后续继续这样组织实验

如果后面要继续推进，我建议把实验明确分成三层：

### 第一层：稳定主线

- 先把 `4b_conservative_mrr` 跑完整。
- 用它验证 4B 是否能在 legacy 绝对奖励下稳定逼近或超过 original。

### 第二层：相对提升主线

- 再跑 `4b_top20_delta_curriculum`。
- 重点看：
  - `delta_mrr20_mean`
  - `delta_mrr20_positive_ratio`
  - `best_reward_hit_best_mrr20_ratio`
  - `unsafe_copy_penalty_mean`
  - `overedit_penalty_mean`

### 第三层：诊断性对比

- 固定同一批 val query，对比：
  - Original
  - Zero-shot
  - 4B conservative RL
  - 4B top20_delta RL
- 不只看均值，还看：
  - 有多少 query 真正赢了 original
  - 是哪些 query 赢了
  - 是靠提升 recall 还是把 hit rank 顶得更前

## 18. 最后的结论

如果只用一句话总结当前项目状态，我会这样写：

- 这套系统已经从“一个可跑的 RL demo”进化成“一个有明确检索目标、有 reward 约束、有多样性控制、有课程学习分支、并且能细粒度诊断训练行为的查询改写训练框架”；当前最现实的下一步，不是再改一堆机制，而是把 `4B conservative_mrr` 和 `4B top20_delta curriculum` 两条线跑成完整可比的正式结果。

