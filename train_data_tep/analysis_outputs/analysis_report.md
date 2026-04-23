# 两阶段 DeepRetrieval-GRPO 实验分析

## 数据概况

- 阶段 1：105 个训练 step，2400 条 group trace。
- 阶段 2：63 个训练 step，1440 条 group trace。
- 最终评估：1396 个 query，对比 Original、Zero-shot rewrite、RL rewrite。

## 训练过程结论

- reward_mean：阶段 1 平均 -0.0688，末 10 step 0.0138；阶段 2 平均 0.0025，末 10 step 0.0117。
- main_reward_mean：阶段 1 平均 0.0249，阶段 2 平均 0.0374；阶段 2 主奖励整体更高。
- delta_mrr20_mean：阶段 1 平均 0.0057，阶段 2 平均 0.0056；阶段 2 的 MRR 增益更稳定地保持在正区间。
- recall_drop_ratio：阶段 1 平均 6.4%，阶段 2 平均 3.0%；阶段 2 的召回下降风险明显降低。
- keyword_preserve_mean：阶段 1 平均 85.0%，阶段 2 平均 90.9%；阶段 2 关键词保留更好。
- trainable_group_ratio：阶段 1 平均 70.3%，阶段 2 平均 52.0%；阶段 2 中可训练组比例下降，说明更多组 reward 变平或差异不足。

## Group trace 结论

- Stage 1：reward gap 达标率 60.1%，平均 reward gap 0.5119，平均生成样本数 9.7975，平均额外采样 1.7975。
- Stage 1：threshold_reached 60.1%，max_group_size_reached 39.9%，平均唯一 final query 数 5.8192。
- Stage 2：reward gap 达标率 41.0%，平均 reward gap 0.2682，平均生成样本数 10.5708，平均额外采样 2.5708。
- Stage 2：threshold_reached 41.0%，max_group_size_reached 59.0%，平均唯一 final query 数 6.4465。

## 最终评估结论

- MRR@20：Original 0.2022，Zero-shot 0.2068，RL 0.2105。RL 相比 Original 提升 0.0083，相比 Zero-shot 提升 0.0037。
- Recall@20：Original 0.4865，RL 0.4969，RL 的平均 delta_recall20 为 0.0104。
- Reward：Original -0.0688，Zero-shot 0.0004，RL 0.0308；RL reward 相比 Original 提升 0.0997。
- 安全/格式惩罚：RL 的 recall_drop_penalty 0.0155，overedit_penalty 0.0008，bad_format_penalty 0.0000，unsafe_copy_penalty 0.0007。
- 按 query 统计：RL MRR 优于 Zero-shot 的比例 9.6%，低于 Zero-shot 的比例 7.7%，优于 Original 的比例 14.3%；RL Recall@20 有增益的比例 3.2%。

## 输出图表

- `combined_training_metrics.png`：两个阶段拼接的主要训练指标，浅蓝为阶段 1，浅橙为阶段 2。
- `combined_training_diagnostics.png`：关键词保留、召回下降、可训练组、KL 等诊断指标。
- `combined_group_trace_dynamics.png`：从 group trace 聚合出的采样与组内质量动态。
- `stage_group_distributions.png`：两个阶段的 group 级分布箱线图。
- `eval_comparison_summary.png`：最终评估的 Original / Zero-shot / RL 对比。