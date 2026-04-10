# DeepRetrieval-GRPO 项目伪代码分析

## 1. 项目概览

**项目名称**: DeepRetrieval-GRPO  
**主要功能**: 使用自定义GRPO（Group Relative Policy Optimization）算法，通过强化学习优化查询重写，提升信息检索质量

**核心思想**:
- 用LLM重写原始查询以改进检索效果
- 通过MRR（Mean Reciprocal Rank）作为奖励信号
- 使用PPO算法与KL正则项约束策略更新
- 采用组内相对优势标准化，避免显式价值函数

---

## 2. 项目总体架构伪代码

```
项目架构 DeepRetrieval-GRPO:
    ├─ 模块: 配置管理 (app_config.py)
    │   ├─ 数据配置 DataConfig
    │   │   ├─ 主题集名称 (topic_name)
    │   │   ├─ 预编译索引名 (prebuilt_index)
    │   │   ├─ 训练/验证分割比例
    │   │   └─ 样本数上限
    │   │
    │   ├─ 模型配置 ModelConfig
    │   │   ├─ 基础模型 (Qwen2.5-3B/0.5B)
    │   │   ├─ 量化配置 (4-bit QLoRA)
    │   │   ├─ LoRA参数 (r=16, alpha=32)
    │   │   └─ 目标层 (q_proj, k_proj, v_proj等)
    │   │
    │   ├─ 训练配置 TrainConfig
    │   │   ├─ 优化器参数 (lr=2e-5, batch_size=2)
    │   │   ├─ GRPO参数 (group_size=4, clip_range=0.2)
    │   │   ├─ KL正则系数 (kl_beta=0.02)
    │   │   └─ 生成超参 (max_tokens=24, temperature=1.0)
    │   │
    │   └─ 奖励配置 RewardConfig
    │       ├─ 检索截断 (topk=10)
    │       ├─ 奖励权重 (mrr_weight=1.0, overlap_weight=0.2)
    │       └─ 文本惩罚阈值
    │
    ├─ 模块: 数据加载 (data/loader.py)
    │   ├─ 从Pyserini加载topics & qrels
    │   ├─ 按比例分割训练/验证集
    │   ├─ 查询样本管理 QueryExample
    │   └─ Java环境验证 (Pyserini依赖)
    │
    ├─ 模块: 模型包装 (core/model_wrapper.py)
    │   ├─ Actor模型 (可训练的LoRA)
    │   │   ├─ 前向推理
    │   │   ├─ 采样生成 (generate_with_logprob)
    │   │   └─ logprob重算 (compute_logprob)
    │   │
    │   ├─ Ref模型 (冻结参考策略)
    │   │   └─ KL惩罚项计算 (logprob_ref)
    │   │
    │   └─ 量化与LoRA配置
    │       ├─ 4-bit量化参数
    │       └─ LoRA适配器管理
    │
    ├─ 模块: 奖励函数 (core/reward_func.py)
    │   ├─ 检索奖励
    │   │   ├─ MRR@k 通过Pyserini BM25
    │   │   └─ 词面重叠分数 (Jaccard)
    │   │
    │   └─ 文本惩罚
    │       ├─ 过短惩罚
    │       ├─ 重复惩罚 (token重复率)
    │       └─ 不可读字符惩罚
    │
    ├─ 模块: GRPO训练引擎 (core/grpo_engine.py)
    │   ├─ 采样阶段
    │   │   ├─ 每个query采样K个重写
    │   │   └─ 记录旧策略logprob
    │   │
    │   ├─ 奖励计算
    │   │   └─ MRR - 文本惩罚
    │   │
    │   ├─ 优势归一化
    │   │   ├─ 组内标准化
    │   │   └─ 相对优势 (reward - mean) / std
    │   │
    │   ├─ 损失函数
    │   │   ├─ PPO clipped objective: min(ratio*adv, clip(ratio)*adv)
    │   │   ├─ KL正则: kl_beta * (logprob_new - logprob_ref)
    │   │   └─ 总loss = loss_pg + loss_kl
    │   │
    │   └─ 参数更新
    │       ├─ 反向传播
    │       ├─ 梯度裁剪 (clip_norm=1.0)
    │       └─ AdamW优化器更新
    │
    ├─ 脚本: 训练主流程 (train.py)
    │   ├─ CLI参数解析
    │   ├─ 配置加载与覆盖
    │   ├─ 随机种子设置
    │   ├─ 低内存模式支持
    │   ├─ 数据源初始化
    │   ├─ 模型与优化器初始化
    │   ├─ 训练循环 执行GRPO
    │   ├─ 周期性评估
    │   ├─ Checkpoint保存 (best/latest)
    │   └─ 训练日志记录 (JSONL格式)
    │
    └─ 脚本: 评估脚本 (eval_compare.py)
        ├─ Original vs Zero-shot vs RL三路对比
        ├─ 模型推理
        ├─ 性能指标计算
        └─ 评估报告生成 (JSON格式)
```

---

## 3. 训练流程详细伪代码

```pseudocode
函数 main_train_pipeline():
    """GRPO训练主循环"""
    
    // 第一步: 初始化
    步骤1_初始化():
        config = 加载默认配置()
        
        if 命令行参数.low_mem_mode:
            config = 应用低内存预设(config)
            // 使用0.5B模型, batch_size=1, group_size=2等
        
        config = 应用CLI参数覆盖(config)
        
        检查CUDA可用性()
        验证group_size >= 2  // GRPO需要至少2个样本
        
        创建输出目录(config.save_dir, config.log_path)
        设置随机种子(config.seed)
    
    // 第二步: 加载数据与检索资源
    步骤2_数据初始化():
        验证Java运行时 >= X版本  // Pyserini需求
        
        queries, qrels = 从Pyserini加载(config.topic_name)
        // 示例: "msmarco-passage-dev-subset" -> ~100K queries
        
        train_queries, val_queries = 按比例分割(
            queries, 
            train_ratio=config.train_ratio, 
            seed=config.seed
        )
        
        train_queries = 可选限制(train_queries, config.max_train_queries)
        val_queries = 可选限制(val_queries, config.max_val_queries)
        
        打印统计 {
            "train_queries": len(train_queries),
            "val_queries": len(val_queries),
            "qrels_qids": len(qrels)
        }
    
    // 第三步: 初始化奖励器 (Pyserini检索)
    步骤3_奖励器初始化():
        rewarder = 创建Rewarder(
            qrels=qrels,  // 相关性标签
            prebuilt_index=config.prebuilt_index,  // "msmarco-v1-passage"
            reward_cfg=config.reward
        )
        
        // 内部功能:
        // - Pyserini BM25检索器初始化
        // - MRR@k计算准备
        // - 文本惩罚规则加载
    
    // 第四步: 基线评估 (原始query)
    步骤4_基线评估():
        baseline_eval = 评估原始查询(
            rewarder,
            val_queries,
            max_queries=config.max_val_queries
        )
        
        打印 "Original Baseline MRR@10: {baseline_eval.mrr_mean:.4f}"
        best_val_mrr = -∞
    
    // 第五步: 初始化模型与优化器
    步骤5_模型初始化():
        model = 创建ModelWrapper(
            model_cfg=config.model,
            prompt_cfg=config.prompt,
            train_mode=True,  // Actor进入训练态
            enable_lora=True,  // 构建LoRA适配器
            load_ref_model=True,  // 加载冻结参考模型
            adapter_path=config.adapter_path  // 可选暖启动
        )
        
        // 模型初始化细节:
        // 1. Actor模型加载 (Qwen2.5-3B)
        //    - 4-bit量化: BitsAndBytesConfig
        //    - LoRA配置: r=16, alpha=32, dropout=0.05
        //    - 目标层: q/k/v/o/gate/up/down_proj
        // 2. Ref模型加载 (冻结, 无梯度)
        // 3. Tokenizer初始化
        //    - pad_token设置为eos_token
        
        optimizer = 创建AdamW(
            params=model.可训练参数(),
            lr=config.train.learning_rate,  // 2e-5
            weight_decay=config.train.weight_decay  // 0.0
        )
        
        engine = 创建GRPOEngine(
            model_wrapper=model,
            rewarder=rewarder,
            optimizer=optimizer,
            group_size=config.train.group_size,  // K=4
            clip_range=config.train.clip_range,  // 0.2
            kl_beta=config.train.kl_beta,  // 0.02
            grad_clip_norm=config.train.grad_clip_norm,  // 1.0
            max_new_tokens=config.train.max_new_tokens,  // 24
            temperature=config.train.temperature,  // 1.0
            top_p=config.train.top_p  // 0.95
        )
    
    // 第六步: 主训练循环
    步骤6_训练循环():
        global_step = 0
        best_val_mrr = -∞
        should_stop = False
        
        对于 epoch 从 1 到 config.train.num_epochs:
            // 每个epoch重新乱序training queries
            乱序(train_queries, seed=config.seed + epoch)
            
            对于 batch 从 按batch_size分割(train_queries):
                global_step += 1
                
                // GRPO训练步骤 (见第7步详细流程)
                metrics = engine.train_step(batch)
                
                metrics.update({
                    "phase": "train",
                    "epoch": epoch,
                    "step": global_step,
                    "timestamp": 当前时间UTC()
                })
                
                追加日志到JSONL(config.log_path, metrics)
                
                打印训练进度 {
                    "step": global_step,
                    "loss": metrics.loss,
                    "loss_pg": metrics.loss_pg,
                    "loss_kl": metrics.loss_kl,
                    "reward_mean": metrics.reward_mean,
                    "mrr_mean": metrics.mrr_mean
                }
                
                // 周期性评估 (eval_every_steps)
                如果 global_step % config.train.eval_every_steps == 0:
                    步骤7_周期评估()
                
                如果 config.train.max_steps 不为空:
                    如果 global_step >= config.train.max_steps:
                        should_stop = True
                        break
            
            如果 should_stop:
                break
    
    // 第七步: 周期评估与Checkpoint保存
    步骤7_周期评估():
        eval_metrics = 评估策略(
            model,
            rewarder,
            val_queries,
            max_queries=config.max_val_queries,
            max_new_tokens=config.train.max_new_tokens
        )
        
        eval_metrics.update({
            "phase": "eval",
            "epoch": epoch,
            "step": global_step,
            "timestamp": 当前时间UTC()
        })
        
        追加日志到JSONL(config.log_path, eval_metrics)
        
        打印评估结果 {
            "step": global_step,
            "val_mrr": eval_metrics.mrr_mean,
            "val_reward": eval_metrics.reward_mean
        }
        
        // Checkpoint保存策略
        model.保存适配器(latest_path)  // 总是保存latest
        
        如果 eval_metrics.mrr_mean > best_val_mrr:
            best_val_mrr = eval_metrics.mrr_mean
            model.保存适配器(best_path)  // 仅在改进时更新best
            打印 "检查点已更新: best_mrr = {best_val_mrr:.4f}"
    
    // 第八步: 训练结束后处理
    步骤8_训练结束():
        model.保存适配器(latest_path)
        
        final_eval = 评估策略(
            model,
            rewarder,
            val_queries,
            max_queries=config.max_val_queries,
            max_new_tokens=config.train.max_new_tokens
        )
        
        打印最终结果 {
            "final_val_mrr": final_eval.mrr_mean,
            "best_val_mrr": best_val_mrr,
            "improvement": final_eval.mrr_mean - baseline_eval.mrr_mean
        }
        
        return 0  // 成功退出

```

---

## 4. GRPO训练步骤详细伪代码

```pseudocode
函数 GRPOEngine.train_step(batch_queries):
    """单个batch的GRPO更新步骤"""
    
    // 初始化
    model.actor.train(True)  // 设置训练态
    optimizer.zero_grad(set_to_none=True)
    
    loss_terms = []
    loss_pg_terms = []
    loss_kl_terms = []
    rewards = []
    mrr_scores = []
    penalties = []
    overlaps = []
    all_advantages = []
    valid_samples = 0
    sampled = 0
    
    // 循环处理batch中的每个查询
    对于 query 在 batch_queries:
        prompt = model.构建提示词(query.text)
        // 示例: "You are an expert search query rewriter...\nUser Query: {query}\nSearch Query:"
        
        group_samples = []
        
        // ============ 第一阶段: 采样 ============
        对于 i 从 1 到 group_size (K=4):
            // 1.1 从当前策略采样K个重写
            generated = model.生成并记录对数概率(
                prompt,
                max_new_tokens=这个.max_new_tokens,
                temperature=这个.temperature,
                top_p=这个.top_p
            )
            // 返回值:
            // - response_text: 生成的重写查询
            // - response_token_ids: token序列
            // - logprob_old: 采样时的对数概率 (用于PPO ratio的分母)
            
            // 1.2 计算序列级奖励 (检索质量 + 文本质量)
            reward_breakdown = rewarder.计算奖励(
                query_id=query.qid,
                rewritten_query=generated.response_text,
                source_query=query.text
            )
            // 返回值包含:
            // - total: MRR - 文本惩罚
            // - mrr: MRR@k值
            // - overlap: 词面重叠分数
            // - penalty: 文本惩罚项
            
            sample = Sample(
                qid=query.qid,
                prompt=prompt,
                response_text=generated.response_text,
                response_token_ids=generated.response_token_ids,
                logprob_old=generated.logprob_old,
                reward=reward_breakdown.total,
                mrr=reward_breakdown.mrr,
                overlap=reward_breakdown.overlap,
                penalty=reward_breakdown.penalty
            )
            group_samples.append(sample)
            sampled += 1
        
        // ============ 第二阶段: 优势计算 ============
        // 2.1 组内奖励标准化 (GRPO核心特性)
        advantages = normalize_advantages(
            [s.reward for s in group_samples],
            eps=1e-8
        )
        
        // normalize_advantages实现:
        // - 计算mean = mean(rewards)
        // - 计算std = std(rewards)
        // - 返回 (rewards - mean) / (std + eps)
        // - 数值稳定: 若std < eps, 返回全0
        
        对于 (sample, advantage) 在 zip(group_samples, advantages):
            sample.advantage = float(advantage)
            all_advantages.append(sample.advantage)
            rewards.append(sample.reward)
            mrr_scores.append(sample.mrr)
            penalties.append(sample.penalty)
            overlaps.append(sample.overlap)
        
        // ============ 第三阶段: 损失计算 ============
        对于 sample 在 group_samples:
            // 3.1 处理空序列
            如果 sample.response_token_ids为空:
                continue  // 跳过无法计算loss的样本
            
            // 3.2 计算Actor新策略logprob (带梯度)
            logprob_new = model.计算对数概率(
                prompt=sample.prompt,
                token_ids=sample.response_token_ids,
                policy="actor",
                no_grad=False  // 保留梯度
            )
            
            // 3.3 计算Ref冻结策略logprob (无梯度)
            logprob_ref = model.计算对数概率(
                prompt=sample.prompt,
                token_ids=sample.response_token_ids,
                policy="ref",
                no_grad=True  // 不计算梯度
            )
            
            // 3.4 处理张量对齐
            // 注: 采样与重算可能token数不一致, 取最小长度
            t = min(
                logprob_new.numel(),
                sample.logprob_old.numel(),
                logprob_ref.numel()
            )
            
            如果 t == 0:
                continue
            
            logprob_new = logprob_new[:t]
            logprob_old = sample.logprob_old[:t].to(logprob_new.device)
            logprob_ref = logprob_ref[:t].to(logprob_new.device)
            
            // 3.5 计算PPO目标函数 (token级别)
            // PPO clipped objective = min(ratio*adv, clip(ratio)*adv)
            clipped_obj = compute_ppo_clipped_objective(
                logprob_new=logprob_new,
                logprob_old=logprob_old,
                advantage=sample.advantage,
                clip_range=这个.clip_range  // 0.2
            )
            
            // compute_ppo_clipped_objective实现:
            // ratio = exp(logprob_new - logprob_old)
            // unclipped = ratio * advantage
            // clipped = clip(ratio, 1-clip_range, 1+clip_range) * advantage
            // return min(unclipped, clipped)
            
            // 3.6 计算策略梯度损失
            loss_pg = -clipped_obj.mean()  // 取反因为要最大化objective
            
            // 3.7 计算KL正则项
            // 约束新策略不要偏离参考策略过远
            loss_kl = 这个.kl_beta * (logprob_new - logprob_ref).mean()
            
            // 3.8 组合总损失
            loss = loss_pg + loss_kl
            
            // 3.9 数值稳定性检查
            如果 不是有限的(loss):
                continue  // 跳过NaN/Inf样本
            
            loss_terms.append(loss)
            loss_pg_terms.append(float(loss_pg.detach().cpu()))
            loss_kl_terms.append(float(loss_kl.detach().cpu()))
            valid_samples += 1
        
        // ============ 第四阶段: 参数更新 ============
        如果 loss_terms非空:
            // 4.1 汇聚所有样本的损失
            总损失 = mean(loss_terms)  // batch内所有样本的平均loss
            
            // 4.2 反向传播
            总损失.backward()
            
            // 4.3 梯度裁剪 (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(
                model.actor.parameters(),
                max_norm=这个.grad_clip_norm  // 1.0
            )
            
            // 4.4 优化器更新
            optimizer.step()
        
        // 否则 (无有效样本):
        //   返回零损失指标, 不更新参数
    
    // ============ 第五阶段: 指标汇总 ============
    nonzero_reward_ratio = sum(1 for r in rewards if r > 0) / len(rewards)
    
    返回 {
        "loss": mean(loss_terms) if loss_terms else 0.0,
        "loss_pg": mean(loss_pg_terms) if loss_pg_terms else 0.0,
        "loss_kl": mean(loss_kl_terms) if loss_kl_terms else 0.0,
        "reward_mean": mean(rewards) if rewards else 0.0,
        "mrr_mean": mean(mrr_scores) if mrr_scores else 0.0,
        "penalty_mean": mean(penalties) if penalties else 0.0,
        "overlap_mean": mean(overlaps) if overlaps else 0.0,
        "nonzero_reward_ratio": nonzero_reward_ratio,
        "adv_mean": mean(all_advantages) if all_advantages else 0.0,
        "adv_std": std(all_advantages) if all_advantages else 0.0,
        "sampled": float(sampled),
        "valid_samples": float(valid_samples)
    }

```

---

## 5. 奖励函数详细伪代码

```pseudocode
函数 Rewarder.计算奖励(query_id, rewritten_query, source_query):
    """计算序列级奖励 = 检索奖励 - 文本惩罚"""
    
    // ============ 第一部分: 检索奖励 ============
    
    // 1.1 解析生成的查询
    candidate_queries = 提取候选查询(rewritten_query)
    // 策略: 查找"Rewritten Query:"标记, 或直接取输出
    
    最终查询 = candidate_queries[0] 如果 candidate_queries非空 否则 ""
    
    // 1.2 通过BM25检索
    如果 最终查询为空:
        mrr = 0.0
        hit_rank = None
    否则:
        docs = 执行BM25检索(
            query=最终查询,
            index=这个.检索器,
            top_k=这个.reward_cfg.topk  // 10
        )
        // 返回值: 相关性排序的文档ID列表
        
        // 1.3 计算MRR@k
        relevant_docids = 这个.qrels.get(query_id, set())
        mrr, hit_rank = 计算MRR(
            result_docids=docs,
            relevant_docids=relevant_docids,
            topk=这个.reward_cfg.topk
        )
        
        // 计算MRR实现:
        // for rank, docid in enumerate(docs[:topk], start=1):
        //     if docid in relevant_docids:
        //         return (1/rank, rank)
        // return (0.0, None)
    
    // 1.4 计算词面重叠 (保真约束)
    overlap_score = 计算词面重叠(
        source_query=source_query,
        rewritten_query=最终查询
    )
    // 实现: Jaccard = |A∩B| / |A∪B|
    
    // ============ 第二部分: 文本惩罚 ============
    
    penalty_details = 计算文本惩罚(最终查询, reward_cfg)
    
    // 2.1 过短惩罚
    如果 len(最终查询.strip()) < reward_cfg.min_query_chars:
        short_penalty = reward_cfg.penalty_short  // 0.2
    否则:
        short_penalty = 0.0
    
    // 2.2 重复惩罚
    tokens = 分词(最终查询)
    unique_ratio = len(set(tokens)) / len(tokens)
    如果 (1 - unique_ratio) > reward_cfg.max_repeat_ratio:
        repeat_penalty = reward_cfg.penalty_repeat  // 0.2
    否则:
        repeat_penalty = 0.0
    
    // 2.3 不可读字符惩罚
    unreadable_ratio = 计算不可读字符比例(最终查询)
    如果 unreadable_ratio > reward_cfg.max_unreadable_char_ratio:
        unreadable_penalty = reward_cfg.penalty_unreadable  // 0.3
    否则:
        unreadable_penalty = 0.0
    
    总惩罚 = short_penalty + repeat_penalty + unreadable_penalty
    
    // ============ 第三部分: 组合奖励 ============
    
    mrr_reward = 这个.reward_cfg.mrr_weight * mrr
    overlap_reward = 这个.reward_cfg.overlap_weight * overlap_score
    
    总奖励 = mrr_reward + overlap_reward - 总惩罚
    
    返回 RewardBreakdown(
        total=总奖励,
        mrr=mrr,
        overlap=overlap_score,
        penalty=总惩罚,
        hit_rank=hit_rank,
        short_penalty=short_penalty,
        repeat_penalty=repeat_penalty,
        unreadable_penalty=unreadable_penalty,
        rewritten_query=最终查询
    )

```

---

## 6. 模型推理关键步骤伪代码

```pseudocode
函数 ModelWrapper.生成并记录对数概率(prompt, max_new_tokens, temperature, top_p):
    """采样生成并记录旧策略对数概率"""
    
    // 1. 编码输入
    input_ids = tokenizer.encode(prompt)
    input_ids = input_ids.to(device)
    
    // 2. 生成配置
    generate_config = {
        max_new_tokens: max_new_tokens,
        do_sample: True,
        temperature: temperature,
        top_p: top_p,
        pad_token_id: tokenizer.pad_token_id,
        eos_token_id: tokenizer.eos_token_id,
        output_scores: True,  // 返回logits用于logprob计算
        return_dict_in_generate: True
    }
    
    // 3. 生成
    with torch.no_grad():  // 采样阶段不需要计算梯度
        outputs = actor_model.generate(
            input_ids,
            **generate_config
        )
    
    // 4. 提取response tokens (去掉input_ids部分)
    response_token_ids = outputs.sequences[0][len(input_ids):]
    response_text = tokenizer.decode(response_token_ids, skip_special_tokens=True)
    
    // 5. 从scores计算logprob (采样时的对数概率)
    logprobs = []
    对于 t, token_id 在 enumerate(response_token_ids):
        logits = outputs.scores[t][0]  // batch_size=1
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        logprob = log_probs[token_id].item()
        logprobs.append(logprob)
    
    logprob_old = torch.tensor(logprobs, dtype=torch.float32)
    
    返回 GeneratedSample(
        response_text=response_text,
        response_token_ids=response_token_ids.tolist(),
        logprob_old=logprob_old
    )


函数 ModelWrapper.计算对数概率(prompt, token_ids, policy, no_grad):
    """重新计算给定token序列的对数概率"""
    
    // 1. 选择要使用的模型
    如果 policy == "actor":
        model = 这个.actor_model
        ctx = torch.no_grad() if no_grad else nullcontext()
    否则 (policy == "ref"):
        model = 这个.ref_model
        ctx = torch.no_grad()  // ref总是冻结
    
    // 2. 编码并组合
    with ctx:
        prompt_ids = tokenizer.encode(prompt)
        response_ids = token_ids  // 已是token_ids
        full_ids = prompt_ids + response_ids
        
        input_tensor = torch.tensor([full_ids]).to(device)
        
        // 3. 前向传播获取logits
        outputs = model(input_ids=input_tensor)
        logits = outputs.logits[0]  // [seq_len, vocab_size]
        
        // 4. 提取response部分的对数概率
        // response_logits = logits[len(prompt_ids)-1:-1]
        // (注: -1因为predict下一个token需要前一步的hidden state)
        
        response_logits = logits[len(prompt_ids)-1:len(prompt_ids)-1+len(response_ids)]
        log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)
        
        // 5. 收集actual token的对数概率
        logprobs = []
        对于 i, token_id 在 enumerate(response_ids):
            logprob = log_probs[i, token_id]
            logprobs.append(logprob)
        
        logprob_tensor = torch.stack(logprobs)
    
    返回 logprob_tensor

```

---

## 7. 关键设计要点总结

### 7.1 GRPO与PPO的区别

| 特性 | PPO (标准) | GRPO (本项目) |
|------|-----------|-------------|
| 价值函数 | V(s) 显式 | **无显式V** |
| 优势估计 | TD-based | **相对优势** |
| 基准线 | V(s) | **组内均值** |
| 组大小 | 1 | **K个samples** |
| 适用场景 | 连续控制 | **序列生成** |

**GRPO优点**: 无需价值函数, 参数少; 组内标准化天然处理稀疏奖励

### 7.2 奖励设计

```
total_reward = mrr_weight * MRR@K + overlap_weight * overlap - text_penalty

| 项目 | 作用 |
|-----|------|
| MRR@K | 主要驱动: 检索质量 |
| overlap | 保真约束: 防止任意改写 |
| text_penalty | 质量约束: 避免生成退化 |
```

### 7.3 模型架构

- **Actor**: Qwen2.5-3B + LoRA (r=16, α=32)
- **Ref**: 冻结基础模型 (不量化, 精度float16)
- **Quantization**: 4-bit QLoRA (BitsAndBytes)
- **可训练参数**: ~3M (vs 3B基础模型)

### 7.4 训练超参关键设置

- **clip_range=0.2**: PPO截断范围 (标准值)
- **kl_beta=0.02**: KL系数 (默认较小, 避免policy偏移过快)
- **group_size=4**: 每query采4个候选 (平衡多样性与计算量)
- **learning_rate=2e-5**: 相对较小 (LoRA微调)
- **temperature=1.0**: 采样温度 (标准随机程度)

### 7.5 低内存模式支持

```
低内存模式配置:
- 模型: Qwen2.5-0.5B (vs 3B)
- batch_size: 1
- group_size: 2 (最少值)
- max_tokens: 16
- reward_topk: 20 (增加) - 提高hit概率
- 数据: max_train=64, max_val=32
- 索引: slim版本 (~0.5GB)

目标: 6GB VRAM快速验证端到端流程
```

---

## 8. 评估流程伪代码

```pseudocode
函数 evaluate_compare_three_ways():
    """三路对比评估"""
    
    // 加载配置与数据
    config = 加载默认配置()
    model_rl = 加载训练后的模型(args.rl_adapter_path)
    rewarder = 初始化Rewarder(...)
    eval_queries = 加载验证集()
    
    // 方式1: 原始查询
    original_results = {}
    对于 query 在 eval_queries:
        score = rewarder.计算奖励(query.qid, query.text, query.text)
        original_results[query.qid] = (query.text, score)
    
    // 方式2: Zero-shot重写 (基础模型无LoRA)
    model_zeroshot = 加载基础模型(no_adapter=True)
    zeroshot_results = {}
    对于 query 在 eval_queries:
        rewritten = model_zeroshot.生成重写(query.text)
        score = rewarder.计算奖励(query.qid, rewritten, query.text)
        zeroshot_results[query.qid] = (rewritten, score)
    
    // 方式3: RL优化重写
    rl_results = {}
    对于 query 在 eval_queries:
        rewritten = model_rl.生成重写(query.text)
        score = rewarder.计算奖励(query.qid, rewritten, query.text)
        rl_results[query.qid] = (rewritten, score)
    
    // 汇总指标
    report = {
        "original": {
            "mrr_mean": mean([v[1].mrr for v in original_results.values()]),
            "sample_count": len(original_results)
        },
        "zeroshot": {
            "mrr_mean": mean([v[1].mrr for v in zeroshot_results.values()]),
            "improvement_vs_original": ...,
            "sample_count": len(zeroshot_results)
        },
        "rl": {
            "mrr_mean": mean([v[1].mrr for v in rl_results.values()]),
            "improvement_vs_original": ...,
            "improvement_vs_zeroshot": ...,
            "sample_count": len(rl_results)
        }
    }
    
    // 采样打印对比
    对于 i 从 1 到 args.sample_print:
        query_id = sample_queries[i]
        打印 """
        Query ID: {query_id}
        Original: {original_results[query_id][0]}
        Zero-shot: {zeroshot_results[query_id][0]}
        RL-trained: {rl_results[query_id][0]}
        MRR (Orig/Zero/RL): {original_results[query_id][1].mrr:.4f} / 
                           {zeroshot_results[query_id][1].mrr:.4f} / 
                           {rl_results[query_id][1].mrr:.4f}
        """
    
    保存报告(args.report_path, report)

```

---

## 9. 数据流向图

```
输入 queries
    ↓
┌────────────────────────────────┐
│ 1. 数据加载 (loader.py)        │
│   - Pyserini topics/qrels      │
│   - 按比例分割train/val        │
└────────────────────────────────┘
    ↓ train_queries
┌────────────────────────────────┐
│ 2. 训练循环 (train.py)         │
│   - batch采样    ←─────────┐   │
│   - 生成重写     │         │   │
│   - 计算奖励     │ epoch   │   │
│   - 优势标准化   │ loop    │   │
│   - 损失计算与反传└─────────┤   │
│   - 梯度更新            │   │
└────────────────────────────────┘
    ↓ actor模型
┌────────────────────────────────┐
│ 3. 周期评估 (train.py)         │
│   - 验证集推理                 │
│   - 计算MRR@10                 │
│   - 保存best/latest checkpoint │
└────────────────────────────────┘
    ↓ 训练后模型
┌────────────────────────────────┐
│ 4. 三路对比 (eval_compare.py)  │
│   - Original baseline          │
│   - Zero-shot baseline         │
│   - RL-trained query rewriter  │
│   - 性能汇总                   │
└────────────────────────────────┘
    ↓
输出: eval_report.json + train_log.jsonl
```

---

## 10. 文件关系依赖图

```
train.py (主训练脚本)
├── imports: app_config
├── imports: core.grpo_engine
├── imports: core.model_wrapper
├── imports: core.reward_func
├── imports: data.loader
└── orchestrates: 整个训练流程

app_config.py (全局配置)
├── DataConfig
├── ModelConfig
├── TrainConfig
├── RewardConfig
└── PromptConfig

core/
├── grpo_engine.py (GRPO核心)
│   ├── uses: model_wrapper (生成与logprob计算)
│   ├── uses: reward_func (奖励)
│   ├── uses: optimizer (参数更新)
│   └── class GRPOEngine
│       └── train_step(): batch → metrics
│
├── model_wrapper.py (模型管理)
│   ├── loads: transformers (LLM)
│   ├── loads: peft (LoRA)
│   ├── manages: actor & ref模型
│   ├── provides: generate_with_logprob()
│   └── provides: compute_logprob()
│
└── reward_func.py (奖励计算)
    ├── uses: pyserini (检索)
    ├── provides: MRR@k
    ├── provides: text penalties
    └── class Rewarder
        └── score(): (qid, query) → RewardBreakdown

data/
└── loader.py (数据加载)
    ├── loads: pyserini (topics/qrels)
    ├── ensures: Java runtime验证
    └── provides: split_queries()

eval_compare.py (评估脚本)
├── loads: 训练后的adapter
├── loads: zero-shot基础模型
└── compares: Original vs Zero-shot vs RL
```

---

## 总结

这个项目实现了**完整的查询重写强化学习系统**:

1. **核心创新**: 自写GRPO算法，用组内相对优势替代显式价值函数
2. **效率优化**: 4-bit QLoRA使3B模型在单卡24GB运行；LoRA适配器仅3M参数
3. **奖励设计**: 多目标奖励（MRR检索+词面保真+文本质量）实现平衡优化
4. **灵活配置**: 支持低内存模式快速验证，支持CLI灵活覆盖参数
5. **完整流程**: 从数据加载→训练→评估→对比，一站式方案

**关键训练指标**: 检索MRR、PPO策略loss、KL散度、文本质量指标等全面监控
