---
marp: true
theme: default
paginate: true
style: |
  section {
    display: flex;
    flex-direction: column;    /* 纵向排列 */
    justify-content: flex-start; /* 核心：靠顶对齐 */
    align-items: flex-start;   /* 核心：靠左对齐 */
    padding: 40px 60px;        /* 控制四周留白 */
    background-color: #ffffff;
    font-size: 24px;           /* 全局字号控制 */
  }

  /* 2. 标题固定样式：每一页标题的位置都会保持一致 */
  h2 {
    margin-top: 0;             /* 消除标题上方多余间距 */
    margin-bottom: 20px;       /* 标题与下方内容的固定间距 */
    width: 100%;               /* 标题线撑满宽度 */
    color:  #000000;
    font-size: 1.5em;          /* 标题字号 */
    border-bottom: 2px solid  #000000; /* 程序员黑下划线 */
    padding-bottom: 10px;
    flex-shrink: 0;            /* 防止标题被内容挤压变形 */
  }

  /* 3. 内容区样式：紧跟标题，不上下乱动 */
  h3, ul, p {
    margin-top: 10px;
    margin-bottom: 10px;
    width: 100%;
  }

  /* 4. 页码格式：1 / 13 模式 */
  section::after {
    content: attr(data-marpit-pagination) ' / ' attr(data-marpit-pagination-total);
    font-size: 18px;
    color: #bdc3c7;
    position: absolute;        /* 绝对定位，固定在右下角 */
    bottom: 30px;
    right: 50px;
  }
  
  table {
    align-self: center; 
    margin-top: 20px;
    margin-bottom: 20px;
    border-collapse: collapse;
  }

  table th, table td {
    text-align: center;
    border: 1px solid #000;
    padding: 8px 15px;
  }

  strong {
    color:  #000000;
  }

  section.title-page {
    justify-content: center !important;
    align-items: center !important;
    text-align: center !important;
  }

  section.title-page h1 {
    font-size: 2.8em;
    color:  #000000;
    margin-bottom: 0.3em;
    border: none;
  }

  section.title-page h2 {
    all: unset;
    display: block;
    font-size: 1.3em;
    color:  #000000;
    margin-top: 10px;
    border: none;
  }

  section {
    width: 1280px;
    height: 720px;
  }
---
<!-- _class: title-page -->
# 基于强化学习的生成式检索
## Generative Retrieval via Reinforcement Learning

**汇报人：** 晁恒
**指导老师：** 辛鑫
**单位：** 山东大学计算机科学与技术学院
**日期：** 2026年4月9日

---

## 1. 研究背景 (Research Background)

### 核心痛点
* **语义鸿沟：** 用户输入的 `Query` 通常模糊或缺乏关键语义，导致检索召回率低。
* **匹配限制：** 传统检索器（如 BM25）强依赖词面匹配，无法理解深层意图。

### 生成式检索的优势
* **Query 重写：** 利用大语言模型（LLM）的生成能力，将原始 Query 补全为高质量检索文本。
* **端到端优化：** 通过反馈机制（Reinforcement Learning）使改写结果直接对齐检索精度。

---

## 2. 研究目标与任务 (Research Objectives)

### 核心目标
本项目的本质是利用 GRPO 的组内博弈机制，在海量的语义改写候选中，挖掘出那些**对检索器而言最具区分度**的 Token 组合，从而弥补预训练语言模型与特定检索工具（BM25等）之间的任务鸿沟。

### 关键阶段
1. **理论建模：** 定义检索场景下的状态空间与动作空间。
2. **算法实现：** 适配 **GRPO** 算法，解决算力受限下的强化学习训练问题。
3. **闭环实验：** 在 MS MARCO 数据集上验证检索增益，并完成性能对比。

---

## 3. 理论建模：面向生成式检索的 MDP 设计

| **要素** | **定义** | **含义** |
| :--- | :--- | :--- |
| **状态 (State, $S_t$)** | 上下文环境 | 初始 Query Prompt + 当前已生成的 Token 序列。 |
| **动作 (Action, $A_t$)** | 模型决策 | 词表上的概率分布，决定生成的下一个 Token。 |
| **奖励 (Reward, $R$)** | 反馈信号 | 序列结束后的延迟奖励，衡量改写后的检索质量。 |

> **动态逻辑：** 模型作为 Agent，随生成进程不断更新状态，最终由外部检索环境给出综合评分。

---

## 4. 核心算法：为什么从 PPO 转向 GRPO？
<style>
img[alt~="center"] {
display: block;
margin: 0 auto;
}
</style>
![w:640 center](img/GRPOvsPPO.png)

### PPO 的显存瓶颈
* 必须训练额外的 **Critic 网络** 来估算 Value，显存开销通常是策略网络的 2 倍。

### GRPO 的优势
* **去 Critic 化：** 通过组内采样替代价值估算。
* **算力友好：** 在同等显存下支持更大规模的 LLM（如从 0.5B 扩展至 3B/7B）。

---

## 5. GRPO 的优势函数与组内采样机制

### 核心公式
$$A_i = \frac{r_i - \mu}{\sigma + \epsilon}$$

### 组内相对优势
* **采样：** 对同一个 Query 同时采样 $G$ 个不同的回答。
* **标准化：** $r_i$ 为第 $i$ 个回答的得分，$\mu$ 和 $\sigma$ 是这 $G$ 个样本得分的均值和标准差。
* **逻辑：** 这种机制使得模型学习“哪些改写比同组其他改写更好”，而非依赖绝对分数值。

---

## 6. GRPO 的损失函数与 KL 约束

### 损失函数分解
1. **裁剪代理目标 (Clipped Surrogate Objective):**
$$L_{GRPO} = \mathbb{E} \left[ \min \left( \frac{\pi_{\theta}}{\pi_{old}} A_i, \text{clip} \left( \frac{\pi_{\theta}}{\pi_{old}}, 1-\epsilon, 1+\epsilon \right) A_i \right) \right]$$
   * 限制更新步长，防止模型因单次奖励过大而导致策略崩溃。

2. **KL 正则项约束:**
$$L = L_{GRPO} - \beta \cdot KL(\pi_{\theta} || \pi_{ref})$$
   * 惩罚偏离参考模型过远的行为，确保生成的 Query 保持 **人类可读性**。

---

## 7. 奖励函数设计：检索驱动 (Core)

**奖励 = 检索质量得分 + 文本约束 - 负面惩罚**

### 检索质量得分
* **工具：** 集成 **Pyserini BM25** 检索器。
* **指标：** 计算 **MRR@50** (Mean Reciprocal Rank)。
* **逻辑：** 利用 MRR@50 衡量检索增益，引导模型优化 Query 的语义命中精度，确保目标文档排位靠前。

### 文本约束
* **Jaccard 相似度：** 计算改写词与原词的交并比，防止模型在改写时完全脱离原意。

### 负面惩罚
* **长度惩罚：** 防止生成无意义的长文本。
* **重复惩罚：** 针对 RL 训练中常见的 Token 循环生成问题进行强力截断。
* **自然性：** 剔除包含大量不可读乱码或特殊符号的序列。

---

## 9. 实验设计：模型演化与规模化策略 (Scaling Strategy)

### 1. 验证期：0.5B 规模 (全链路闭环)
* **目标：** 在极低显存占用下，验证 **LLM + GRPO + Pyserini** 自动化训练流。
* **核心：** 确立奖励函数（Reward Function）收敛性，防止强化学习早期的“乱码”输出。

### 2. 训练期：3B 规模 (性能对齐)
* **目标：** 利用中等规模模型的表达能力，实现改写语义与检索性能的实质性对齐。
* **策略：** 基于 **MS MARCO** 数据集抽取 **20% Query** 进行高强度迭代，兼顾训练效率与统计显著性。

### 3. 优化期：7B 规模 (指标突破)
* **目标：** 进一步挖掘模型潜力，通过更精细的超参数微调，追求 MRR@50 指标的极限突破。
* **方向：** 对标生成式检索领域的 SOTA 表现，验证算法在高参数量下的泛化性。

---

## 10. 环境设置：资产与算力的解耦工程

### 核心架构：资产云端化 + 算力容器化
* **数据仓库管理：**
  * 建立统一的 **Remote Data Warehouse**，存储 MS MARCO 全量数据集与各版本 Checkpoints。
  * 实现“数据与算力分离”，确保核心研究资产不随算力释放而丢失。
* **按量算力调度：**
  * **动态租赁：** 根据实验规模（0.5B/3B/7B）按需租赁 GPU 容器实例（如 RTX 4090/A100）。
  * **自动同步：** 开发自动化脚本，在训练节点启动时自动拉取数据，任务结束后自动同步权重并释放资源。
* **实验环境：**
  * **软件栈：** Python 3.13+ / PyTorch / Transformers / Pyserini (Lucence Indexing)。
  * **硬件基准：** 算力调度确保在显存与计算密度之间达成最优性价比。

---

## 11. 阶段性成果：定量指标分析

### 实验结果对比

| 方案 | MRR@50 (Mean) | Reward (Mean) |
| :--- | :---: | :---: |
| **Original (原始检索)** | **0.2411** | **0.4411** |
| **Zero-shot (模型改写)** | 0.1707 | 0.2246 |
| **RL-trained (初步训练)** | 0.2090 | 0.2857 |

### 结果解读
* **现状：** 目前初步训练的模型在检索精度上尚未超越原始 Original，但是比 Zero-shot 精度高。
* **成因：** 模型参数量不足以同时表征复杂的“语义重写逻辑”与“底层检索信号”。0.5B 模型难以捕捉到 Token 变化与 MRR 增益之间的微弱关联。

---
## 12. 未来计划 (Future Plans)

### 1. 规模化扩展
* 迁移至 **3B/7B 参数规模** 模型，利用更强的语言先验知识提升改写质量。

### 2. 奖励机制迭代
* 进一步分析 KL 散度与检索准确率之间的 **Pareto 边界**，平衡改写强度与检索性能。

### 3. 消融实验
* 探究不同采样数 $G$ 对 GRPO 优势函数稳定性的影响。

---
<!-- _class: title-page -->
# 感谢各位专家的聆听
### 欢迎批评指正！

**汇报人：** 晁恒
**日期：** 2026年4月9日
