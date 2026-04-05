# Minimal Retrieval Eval Environment (MS MARCO)

这个仓库用于快速构建一个最小可运行的检索评估闭环：

- `corpus`（文档库）
- `queries`（查询）
- `qrels`（query -> relevant docs）
- BM25 检索（优先 Pyserini，自动兜底 rank_bm25）
- reward 计算（MRR）

## 1. 环境准备

```bash
python -m pip install -r requirements.txt
```

可选（若你希望使用 Lucene/Pyserini 索引）：

```bash
python -m pip install pyserini
```

说明：当前 Python 3.13 下，`pyserini` 可能安装失败。脚本会自动回退到 `rank_bm25`，不会阻塞 reward 验证。

## 2. 准备数据

```bash
python scripts/prepare_data.py --dataset ms_marco --split "train[:5000]" --seed 42
```

输出：

- `data/corpus.jsonl`（`{"id": "...", "contents": "..."}`）
- `data/queries.jsonl`（`{"query_id": "...", "text": "..."}`）
- `data/qrels.json`（`{"Q...": ["D..."]}`）
- `data/stats.json`

如果 `ms_marco` 下载异常，会自动尝试 `BeIR/msmarco` 结构。

`train[:N]` 会优先走流式读取（streaming），减少整包下载等待时间。

## 3. 建索引

```bash
python scripts/build_index.py --input data --index index --backend auto
```

- 若 `pyserini` 可用：构建 Lucene 索引。
- 若不可用或构建失败：写入 `index/backend.json` 并切换到 `rank_bm25`。

## 4. 评估 MRR reward

```bash
python scripts/eval_mrr.py \
  --queries data/queries.jsonl \
  --qrels data/qrels.json \
  --corpus data/corpus.jsonl \
  --index index \
  --topk 10 \
  --sample-size 100 \
  --seed 42
```

验证阈值（默认）：

- 平均 reward `> 0.2`
- 非 0 比例 `> 0.5`

若不达标，脚本会返回非 0 退出码，并打印最多 20 个失败样例。详细报告保存到 `runs/eval_report.json`。

## 5. 单元测试

```bash
python -m unittest discover -s tests -p "test_*.py"
```

覆盖点：

- `compute_reward` 的 rank 命中逻辑
- `text -> doc_id` 映射稳定性与空值过滤
