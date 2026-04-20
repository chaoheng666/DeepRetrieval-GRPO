# Prompt Evaluation (0.8B Zero-shot)

This directory evaluates multiple retrieval rewrite prompts on the same random query sample, using an untrained base model (`Qwen/Qwen3.5-0.8B`) without any LoRA adapter.

## What it does

- Builds 50 prompt candidates from 25 BM25-specific strategy patterns x 2 profiles
  (`Direct` + `FewShot`).
- Every prompt is tied to this project rather than generic paraphrasing:
  - Lucene BM25 / MS MARCO passage retrieval
  - higher `MRR@50` instead of nicer prose
  - one-line English lexical query output
  - copy penalty / format penalty awareness
  - entity, acronym, number, version, negation preservation
- Samples 200 random validation queries with a fixed seed (default).
- Compares:
  - `Original` (no rewrite)
  - 50 prompt-based rewrites
- Reports aggregate metrics (`mrr`, `recall`, `reward_mean`) and deltas against Original.
- Computes query-level `win/tie/loss` against Original by per-query `mrr`.
- Cleans raw generations with the same query extractor used elsewhere in the project.
- Falls back to the original query when a generated rewrite is invalid or drops locked constraints
  (for example numbers, acronyms, or negations).
- Records prompt diagnostics such as fallback rate, raw format-fail rate, raw `<think>` rate,
  overlap, and rewrite rate.
- Saves progress after each prompt by default, so long runs are checkpointed.

## Quick start

```bash
python prompt_eval/run_prompt_eval.py --sample-size 200
```

Use prompt-level model parallel (5 model instances by default):

```bash
python prompt_eval/run_prompt_eval.py --sample-size 200 --prompt-model-parallel 5
```

### Optional args

- `--model-name` (default: `Qwen/Qwen3.5-0.8B`)
- `--seed` (default: `42`)
- `--query-batch-size` (default: `5`, higher = more GPU parallel generation)
- `--prompt-model-parallel` (default: `5`, load multiple model instances and assign prompts in parallel)
- `--topic-name`
- `--prebuilt-index`
- `--train-ratio`
- `--max-new-tokens`
- `--disable-4bit`
- `--reward-mrr-k`, `--reward-recall-k`, `--reward-recall-dense-k`
- `--reward-w-mrr`, `--reward-w-recall`, `--reward-w-recall-dense`
- `--reward-w-term-preserve`, `--reward-w-length-score`, `--reward-w-clean-format`
- `--reward-w-bad-format`, `--reward-w-unsafe-copy`
- `--report-path` (custom output path)
- `--prompt-ids` (comma-separated subset for slow incremental testing)
- `--max-prompts` (only run first N prompts after filtering)
- `--history-path` (custom history path)
- `--no-save-each-prompt` (disable per-prompt checkpointing)
- `--no-append-history` (disable history append)

### Slow incremental run examples

Run a small subset first:

```bash
python prompt_eval/run_prompt_eval.py \
  --sample-size 200 \
  --prompt-ids p01_det,p02_det,p03_det,p07_demo
```

Then continue with more prompts:

```bash
python prompt_eval/run_prompt_eval.py \
  --sample-size 200 \
  --max-prompts 10
```

## Output

By default, report is written to:

```text
prompt_eval/artifacts/seed<seed>_n<sample_size>/prompt_eval_report.json
```

And run summaries are appended to:

```text
prompt_eval/artifacts/seed<seed>_n<sample_size>/prompt_eval_history.json
```

Top-level JSON keys:

- `run_config`
- `sample_info`
- `baseline_original`
- `prompt_results`
- `leaderboard`
- `best_prompt`

## Notes

- "Untrained 0.8B model" means base model only (`enable_lora=False`, no adapter loaded).
- The sweep is now explicitly biased toward not losing to baseline for avoidable reasons:
  invalid rewrites are cleaned, and unsafe rewrites fall back to the source query.
- This improves the chance of beating baseline, but it still does not mathematically guarantee a
  positive `delta_mrr` on every sample; the report always shows whether the best prompt actually won.
