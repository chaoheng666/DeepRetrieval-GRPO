Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $repoRoot

try {
  $pythonBin = "python"
  $logDir = "log"
  $runRoot = Join-Path "train_and_eval_data_model_0420" "artifacts_4b_top20_delta_curriculum"
  $phase1Dir = Join-Path $runRoot "phase1"
  $phase2Dir = Join-Path $runRoot "phase2"
  $evalDir = Join-Path $runRoot "eval"
  $curriculumMetadataPath = Join-Path $runRoot "curriculum_query_metadata.jsonl"
  $phase1CheckpointDir = Join-Path $phase1Dir "checkpoints"
  $phase2CheckpointDir = Join-Path $phase2Dir "checkpoints"
  $phase1LogPath = Join-Path $phase1Dir "train_log.jsonl"
  $phase2LogPath = Join-Path $phase2Dir "train_log.jsonl"
  $phase1TracePath = Join-Path $phase1Dir "group_trace_log.jsonl"
  $phase2TracePath = Join-Path $phase2Dir "group_trace_log.jsonl"
  $evalReportPath = Join-Path $evalDir "eval_compare_report_full.json"

  New-Item -ItemType Directory -Path $logDir -Force | Out-Null
  New-Item -ItemType Directory -Path $phase1Dir -Force | Out-Null
  New-Item -ItemType Directory -Path $phase2Dir -Force | Out-Null
  New-Item -ItemType Directory -Path $evalDir -Force | Out-Null

  Write-Host "[env] python=$pythonBin model=Qwen/Qwen3-4B-Instruct-2507 search_threads=16"
  Write-Host "[env] curriculum_metadata=$curriculumMetadataPath"

  $commonTrainArgs = @(
    "--model-name", "Qwen/Qwen3-4B-Instruct-2507",
    "--clip-range", "0.2",
    "--ref-precision-mode", "4bit",
    "--search-threads", "16",
    "--eval-query-batch-size", "8",
    "--group-temperature-stride", "0.08",
    "--group-top-p-stride", "0.02",
    "--min-unique-final-queries", "5",
    "--max-regen-rounds", "3",
    "--reward-gap-threshold", "0.12",
    "--gap-sampling-temperature-delta", "0.18",
    "--actor-chunk-size", "2",
    "--projection-chunk-size", "64",
    "--max-val-queries", "400",
    "--reward-mode", "top20_delta",
    "--reward-mrr-k", "20",
    "--reward-recall-k", "20",
    "--reward-recall-dense-k", "50",
    "--reward-w-bad-format", "0.18",
    "--reward-w-unsafe-copy", "0.14",
    "--reward-w-overedit", "0.08",
    "--overedit-tau", "0.45",
    "--recall-drop-lambda", "0.8",
    "--anchor-bonus-value", "0.05",
    "--format-max-tokens", "12",
    "--format-min-english-ratio", "0.85",
    "--format-max-unreadable-ratio", "0.20",
    "--curriculum-enable",
    "--curriculum-metadata-path", $curriculumMetadataPath
  )

  Write-Host "[phase1] epochs=2 batch=24 group=8 max_group=12 lr=1.0e-5 kl=0.040 decode=(10,0.82,0.93) reward=(0.40,0.28,0.22,0.10)"
  & $pythonBin train.py @commonTrainArgs `
    --curriculum-phase phase1 `
    --num-epochs 2 `
    --batch-size 24 `
    --group-size 8 `
    --max-group-size 12 `
    --learning-rate 1.0e-5 `
    --kl-beta 0.040 `
    --max-new-tokens 10 `
    --temperature 0.82 `
    --top-p 0.93 `
    --eval-max-new-tokens 10 `
    --eval-temperature 0.82 `
    --eval-top-p 0.93 `
    --eval-every-steps 20 `
    --max-steps 100 `
    --reward-w-mrr 0.40 `
    --reward-w-recall 0.28 `
    --reward-w-recall-dense 0.22 `
    --reward-w-rank-bonus 0.10 `
    --save-dir $phase1CheckpointDir `
    --log-path $phase1LogPath `
    --group-trace-log-path $phase1TracePath

  $phase1Best = Join-Path $phase1CheckpointDir "best"
  if (-not (Test-Path -LiteralPath $phase1Best)) {
    throw "phase1 best checkpoint not found: $phase1Best"
  }

  Write-Host "[phase2] epochs=1 batch=24 group=8 max_group=12 lr=6.0e-6 kl=0.055 decode=(10,0.80,0.92) reward=(0.52,0.22,0.16,0.10)"
  & $pythonBin train.py @commonTrainArgs `
    --curriculum-phase phase2 `
    --adapter-path $phase1Best `
    --num-epochs 1 `
    --batch-size 24 `
    --group-size 8 `
    --max-group-size 12 `
    --learning-rate 6.0e-6 `
    --kl-beta 0.055 `
    --max-new-tokens 10 `
    --temperature 0.80 `
    --top-p 0.92 `
    --eval-max-new-tokens 10 `
    --eval-temperature 0.80 `
    --eval-top-p 0.92 `
    --eval-every-steps 20 `
    --max-steps 60 `
    --reward-w-mrr 0.52 `
    --reward-w-recall 0.22 `
    --reward-w-recall-dense 0.16 `
    --reward-w-rank-bonus 0.10 `
    --save-dir $phase2CheckpointDir `
    --log-path $phase2LogPath `
    --group-trace-log-path $phase2TracePath

  $phase2Best = Join-Path $phase2CheckpointDir "best"
  if (-not (Test-Path -LiteralPath $phase2Best)) {
    throw "phase2 best checkpoint not found: $phase2Best"
  }

  Write-Host "[eval] adapter=$phase2Best query_batch_size=8"
  & $pythonBin eval_compare.py `
    --rl-adapter-path $phase2Best `
    --model-name Qwen/Qwen3-4B-Instruct-2507 `
    --strict-tokenizer-model-match `
    --max-eval-queries 1000000 `
    --search-threads 16 `
    --max-new-tokens 10 `
    --temperature 0.80 `
    --top-p 0.92 `
    --query-batch-size 8 `
    --reward-mode top20_delta `
    --reward-mrr-k 20 `
    --reward-recall-k 20 `
    --reward-recall-dense-k 50 `
    --reward-w-mrr 0.52 `
    --reward-w-recall 0.22 `
    --reward-w-recall-dense 0.16 `
    --reward-w-rank-bonus 0.10 `
    --reward-w-bad-format 0.18 `
    --reward-w-unsafe-copy 0.14 `
    --reward-w-overedit 0.08 `
    --overedit-tau 0.45 `
    --recall-drop-lambda 0.8 `
    --anchor-bonus-value 0.05 `
    --format-max-tokens 12 `
    --format-min-english-ratio 0.85 `
    --format-max-unreadable-ratio 0.20 `
    --sample-print 20 `
    --report-path $evalReportPath

  Write-Host "[done] report: $evalReportPath"
}
finally {
  Pop-Location
}
