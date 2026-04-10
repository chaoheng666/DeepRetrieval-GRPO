$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $repoRoot

try {
  Write-Host "[1/2] Training 3B experiment..."
  python train.py `
    --model-name Qwen/Qwen3-4B-Instruct-2507 `
    --num-epochs 3 `
    --batch-size 4 `
    --group-size 8 `
    --max-new-tokens 20 `
    --temperature 0.7 `
    --top-p 0.9 `
    --save-dir artifacts_3b_train/checkpoints `
    --log-path artifacts_3b_train/train_log.jsonl `
    --group-trace-log-path artifacts_3b_train/group_trace_log.jsonl

  $bestAdapter = "artifacts_3b_train/checkpoints/best"
  $latestAdapter = "artifacts_3b_train/checkpoints/latest"
  $adapterPath = $null

  if (Test-Path -LiteralPath $bestAdapter) {
    $adapterPath = $bestAdapter
  }
  elseif (Test-Path -LiteralPath $latestAdapter) {
    $adapterPath = $latestAdapter
    Write-Warning "best adapter not found, fallback to latest adapter."
  }
  else {
    throw "No adapter found under artifacts_3b_train/checkpoints (expected best or latest)."
  }

  Write-Host "[2/2] Running full evaluation for the 3B experiment..."
  python eval_compare.py `
    --rl-adapter-path $adapterPath `
    --model-name Qwen/Qwen3-4B-Instruct-2507 `
    --strict-tokenizer-model-match `
    --max-eval-queries 1000000 `
    --sample-print 20 `
    --report-path artifacts_3b_eval/eval_compare_report_full.json

  Write-Host "Done. Report: artifacts_3b_eval/eval_compare_report_full.json"

  Write-Host "[3/3] Auto-committing all changes to git repository..."
  git rev-parse --is-inside-work-tree *> $null
  if ($LASTEXITCODE -ne 0) {
    throw "Current directory is not a git repository."
  }

  git add -A
  git diff --cached --quiet
  if ($LASTEXITCODE -eq 0) {
    Write-Host "[3/3] No staged changes to commit. Skipped."
  }
  else {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $commitMessage = "chore: auto commit training and full eval artifacts $timestamp"
    git commit -m $commitMessage
    Write-Host "[3/3] Commit created: $commitMessage"
  }
}
finally {
  Pop-Location
}
