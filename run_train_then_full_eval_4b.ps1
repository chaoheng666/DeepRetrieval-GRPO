$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $repoRoot

try {
  $pythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
  $artifactRoot = if ($env:ARTIFACT_ROOT) { $env:ARTIFACT_ROOT } else { "train_and_eval_data_model" }
  $expName = if ($env:EXP_NAME) { $env:EXP_NAME } else { "4b" }

  $trainDir = Join-Path $artifactRoot "artifacts_${expName}_train"
  $evalDir = Join-Path $artifactRoot "artifacts_${expName}_eval"
  $trainCheckpointDir = Join-Path $trainDir "checkpoints"
  $trainLogPath = Join-Path $trainDir "train_log.jsonl"
  $trainTracePath = Join-Path $trainDir "group_trace_log.jsonl"
  $evalReportPath = Join-Path $evalDir "eval_compare_report_full.json"

  New-Item -ItemType Directory -Path $trainDir -Force | Out-Null
  New-Item -ItemType Directory -Path $evalDir -Force | Out-Null

  Write-Host "[1/3] Training 4B experiment..."
  & $pythonBin train.py `
    --model-name Qwen/Qwen3-4B-Instruct-2507 `
    --num-epochs 3 `
    --batch-size 4 `
    --group-size 8 `
    --max-new-tokens 20 `
    --temperature 0.7 `
    --top-p 0.9 `
    --save-dir $trainCheckpointDir `
    --log-path $trainLogPath `
    --group-trace-log-path $trainTracePath

  $bestAdapter = Join-Path $trainCheckpointDir "best"
  $latestAdapter = Join-Path $trainCheckpointDir "latest"
  $adapterPath = $null

  if (Test-Path -LiteralPath $bestAdapter) {
    $adapterPath = $bestAdapter
  }
  elseif (Test-Path -LiteralPath $latestAdapter) {
    $adapterPath = $latestAdapter
    Write-Warning "best adapter not found, fallback to latest adapter."
  }
  else {
    throw "No adapter found under $trainCheckpointDir (expected best or latest)."
  }

  Write-Host "[2/3] Running full evaluation for the 4B experiment..."
  & $pythonBin eval_compare.py `
    --rl-adapter-path $adapterPath `
    --model-name Qwen/Qwen3-4B-Instruct-2507 `
    --strict-tokenizer-model-match `
    --max-eval-queries 1000000 `
    --sample-print 20 `
    --report-path $evalReportPath

  Write-Host "Done. Report: $evalReportPath"

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
    git push
    Write-Host "[3/3] Commit created and pushed: $commitMessage"
  }
}
finally {
  Pop-Location
}
