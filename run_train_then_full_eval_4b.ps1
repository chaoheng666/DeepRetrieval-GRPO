$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $repoRoot

$transcriptStarted = $false
$runLogPath = $null

try {
  $logDir = if ($env:LOG_DIR) { $env:LOG_DIR } else { "log" }
  New-Item -ItemType Directory -Path $logDir -Force | Out-Null
  $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $runLogPath = Join-Path $logDir "run_train_then_full_eval_4b_$timestamp.log"

  try {
    Start-Transcript -Path $runLogPath -Force | Out-Null
    $transcriptStarted = $true
    Write-Host "[log] command output is also saved to: $runLogPath"
  }
  catch {
    Write-Warning "Failed to start transcript log at '$runLogPath'. Continuing without transcript."
  }

  $pythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
  $artifactRoot = if ($env:ARTIFACT_ROOT) { $env:ARTIFACT_ROOT } else { "train_and_eval_data_model_0420" }
  $expName = if ($env:EXP_NAME) { $env:EXP_NAME } else { "4b_conservative_mrr" }
  $modelName = if ($env:MODEL_NAME) { $env:MODEL_NAME } else { "Qwen/Qwen3-4B-Instruct-2507" }
  $autoGitCommit = if ($env:AUTO_GIT_COMMIT) { $env:AUTO_GIT_COMMIT } else { "1" }
  $autoGitPush = if ($env:AUTO_GIT_PUSH) { $env:AUTO_GIT_PUSH } else { "1" }
  $searchThreads = if ($env:SEARCH_THREADS) { $env:SEARCH_THREADS } else { "16" }
  $trainBatchSize = if ($env:TRAIN_BATCH_SIZE) { $env:TRAIN_BATCH_SIZE } else { "24" }
  $trainGroupSize = if ($env:TRAIN_GROUP_SIZE) { $env:TRAIN_GROUP_SIZE } else { "8" }
  $trainMaxGroupSize = if ($env:TRAIN_MAX_GROUP_SIZE) { $env:TRAIN_MAX_GROUP_SIZE } else { "12" }
  $trainLearningRate = if ($env:TRAIN_LEARNING_RATE) { $env:TRAIN_LEARNING_RATE } else { "1.5e-5" }
  $trainClipRange = if ($env:TRAIN_CLIP_RANGE) { $env:TRAIN_CLIP_RANGE } else { "0.2" }
  $trainKlBeta = if ($env:TRAIN_KL_BETA) { $env:TRAIN_KL_BETA } else { "0.03" }
  $trainRefPrecisionMode = if ($env:TRAIN_REF_PRECISION_MODE) { $env:TRAIN_REF_PRECISION_MODE } else { "4bit" }
  $trainMaxNewTokens = if ($env:TRAIN_MAX_NEW_TOKENS) { $env:TRAIN_MAX_NEW_TOKENS } else { "12" }
  $trainTemperature = if ($env:TRAIN_TEMPERATURE) { $env:TRAIN_TEMPERATURE } else { "0.85" }
  $trainTopP = if ($env:TRAIN_TOP_P) { $env:TRAIN_TOP_P } else { "0.95" }
  $evalMaxNewTokens = if ($env:EVAL_MAX_NEW_TOKENS) { $env:EVAL_MAX_NEW_TOKENS } else { $trainMaxNewTokens }
  $evalTemperature = if ($env:EVAL_TEMPERATURE) { $env:EVAL_TEMPERATURE } else { $trainTemperature }
  $evalTopP = if ($env:EVAL_TOP_P) { $env:EVAL_TOP_P } else { $trainTopP }
  $evalQueryBatchSize = if ($env:EVAL_QUERY_BATCH_SIZE) { $env:EVAL_QUERY_BATCH_SIZE } else { "8" }
  $trainGroupTemperatureStride = if ($env:TRAIN_GROUP_TEMPERATURE_STRIDE) { $env:TRAIN_GROUP_TEMPERATURE_STRIDE } else { "0.07" }
  $trainGroupTopPStride = if ($env:TRAIN_GROUP_TOP_P_STRIDE) { $env:TRAIN_GROUP_TOP_P_STRIDE } else { "0.015" }
  $trainMinUniqueFinalQueries = if ($env:TRAIN_MIN_UNIQUE_FINAL_QUERIES) { $env:TRAIN_MIN_UNIQUE_FINAL_QUERIES } else { "4" }
  $trainMaxRegenRounds = if ($env:TRAIN_MAX_REGEN_ROUNDS) { $env:TRAIN_MAX_REGEN_ROUNDS } else { "2" }
  $trainRewardGapThreshold = if ($env:TRAIN_REWARD_GAP_THRESHOLD) { $env:TRAIN_REWARD_GAP_THRESHOLD } else { "0.08" }
  $trainGapSamplingTemperatureDelta = if ($env:TRAIN_GAP_SAMPLING_TEMPERATURE_DELTA) { $env:TRAIN_GAP_SAMPLING_TEMPERATURE_DELTA } else { "0.15" }
  $trainEvalEverySteps = if ($env:TRAIN_EVAL_EVERY_STEPS) { $env:TRAIN_EVAL_EVERY_STEPS } else { "50" }
  $trainMaxSteps = if ($env:TRAIN_MAX_STEPS) { $env:TRAIN_MAX_STEPS } else { "500" }
  $trainMaxValQueries = if ($env:TRAIN_MAX_VAL_QUERIES) { $env:TRAIN_MAX_VAL_QUERIES } else { "400" }

  $rewardMrrK = if ($env:REWARD_MRR_K) { $env:REWARD_MRR_K } else { "50" }
  $rewardRecallK = if ($env:REWARD_RECALL_K) { $env:REWARD_RECALL_K } else { "50" }
  $rewardRecallDenseK = if ($env:REWARD_RECALL_DENSE_K) { $env:REWARD_RECALL_DENSE_K } else { "100" }
  $rewardWMrr = if ($env:REWARD_W_MRR) { $env:REWARD_W_MRR } else { "0.40" }
  $rewardWRecall = if ($env:REWARD_W_RECALL) { $env:REWARD_W_RECALL } else { "0.20" }
  $rewardWRecallDense = if ($env:REWARD_W_RECALL_DENSE) { $env:REWARD_W_RECALL_DENSE } else { "0.15" }
  $rewardWTermPreserve = if ($env:REWARD_W_TERM_PRESERVE) { $env:REWARD_W_TERM_PRESERVE } else { "0.10" }
  $rewardWLengthScore = if ($env:REWARD_W_LENGTH_SCORE) { $env:REWARD_W_LENGTH_SCORE } else { "0.08" }
  $rewardWCleanFormat = if ($env:REWARD_W_CLEAN_FORMAT) { $env:REWARD_W_CLEAN_FORMAT } else { "0.07" }
  $rewardWBadFormat = if ($env:REWARD_W_BAD_FORMAT) { $env:REWARD_W_BAD_FORMAT } else { "0.15" }
  $rewardWUnsafeCopy = if ($env:REWARD_W_UNSAFE_COPY) { $env:REWARD_W_UNSAFE_COPY } else { "0.08" }
  $formatMaxTokens = if ($env:FORMAT_MAX_TOKENS) { $env:FORMAT_MAX_TOKENS } else { "12" }
  $formatMinEnglishRatio = if ($env:FORMAT_MIN_ENGLISH_RATIO) { $env:FORMAT_MIN_ENGLISH_RATIO } else { "0.8" }
  $formatMaxUnreadableRatio = if ($env:FORMAT_MAX_UNREADABLE_RATIO) { $env:FORMAT_MAX_UNREADABLE_RATIO } else { "0.25" }

  $trainDir = Join-Path $artifactRoot "artifacts_${expName}_train"
  $evalDir = Join-Path $artifactRoot "artifacts_${expName}_eval"
  $trainCheckpointDir = Join-Path $trainDir "checkpoints"
  $trainLogPath = Join-Path $trainDir "train_log.jsonl"
  $trainTracePath = Join-Path $trainDir "group_trace_log.jsonl"
  $evalReportPath = Join-Path $evalDir "eval_compare_report_full.json"

  New-Item -ItemType Directory -Path $trainDir -Force | Out-Null
  New-Item -ItemType Directory -Path $evalDir -Force | Out-Null

  $looksLikeLocalPath =
    [System.IO.Path]::IsPathRooted($modelName) -or
    $modelName.StartsWith(".\") -or
    $modelName.StartsWith("./") -or
    $modelName.StartsWith("..\") -or
    $modelName.StartsWith("../")
  $modelPathExists = Test-Path -LiteralPath $modelName
  if ($looksLikeLocalPath -and -not $modelPathExists) {
    throw "MODEL_NAME local directory not found: $modelName. Download model first, or set MODEL_NAME to an existing local directory/model id."
  }
  if ($modelPathExists) {
    $item = Get-Item -LiteralPath $modelName
    if (-not $item.PSIsContainer) {
      throw "MODEL_NAME exists but is not a directory: $modelName."
    }
    Write-Host "[env] model source (local dir): $modelName"
  }
  else {
    Write-Host "[env] model source (model id): $modelName"
  }

  if (-not $env:PYTORCH_CUDA_ALLOC_CONF) {
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
  }

  Write-Host "[preset] batch=$trainBatchSize group=$trainGroupSize max_group=$trainMaxGroupSize train_decode=($trainMaxNewTokens,$trainTemperature,$trainTopP) eval_decode=($evalMaxNewTokens,$evalTemperature,$evalTopP) eval_query_batch_size=$evalQueryBatchSize strides=($trainGroupTemperatureStride,$trainGroupTopPStride) unique=$trainMinUniqueFinalQueries regen=$trainMaxRegenRounds gap=$trainRewardGapThreshold delta=$trainGapSamplingTemperatureDelta reward=(mrr=$rewardMrrK recall=$rewardRecallK recall_dense=$rewardRecallDenseK)"

  Write-Host "[1/3] Training 4B experiment..."
  & $pythonBin train.py `
    --model-name $modelName `
    --num-epochs 1 `
    --batch-size $trainBatchSize `
    --group-size $trainGroupSize `
    --max-group-size $trainMaxGroupSize `
    --learning-rate $trainLearningRate `
    --clip-range $trainClipRange `
    --kl-beta $trainKlBeta `
    --ref-precision-mode $trainRefPrecisionMode `
    --search-threads $searchThreads `
    --max-new-tokens $trainMaxNewTokens `
    --temperature $trainTemperature `
    --top-p $trainTopP `
    --eval-max-new-tokens $evalMaxNewTokens `
    --eval-temperature $evalTemperature `
    --eval-top-p $evalTopP `
    --eval-query-batch-size $evalQueryBatchSize `
    --group-temperature-stride $trainGroupTemperatureStride `
    --group-top-p-stride $trainGroupTopPStride `
    --min-unique-final-queries $trainMinUniqueFinalQueries `
    --max-regen-rounds $trainMaxRegenRounds `
    --reward-gap-threshold $trainRewardGapThreshold `
    --gap-sampling-temperature-delta $trainGapSamplingTemperatureDelta `
    --eval-every-steps $trainEvalEverySteps `
    --max-val-queries $trainMaxValQueries `
    --max-steps $trainMaxSteps `
    --reward-mrr-k $rewardMrrK `
    --reward-recall-k $rewardRecallK `
    --reward-recall-dense-k $rewardRecallDenseK `
    --reward-w-mrr $rewardWMrr `
    --reward-w-recall $rewardWRecall `
    --reward-w-recall-dense $rewardWRecallDense `
    --reward-w-term-preserve $rewardWTermPreserve `
    --reward-w-length-score $rewardWLengthScore `
    --reward-w-clean-format $rewardWCleanFormat `
    --reward-w-bad-format $rewardWBadFormat `
    --reward-w-unsafe-copy $rewardWUnsafeCopy `
    --format-max-tokens $formatMaxTokens `
    --format-min-english-ratio $formatMinEnglishRatio `
    --format-max-unreadable-ratio $formatMaxUnreadableRatio `
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
    --model-name $modelName `
    --strict-tokenizer-model-match `
    --max-eval-queries 1000000 `
    --search-threads $searchThreads `
    --max-new-tokens $evalMaxNewTokens `
    --temperature $evalTemperature `
    --top-p $evalTopP `
    --query-batch-size $evalQueryBatchSize `
    --reward-mrr-k $rewardMrrK `
    --reward-recall-k $rewardRecallK `
    --reward-recall-dense-k $rewardRecallDenseK `
    --reward-w-mrr $rewardWMrr `
    --reward-w-recall $rewardWRecall `
    --reward-w-recall-dense $rewardWRecallDense `
    --reward-w-term-preserve $rewardWTermPreserve `
    --reward-w-length-score $rewardWLengthScore `
    --reward-w-clean-format $rewardWCleanFormat `
    --reward-w-bad-format $rewardWBadFormat `
    --reward-w-unsafe-copy $rewardWUnsafeCopy `
    --format-max-tokens $formatMaxTokens `
    --format-min-english-ratio $formatMinEnglishRatio `
    --format-max-unreadable-ratio $formatMaxUnreadableRatio `
    --sample-print 20 `
    --report-path $evalReportPath

  Write-Host "Done. Report: $evalReportPath"

  Write-Host "[3/3] Auto-committing all changes to git repository..."
  if ($autoGitCommit -ne "1") {
    Write-Host "[3/3] AUTO_GIT_COMMIT=0, skipped."
  }
  else {
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
      $commitTimestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
      $commitMessage = "chore: auto commit training and full eval artifacts $commitTimestamp"
      git commit -m $commitMessage
      if ($autoGitPush -eq "1") {
        git push
        if ($LASTEXITCODE -eq 0) {
          Write-Host "[3/3] Commit created and pushed: $commitMessage"
        }
        else {
          Write-Warning "Commit created, but git push failed. Please push manually."
        }
      }
      else {
        Write-Host "[3/3] Commit created locally (AUTO_GIT_PUSH=0): $commitMessage"
      }
    }
  }
}
finally {
  if ($transcriptStarted) {
    Stop-Transcript | Out-Null
  }
  Pop-Location
}
