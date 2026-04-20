Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $repoRoot

try {
  $pythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
  $artifactRoot = if ($env:ARTIFACT_ROOT) { $env:ARTIFACT_ROOT } else { "train_and_eval_data_model_0420" }
  $expName = if ($env:EXP_NAME) { $env:EXP_NAME } else { "4b_top20_delta_curriculum" }
  $logDir = if ($env:LOG_DIR) { $env:LOG_DIR } else { "log" }
  $modelName = if ($env:MODEL_NAME) { $env:MODEL_NAME } else { "Qwen/Qwen3-4B-Instruct-2507" }
  $searchThreads = if ($env:SEARCH_THREADS) { $env:SEARCH_THREADS } else { "16" }
  $trainMaxValQueries = if ($env:TRAIN_MAX_VAL_QUERIES) { $env:TRAIN_MAX_VAL_QUERIES } else { "400" }
  $evalQueryBatchSize = if ($env:EVAL_QUERY_BATCH_SIZE) { $env:EVAL_QUERY_BATCH_SIZE } else { "8" }
  $actorChunkSize = if ($env:ACTOR_CHUNK_SIZE) { $env:ACTOR_CHUNK_SIZE } else { "4" }
  $projectionChunkSize = if ($env:PROJECTION_CHUNK_SIZE) { $env:PROJECTION_CHUNK_SIZE } else { "64" }
  $groupTemperatureStride = if ($env:GROUP_TEMPERATURE_STRIDE) { $env:GROUP_TEMPERATURE_STRIDE } else { "0.07" }
  $groupTopPStride = if ($env:GROUP_TOP_P_STRIDE) { $env:GROUP_TOP_P_STRIDE } else { "0.015" }
  $minUniqueFinalQueries = if ($env:MIN_UNIQUE_FINAL_QUERIES) { $env:MIN_UNIQUE_FINAL_QUERIES } else { "4" }
  $maxRegenRounds = if ($env:MAX_REGEN_ROUNDS) { $env:MAX_REGEN_ROUNDS } else { "2" }
  $rewardGapThreshold = if ($env:REWARD_GAP_THRESHOLD) { $env:REWARD_GAP_THRESHOLD } else { "0.08" }
  $gapSamplingTemperatureDelta = if ($env:GAP_SAMPLING_TEMPERATURE_DELTA) { $env:GAP_SAMPLING_TEMPERATURE_DELTA } else { "0.15" }
  $formatMaxTokens = if ($env:FORMAT_MAX_TOKENS) { $env:FORMAT_MAX_TOKENS } else { "12" }
  $formatMinEnglishRatio = if ($env:FORMAT_MIN_ENGLISH_RATIO) { $env:FORMAT_MIN_ENGLISH_RATIO } else { "0.8" }
  $formatMaxUnreadableRatio = if ($env:FORMAT_MAX_UNREADABLE_RATIO) { $env:FORMAT_MAX_UNREADABLE_RATIO } else { "0.25" }

  $phase1Batch = if ($env:PHASE1_BATCH_SIZE) { $env:PHASE1_BATCH_SIZE } else { "24" }
  $phase1Group = if ($env:PHASE1_GROUP_SIZE) { $env:PHASE1_GROUP_SIZE } else { "8" }
  $phase1MaxGroup = if ($env:PHASE1_MAX_GROUP_SIZE) { $env:PHASE1_MAX_GROUP_SIZE } else { "12" }
  $phase1Lr = if ($env:PHASE1_LR) { $env:PHASE1_LR } else { "1.0e-5" }
  $phase1Kl = if ($env:PHASE1_KL_BETA) { $env:PHASE1_KL_BETA } else { "0.045" }
  $phase1Temp = if ($env:PHASE1_TEMPERATURE) { $env:PHASE1_TEMPERATURE } else { "0.75" }
  $phase1TopP = if ($env:PHASE1_TOP_P) { $env:PHASE1_TOP_P } else { "0.92" }
  $phase1MaxNewTokens = if ($env:PHASE1_MAX_NEW_TOKENS) { $env:PHASE1_MAX_NEW_TOKENS } else { "10" }
  $phase1EvalEvery = if ($env:PHASE1_EVAL_EVERY_STEPS) { $env:PHASE1_EVAL_EVERY_STEPS } else { "20" }
  $phase1MaxSteps = if ($env:PHASE1_MAX_STEPS) { $env:PHASE1_MAX_STEPS } else { "80" }

  $phase2Batch = if ($env:PHASE2_BATCH_SIZE) { $env:PHASE2_BATCH_SIZE } else { "24" }
  $phase2Group = if ($env:PHASE2_GROUP_SIZE) { $env:PHASE2_GROUP_SIZE } else { "8" }
  $phase2MaxGroup = if ($env:PHASE2_MAX_GROUP_SIZE) { $env:PHASE2_MAX_GROUP_SIZE } else { "12" }
  $phase2Lr = if ($env:PHASE2_LR) { $env:PHASE2_LR } else { "8e-6" }
  $phase2Kl = if ($env:PHASE2_KL_BETA) { $env:PHASE2_KL_BETA } else { "0.05" }
  $phase2Temp = if ($env:PHASE2_TEMPERATURE) { $env:PHASE2_TEMPERATURE } else { "0.70" }
  $phase2TopP = if ($env:PHASE2_TOP_P) { $env:PHASE2_TOP_P } else { "0.90" }
  $phase2MaxNewTokens = if ($env:PHASE2_MAX_NEW_TOKENS) { $env:PHASE2_MAX_NEW_TOKENS } else { "10" }
  $phase2EvalEvery = if ($env:PHASE2_EVAL_EVERY_STEPS) { $env:PHASE2_EVAL_EVERY_STEPS } else { "20" }
  $phase2MaxSteps = if ($env:PHASE2_MAX_STEPS) { $env:PHASE2_MAX_STEPS } else { "60" }

  $runRoot = Join-Path $artifactRoot "artifacts_${expName}"
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

  Write-Host "[env] python=$pythonBin model=$modelName search_threads=$searchThreads"
  Write-Host "[env] curriculum_metadata=$curriculumMetadataPath"

  $commonTrainArgs = @(
    "--model-name", $modelName,
    "--num-epochs", "1",
    "--clip-range", "0.2",
    "--ref-precision-mode", "4bit",
    "--search-threads", $searchThreads,
    "--eval-query-batch-size", $evalQueryBatchSize,
    "--group-temperature-stride", $groupTemperatureStride,
    "--group-top-p-stride", $groupTopPStride,
    "--min-unique-final-queries", $minUniqueFinalQueries,
    "--max-regen-rounds", $maxRegenRounds,
    "--reward-gap-threshold", $rewardGapThreshold,
    "--gap-sampling-temperature-delta", $gapSamplingTemperatureDelta,
    "--actor-chunk-size", $actorChunkSize,
    "--projection-chunk-size", $projectionChunkSize,
    "--max-val-queries", $trainMaxValQueries,
    "--reward-mode", "top20_delta",
    "--reward-mrr-k", "20",
    "--reward-recall-k", "20",
    "--reward-recall-dense-k", "50",
    "--reward-w-bad-format", "0.18",
    "--reward-w-unsafe-copy", "0.12",
    "--reward-w-overedit", "0.10",
    "--overedit-tau", "0.40",
    "--format-max-tokens", $formatMaxTokens,
    "--format-min-english-ratio", $formatMinEnglishRatio,
    "--format-max-unreadable-ratio", $formatMaxUnreadableRatio,
    "--curriculum-enable",
    "--curriculum-metadata-path", $curriculumMetadataPath
  )

  Write-Host "[phase1] batch=$phase1Batch group=$phase1Group max_group=$phase1MaxGroup lr=$phase1Lr kl=$phase1Kl decode=($phase1MaxNewTokens,$phase1Temp,$phase1TopP) reward=(0.55,0.20,0.15,0.10)"
  & $pythonBin train.py @commonTrainArgs `
    --curriculum-phase phase1 `
    --batch-size $phase1Batch `
    --group-size $phase1Group `
    --max-group-size $phase1MaxGroup `
    --learning-rate $phase1Lr `
    --kl-beta $phase1Kl `
    --max-new-tokens $phase1MaxNewTokens `
    --temperature $phase1Temp `
    --top-p $phase1TopP `
    --eval-max-new-tokens $phase1MaxNewTokens `
    --eval-temperature $phase1Temp `
    --eval-top-p $phase1TopP `
    --eval-every-steps $phase1EvalEvery `
    --max-steps $phase1MaxSteps `
    --reward-w-mrr 0.55 `
    --reward-w-recall 0.20 `
    --reward-w-recall-dense 0.15 `
    --reward-w-rank-bonus 0.10 `
    --save-dir $phase1CheckpointDir `
    --log-path $phase1LogPath `
    --group-trace-log-path $phase1TracePath

  $phase1Best = Join-Path $phase1CheckpointDir "best"
  if (-not (Test-Path -LiteralPath $phase1Best)) {
    throw "phase1 best checkpoint not found: $phase1Best"
  }

  Write-Host "[phase2] batch=$phase2Batch group=$phase2Group max_group=$phase2MaxGroup lr=$phase2Lr kl=$phase2Kl decode=($phase2MaxNewTokens,$phase2Temp,$phase2TopP) reward=(0.65,0.15,0.10,0.10)"
  & $pythonBin train.py @commonTrainArgs `
    --curriculum-phase phase2 `
    --adapter-path $phase1Best `
    --batch-size $phase2Batch `
    --group-size $phase2Group `
    --max-group-size $phase2MaxGroup `
    --learning-rate $phase2Lr `
    --kl-beta $phase2Kl `
    --max-new-tokens $phase2MaxNewTokens `
    --temperature $phase2Temp `
    --top-p $phase2TopP `
    --eval-max-new-tokens $phase2MaxNewTokens `
    --eval-temperature $phase2Temp `
    --eval-top-p $phase2TopP `
    --eval-every-steps $phase2EvalEvery `
    --max-steps $phase2MaxSteps `
    --reward-w-mrr 0.65 `
    --reward-w-recall 0.15 `
    --reward-w-recall-dense 0.10 `
    --reward-w-rank-bonus 0.10 `
    --save-dir $phase2CheckpointDir `
    --log-path $phase2LogPath `
    --group-trace-log-path $phase2TracePath

  $phase2Best = Join-Path $phase2CheckpointDir "best"
  if (-not (Test-Path -LiteralPath $phase2Best)) {
    throw "phase2 best checkpoint not found: $phase2Best"
  }

  Write-Host "[eval] adapter=$phase2Best query_batch_size=$evalQueryBatchSize"
  & $pythonBin eval_compare.py `
    --rl-adapter-path $phase2Best `
    --model-name $modelName `
    --strict-tokenizer-model-match `
    --max-eval-queries 1000000 `
    --search-threads $searchThreads `
    --max-new-tokens $phase2MaxNewTokens `
    --temperature $phase2Temp `
    --top-p $phase2TopP `
    --query-batch-size $evalQueryBatchSize `
    --reward-mode top20_delta `
    --reward-mrr-k 20 `
    --reward-recall-k 20 `
    --reward-recall-dense-k 50 `
    --reward-w-mrr 0.65 `
    --reward-w-recall 0.15 `
    --reward-w-recall-dense 0.10 `
    --reward-w-rank-bonus 0.10 `
    --reward-w-bad-format 0.18 `
    --reward-w-unsafe-copy 0.12 `
    --reward-w-overedit 0.10 `
    --overedit-tau 0.40 `
    --format-max-tokens $formatMaxTokens `
    --format-min-english-ratio $formatMinEnglishRatio `
    --format-max-unreadable-ratio $formatMaxUnreadableRatio `
    --sample-print 20 `
    --report-path $evalReportPath

  Write-Host "[done] report: $evalReportPath"
}
finally {
  Pop-Location
}
