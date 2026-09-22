<##
.SYNOPSIS
  Stage-aware, non-overwriting Table-3 Delay-Only repeat-1 launcher.
.EXAMPLE
  .\table3_delay_only_trueppo_20260922_repeat1.ps1 -Stage calibration -DryRun
  .\table3_delay_only_trueppo_20260922_repeat1.ps1 -Stage train
#>
[CmdletBinding()]
param(
    [ValidateSet('calibration', 'train')]
    [string]$Stage,
    [switch]$DryRun
)

# Lazy, stage-aware launcher for the only physical Table-3 Delay-Only row.
# No Stage (or -DryRun) is always inert.  All mutable target/log paths are in
# the main checkout; code is executed from the pinned, clean D:\w\t3delay tree.
# The superseded launcher (SHA 9047a8c5...) is retained as evidence.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$WorktreeRoot = 'D:\w\t3delay\chapter2_region_detection'
$Python = Join-Path $RepoRoot '.venv_cc4\Scripts\python.exe'
$Runner = Join-Path $WorktreeRoot 'formal_experiments\ours\run_table23_true_ppo.py'
$Target = Join-Path $RepoRoot 'outputs\formal_v3\table3\delay_only_trueppo_20260922_repeat1'
$LogDir = Join-Path $RepoRoot 'outputs\launcher_logs'
$ExpectedHead = 'a3c09bcdcb69bd2eda5dd40a5e7365a528bee705'
$ExpectedRunnerSha = 'ec2de89ebb2e0894df4cf7d4194afdb502503e9f94bf121915cf114ed6282a6e'
$ExpectedCalibrationSha = '0543411324c483247e599ea95bfc3cbef2a0396490dec4b0f013a70a850b121e'
$ExpectedCalibrationManifestSha = 'fe0a7eb95b1fc5ff02775b85b05135f3be8555118880d68a1d0dd1db14c356d9'
$ExpectedNormalizerSha = 'e50df37c17cf7c49d2c3127800144a948a1a709c5a8396e773d2fb71f8765025'
$ExpectedRewardSha = '192efbe6512fbd83a746d86d759ebe5e68b1437ee78e570747a119d74b0e683f'
$ExpectedRewardManifestSha = 'fb9fdd64a6f576d0ffb444daacce07e891ba5505da66381b57acb52631c71eaf'
$Mandatory = Join-Path $RepoRoot 'docs\USER_MANDATORY_REQUIREMENTS.md'
$Taskbook = Join-Path (Split-Path $RepoRoot -Parent) 'CC4_V3_0917_FINAL_EXPERIMENT_TASKBOOK.md'
$ReplayManifest = Join-Path $RepoRoot 'docs\FINAL_REWARD_REPLAY_MANIFEST.json'
$ModelManifest = Join-Path $RepoRoot 'docs\FINAL_REWARD_MODEL_MANIFEST.json'
$PriorArtifact = Join-Path $RepoRoot 'outputs\priorrl_cc4\prototypes\frozen_prototypes.json'
$CacheManifest = Join-Path $RepoRoot 'outputs\lwm_rl_final_20260917\prior_cache_final\manifest.json'
$CacheBinding = Join-Path $RepoRoot 'outputs\lwm_rl_final_20260917\prior_cache_final\manifest.binding.json'
$WorldModel = Join-Path $RepoRoot 'outputs\world_model_final_20260917\a4_5b\world_model_absolute.pt'
$RewardRoot = Join-Path $RepoRoot 'outputs\formal_v3\table3_reward_models\delay_only'
$RewardModel = Join-Path $RewardRoot 'response_reward_predictor.pt'
$RewardManifest = Join-Path $RewardRoot 'frozen_manifest.json'
$ReplayTrain = Join-Path $RepoRoot 'outputs\formal_replay_final_20260917\train.jsonl'
$ReplayValidation = Join-Path $RepoRoot 'outputs\formal_replay_final_20260917\validation.jsonl'
$CalibrationJsonl = Join-Path $RepoRoot 'outputs\formal_v3\dependencies\trueppo_calibration_3000_3007_500\calibration_states.jsonl'
$CalibrationManifest = Join-Path $RepoRoot 'outputs\formal_v3\dependencies\trueppo_calibration_3000_3007_500\fitted\calibration_manifest.json'
$CalibrationStdout = Join-Path $LogDir 'table3_delay_only_trueppo_20260922_repeat1_calibration.stdout.log'
$CalibrationStderr = Join-Path $LogDir 'table3_delay_only_trueppo_20260922_repeat1_calibration.stderr.log'
$TrainStdout = Join-Path $LogDir 'table3_delay_only_trueppo_20260922_repeat1_train.stdout.log'
$TrainStderr = Join-Path $LogDir 'table3_delay_only_trueppo_20260922_repeat1_train.stderr.log'

function Sha([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function RequireSha([string]$Path, [string]$Expected, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "BLOCKED: missing ${Label}: $Path" }
    $actual = Sha $Path
    if ($actual -ne $Expected) { throw "BLOCKED: $Label SHA mismatch: $actual (expected $Expected)" }
}
function RequireCleanWorktree {
    if ((git -C $WorktreeRoot rev-parse HEAD).Trim() -ne $ExpectedHead) { throw 'BLOCKED: pinned Delay-Only worktree HEAD drift' }
    $dirty = @(git -C $WorktreeRoot status --porcelain=v1)
    if ($dirty.Count -ne 0) { throw "BLOCKED: pinned Delay-Only worktree is dirty: $($dirty -join ' | ')" }
}
function RequireIdentity {
    RequireCleanWorktree
    RequireSha $Runner $ExpectedRunnerSha 'true-PPO runner'
    RequireSha $Mandatory '5450bc5d692f4441497737f291121923693d15f6a27f92b5f55d0c24d91026a8' 'mandatory requirements'
    RequireSha $Taskbook '6d862e9475c92a668358f69d9e09e9b32b7d23cd83679dcd4bd0d3847eae00a5' 'taskbook'
    RequireSha $ReplayManifest 'b7aae1a9189afad9717be7ebfff97ef2016bb25634ae8e8c83c254987653f8ee' 'replay manifest'
    RequireSha $ModelManifest '0a8a2c5c196dee74eda94cca24e4904cf93badf9008bf6ddcc27e6b854503931' 'model manifest'
    RequireSha $PriorArtifact 'e4359df9b6703a767ebe0e52ad09d2b208b8ddd19bd11e3820a887a249c96d09' 'frozen prior'
    RequireSha $CacheManifest '64f374b15c8cc88f463ac8d3e25eda451eed13694302abab20e1f5c61bbba40e' 'prior cache manifest'
    RequireSha $CacheBinding 'e470cd41441ae5f022e426f5f1ddb9bc58e131915fac4d2566d7fb6025e35fab' 'prior cache binding'
    RequireSha $WorldModel '3b86593aa8adda3e0e173bfb700c543bd88641d23cf3e73ddc3992284c8f900f' 'world model'
    RequireSha $RewardModel $ExpectedRewardSha 'Delay-Only predictor'
    RequireSha $RewardManifest $ExpectedRewardManifestSha 'Delay-Only frozen manifest'
    RequireSha $CalibrationJsonl $ExpectedCalibrationSha 'calibration JSONL'
    RequireSha $CalibrationManifest $ExpectedCalibrationManifestSha 'calibration manifest'
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "BLOCKED: missing Python: $Python" }
    $reward = Get-Content -LiteralPath $RewardManifest -Raw | ConvertFrom-Json
    if ($reward.mode -ne 'Delay-Only' -or $reward.frozen -ne $true -or $reward.quality_gate.pass -ne $true -or
        $reward.training_loss.name -ne 'mse' -or $reward.training_loss.sample_weighting -ne 'none' -or
        $reward.training_loss.resampling -ne 'none' -or $reward.test_seeds_used -ne $false) { throw 'BLOCKED: Delay-Only frozen MSE contract drift' }
}
function RequireLogFresh([string]$Path) { if (Test-Path -LiteralPath $Path) { throw "BLOCKED: refusing to overwrite launcher log: $Path" } }
function RequireFreshCalibrationTarget {
    if (Test-Path -LiteralPath $Target) { throw "BLOCKED: calibration requires a fresh target: $Target" }
}
function GetTargetWriters {
    $needle = [regex]::Escape($Target)
    @(Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and $_.CommandLine -match $needle -and $_.Name -match '^(python|pwsh|powershell)(\.exe)?$'
    })
}
function RequireUniqueTargetWriter {
    $writers = @(GetTargetWriters)
    if ($writers.Count -ne 0) { throw "BLOCKED: target already has writer(s): $($writers.ProcessId -join ',')" }
    if (Test-Path -LiteralPath $Target) {
        $lockPath = Join-Path $Target '.true_ppo_writer.lock.json'
        if (Test-Path -LiteralPath $lockPath) {
            $lock = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json
            if ($lock.status -eq 'ACTIVE') {
                try { Get-Process -Id ([int]$lock.pid) -ErrorAction Stop | Out-Null; throw "BLOCKED: target lock is ACTIVE (pid=$($lock.pid))" } catch [Microsoft.PowerShell.Commands.ProcessCommandException] { }
            }
        }
    }
}
function RequireTrainCalibration {
    if (-not (Test-Path -LiteralPath $Target -PathType Container)) { throw "BLOCKED: train requires completed calibration target: $Target" }
    $items = @(Get-ChildItem -LiteralPath $Target -Force)
    $allowed = @('.true_ppo_writer.lock.json', 'normalizer.json', 'calibration_manifest.json')
    $unexpected = @($items | Where-Object { $allowed -notcontains $_.Name })
    if ($unexpected.Count -ne 0 -or @($items | Where-Object { $_.PSIsContainer }).Count -ne 0 -or $items.Count -ne 3) {
        throw "BLOCKED: train target must contain only frozen calibration plus RELEASED lock: $($items.Name -join ',')"
    }
    $normalizerPath = Join-Path $Target 'normalizer.json'; $manifestPath = Join-Path $Target 'calibration_manifest.json'
    $normalizer = Get-Content -LiteralPath $normalizerPath -Raw | ConvertFrom-Json
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ((Sha $normalizerPath) -ne $ExpectedNormalizerSha -or $manifest.status -ne 'PASS' -or
        $manifest.schema -ne 'cc4_v3_calibration_manifest_v1' -or $manifest.sample_count -ne 4000 -or
        $manifest.episode_ticks -ne 500 -or $manifest.split -ne 'calibration' -or
        ($manifest.calibration_seeds -join ',') -ne '3000,3001,3002,3003,3004,3005,3006,3007' -or
        $manifest.input_sha256 -ne $ExpectedCalibrationSha -or $manifest.normalizer_file_sha256 -ne (Sha $normalizerPath) -or
        $manifest.provider_calls -ne 0 -or $manifest.policy_updated -ne $false -or $manifest.blue_visible_only -ne $true -or
        $normalizer.fit_split -ne 'calibration' -or $normalizer.input_sha256 -ne $ExpectedCalibrationSha -or
        $normalizer.test_leakage -ne $false -or $normalizer.policy_updated -ne $false) { throw 'BLOCKED: frozen calibration identity/provenance failed' }
    $lock = Get-Content -LiteralPath (Join-Path $Target '.true_ppo_writer.lock.json') -Raw | ConvertFrom-Json
    if ($lock.status -ne 'RELEASED' -or $lock.stage -ne 'calibration' -or $lock.variant -ne 'lwm_rl' -or
        $lock.reward_mode -ne 'Delay-Only' -or [int]$lock.policy_seed -ne 51001) { throw 'BLOCKED: train requires RELEASED Delay-Only calibration lock' }
}
function GetResourceGate {
    $os = Get-CimInstance Win32_OperatingSystem
    $ram = [math]::Round([double]$os.FreePhysicalMemory / 1KB, 0)
    $cpu = [math]::Round([double](Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue, 1)
    $rows = @(& nvidia-smi --query-gpu=memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits 2>$null)
    if ($rows.Count -eq 0) { throw 'BLOCKED: nvidia-smi/GPU resource query unavailable' }
    $gpus = @($rows | ForEach-Object { $p = $_ -split ','; [pscustomobject]@{free_mib=[double]$p[0].Trim(); total_mib=[double]$p[1].Trim(); util_pct=[double]$p[2].Trim()} })
    $minFree = [math]::Round(($gpus | Measure-Object free_mib -Minimum).Minimum, 0)
    $maxUtil = [math]::Round(($gpus | Measure-Object util_pct -Maximum).Maximum, 1)
    [pscustomobject]@{ pass=($ram -ge 4096 -and $cpu -lt 85 -and $minFree -ge 3072 -and $maxUtil -lt 90); free_ram_mib=$ram; cpu_pct=$cpu; min_gpu_free_mib=$minFree; max_gpu_util_pct=$maxUtil; thresholds='RAM>=4096MiB; CPU<85%; GPU free>=3072MiB; GPU util<90%' }
}
function RequireStartGates {
    RequireUniqueTargetWriter
    $r = GetResourceGate
    if (-not $r.pass) { throw "BLOCKED: resource gate failed: $($r | ConvertTo-Json -Compress)" }
    return $r
}
function InvokeRunner([string]$RunStage, [string]$Stdout, [string]$Stderr) {
    RequireLogFresh $Stdout; RequireLogFresh $Stderr
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $a = @('-B','-m','formal_experiments.ours.run_table23_true_ppo','--project-root',$WorktreeRoot,'--mode','formal','--stage',$RunStage,'--variant','lwm_rl','--reward-mode','Delay-Only','--policy-seed','51001','--prior-artifact',$PriorArtifact,'--prior-cache-manifest',$CacheManifest,'--prior-cache-binding',$CacheBinding,'--world-model',$WorldModel,'--reward-model',$RewardModel,'--reward-manifest',$RewardManifest,'--replay-manifest',$ReplayManifest,'--replay-train-jsonl',$ReplayTrain,'--replay-validation-jsonl',$ReplayValidation,'--model-manifest',$ModelManifest,'--table-id','table3','--row-id','delay_only','--source','final_reward_20260917','--canonical-run-id','delay_only_trueppo_20260922_repeat1','--device','cuda','--out',$Target)
    if ($RunStage -eq 'calibration') { $a += @('--calibration-jsonl',$CalibrationJsonl,'--calibration-manifest',$CalibrationManifest) }
    $p = Start-Process -FilePath $Python -ArgumentList $a -WorkingDirectory $WorktreeRoot -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru -Wait
    if ($p.ExitCode -ne 0) { throw "BLOCKED: runner $RunStage exit=$($p.ExitCode); logs retained: $Stdout / $Stderr" }
    return [pscustomobject]@{ stage=$RunStage; exit_code=$p.ExitCode; stdout=$Stdout; stderr=$Stderr }
}
$plan = [ordered]@{ launcher='table3_delay_only_trueppo_20260922_repeat1'; row='Delay-Only'; policy_seed=51001; train_seeds='1000..1031'; validation_seeds='2000..2007'; test_seeds='4000..4099'; episode_ticks=500; K=6; H=4; target=$Target; worktree=$WorktreeRoot; head=$ExpectedHead; runner_sha256=$ExpectedRunnerSha; calibration_command="$($MyInvocation.MyCommand.Path) -Stage calibration"; train_command="$($MyInvocation.MyCommand.Path) -Stage train"; note='No Stage or -DryRun never starts a runner; validation/test are intentionally outside this launcher.' }
if ([string]::IsNullOrWhiteSpace($Stage)) { $plan.status='INERT_PLAN'; $plan | ConvertTo-Json -Depth 5; exit 0 }
try {
    RequireIdentity
    $trainTargetReady = $false; $trainTargetBlock = $null
    if ($Stage -eq 'calibration') { RequireFreshCalibrationTarget }
    elseif ($DryRun) {
        if (Test-Path -LiteralPath $Target) {
            try { RequireTrainCalibration; $trainTargetReady = $true } catch { $trainTargetBlock = $_.Exception.Message }
        } else { $trainTargetBlock = 'calibration target is absent (expected before calibration)' }
    } else { RequireTrainCalibration }
    $stdout = if ($Stage -eq 'calibration') { $CalibrationStdout } else { $TrainStdout }
    $stderr = if ($Stage -eq 'calibration') { $CalibrationStderr } else { $TrainStderr }
    RequireLogFresh $stdout; RequireLogFresh $stderr
    $resource = GetResourceGate
    if ($DryRun) { $plan.status='DRY_RUN_PASS'; $plan.stage=$Stage; $plan.start_allowed=($resource.pass -and ($Stage -eq 'calibration' -or $trainTargetReady)); $plan.train_target_ready=$trainTargetReady; $plan.train_target_gate=$trainTargetBlock; $plan.resource_gate=$resource; $plan | ConvertTo-Json -Depth 8; exit 0 }
    if (-not $resource.pass) { throw "BLOCKED: resource gate failed: $($resource | ConvertTo-Json -Compress)" }
    $resource = RequireStartGates
    $run = if ($Stage -eq 'calibration') { InvokeRunner 'calibration' $CalibrationStdout $CalibrationStderr } else { InvokeRunner 'train' $TrainStdout $TrainStderr }
    if ($Stage -eq 'calibration') { RequireTrainCalibration }
    $plan.status='PASS'; $plan.stage=$Stage; $plan.resource_gate=$resource; $plan.run=$run; $plan | ConvertTo-Json -Depth 8; exit 0
} catch { Write-Error $_; exit 2 }
