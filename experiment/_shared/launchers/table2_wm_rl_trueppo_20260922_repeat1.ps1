<#
Table 2 WM-RL repeat_1 lazy calibration/train launcher.

Default mode is inert.  -DryRun performs read-only gates and never creates the
formal target or launcher log directory.  -Execute runs exactly one stage;
calibration must finish and release its runner lock before train is allowed.
WM-RL deliberately binds only the frozen WM/Full-Reward predictor and uses the
audited deterministic non-LLM validity-aware K6/H4 generator.  No prior/cache
or provider input is passed to the runner.
#>
[CmdletBinding()]
param(
    [ValidateSet('calibration', 'train')]
    [string]$Stage = 'calibration',
    [switch]$Execute,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$RepositoryRoot = 'D:\paper\github-me\aptdetect'
$MainProjectRoot = Join-Path $RepositoryRoot 'chapter2_region_detection'
$Worktree = 'D:\w\t2wmready'
$ProjectRoot = Join-Path $Worktree 'chapter2_region_detection'
$Python = Join-Path $MainProjectRoot '.venv_cc4\Scripts\python.exe'
$Runner = Join-Path $ProjectRoot 'formal_experiments\ours\run_table23_true_ppo.py'
$Runtime = Join-Path $ProjectRoot 'formal_experiments\ours\formal_ppo_runtime.py'
$Target = Join-Path $MainProjectRoot 'outputs\formal_v3\table2\wm_rl_trueppo_20260922_repeat1\repeat_1'
$LogRoot = Join-Path $MainProjectRoot 'outputs\launcher_logs\table2_wm_rl_trueppo_20260922_repeat1'
$Lock = Join-Path $Target '.true_ppo_writer.lock.json'
$Mandatory = Join-Path $MainProjectRoot 'docs\USER_MANDATORY_REQUIREMENTS.md'
$Taskbook = Join-Path $RepositoryRoot 'CC4_V3_0917_FINAL_EXPERIMENT_TASKBOOK.md'
$ReplayManifest = Join-Path $MainProjectRoot 'docs\FINAL_REWARD_REPLAY_MANIFEST.json'
$ReplayTrain = Join-Path $MainProjectRoot 'outputs\formal_replay_final_20260917\train.jsonl'
$ReplayValidation = Join-Path $MainProjectRoot 'outputs\formal_replay_final_20260917\validation.jsonl'
$ModelManifest = Join-Path $MainProjectRoot 'docs\FINAL_REWARD_MODEL_MANIFEST.json'
$Prior = Join-Path $MainProjectRoot 'outputs\priorrl_cc4\prototypes\frozen_prototypes.json'
$PriorCacheManifest = Join-Path $MainProjectRoot 'outputs\lwm_rl_final_20260917\prior_cache_final\manifest.json'
$PriorCacheBinding = Join-Path $MainProjectRoot 'outputs\lwm_rl_final_20260917\prior_cache_final\manifest.binding.json'
$WorldModel = Join-Path $MainProjectRoot 'outputs\world_model_final_20260917\a4_5b\world_model_absolute.pt'
$RewardModel = Join-Path $MainProjectRoot 'outputs\world_model_final_20260917\a4_5c\response_reward_predictor.pt'
$CalibrationRaw = Join-Path $MainProjectRoot 'outputs\formal_v3\dependencies\trueppo_calibration_3000_3007_500\calibration_states.jsonl'
$CalibrationInputManifest = Join-Path $MainProjectRoot 'outputs\formal_v3\dependencies\trueppo_calibration_3000_3007_500\fitted\calibration_manifest.json'

$ExpectedHead = '3009d7d9a7fa215f29d817b7a47a3b64666ec18c'
$ExpectedRunnerSha = 'ec2de89ebb2e0894df4cf7d4194afdb502503e9f94bf121915cf114ed6282a6e'
$ExpectedSha = [ordered]@{
    Mandatory = '5450bc5d692f4441497737f291121923693d15f6a27f92b5f55d0c24d91026a8'
    Taskbook = '6d862e9475c92a668358f69d9e09e9b32b7d23cd83679dcd4bd0d3847eae00a5'
    ReplayManifest = 'b7aae1a9189afad9717be7ebfff97ef2016bb25634ae8e8c83c254987653f8ee'
    ReplayTrain = 'ba608ed7bd63d6a6c28a259738ca35e6122bc2f0a6cf7f427df1f2b3252c8cee'
    ReplayValidation = '7e23294678167078704c9603932b014d893fbd35e06b5a145259aef16c23588d'
    ModelManifest = '0a8a2c5c196dee74eda94cca24e4904cf93badf9008bf6ddcc27e6b854503931'
    Prior = 'e4359df9b6703a767ebe0e52ad09d2b208b8ddd19bd11e3820a887a249c96d09'
    PriorCacheManifest = '64f374b15c8cc88f463ac8d3e25eda451eed13694302abab20e1f5c61bbba40e'
    PriorCacheBinding = 'e470cd41441ae5f022e426f5f1ddb9bc58e131915fac4d2566d7fb6025e35fab'
    WorldModel = '3b86593aa8adda3e0e173bfb700c543bd88641d23cf3e73ddc3992284c8f900f'
    RewardModel = 'f333330510b5de8e78fb9e2dc4287dd87fee29695fe38cbec5e60700ee615624'
    CalibrationRaw = '0543411324c483247e599ea95bfc3cbef2a0396490dec4b0f013a70a850b121e'
    CalibrationInputManifest = 'fe0a7eb95b1fc5ff02775b85b05135f3be8555118880d68a1d0dd1db14c356d9'
}
$ExpectedNormalizerSha = 'e50df37c17cf7c49d2c3127800144a948a1a709c5a8396e773d2fb71f8765025'
$ExpectedCalibrationInputSha = $ExpectedSha.CalibrationInputManifest
$Variant = 'wm_rl'; $RowId = 'wm_rl'
$LauncherPath = $PSCommandPath

function Require-File([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "BLOCKED: missing ${Label}: $Path" }
}
function Sha([string]$Path) {
    Require-File $Path 'SHA-bound file'
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Require-Sha([string]$Path, [string]$Expected, [string]$Label) {
    $actual = Sha $Path
    if ($actual -ne $Expected.ToLowerInvariant()) { throw "BLOCKED: $Label SHA drift: $actual" }
    return $actual
}
function Require-Contains([string]$Path, [string]$Needle, [string]$Label) {
    Require-File $Path $Label
    if (-not (Get-Content -Raw -LiteralPath $Path).Contains($Needle)) {
        throw "BLOCKED: $Label lacks required contract: $Needle"
    }
}
function Assert-WorktreeSnapshot {
    $state = (git -C $Worktree status --porcelain=v1 --untracked-files=all | Out-String).Trim()
    if ($state) { throw "BLOCKED: ready worktree has uncommitted state: $state" }
    if ((git -C $Worktree rev-parse HEAD).Trim() -ne $ExpectedHead) { throw 'BLOCKED: WM-RL worktree HEAD drift' }
}
function Assert-ResourceGate {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop
    $freeRamMiB = [double]$os.FreePhysicalMemory / 1024.0
    $cpu = [double](Get-Counter '\Processor(_Total)\% Processor Time' -ErrorAction Stop).CounterSamples.CookedValue
    if ($freeRamMiB -lt 4096.0) { throw "BLOCKED: free RAM ${freeRamMiB}MiB < 4096MiB" }
    if ($cpu -ge 85.0) { throw "BLOCKED: CPU ${cpu}% >= 85%" }
    $rows = @(& nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits 2>$null)
    if ($rows.Count -lt 1) { throw 'BLOCKED: nvidia-smi GPU resource evidence unavailable' }
    $parts = $rows[0].ToString().Split(',') | ForEach-Object { $_.Trim() }
    if ($parts.Count -lt 3) { throw 'BLOCKED: malformed nvidia-smi GPU resource evidence' }
    $gpuUtil = [double]($parts[1] -replace '[^0-9.]', '')
    $gpuFreeMiB = [double]($parts[2] -replace '[^0-9.]', '')
    if ($gpuFreeMiB -lt 3072.0 -or $gpuUtil -ge 90.0) {
        throw "BLOCKED: GPU gate failed (free=${gpuFreeMiB}MiB, util=${gpuUtil}%)"
    }
    return [ordered]@{ free_ram_mib = [math]::Round($freeRamMiB, 2); cpu_percent = [math]::Round($cpu, 2); gpu_free_mib = $gpuFreeMiB; gpu_util_percent = $gpuUtil }
}
function Assert-NoTargetWriter {
    $token = [IO.Path]::GetFullPath($Target).TrimEnd('\')
    $writers = @(Get-CimInstance -ClassName Win32_Process -ErrorAction Stop | Where-Object {
        $cmd = [string]$_.CommandLine
        ($cmd.IndexOf($token, [StringComparison]::OrdinalIgnoreCase) -ge 0) -or
        ($cmd -match '(?i)run_table23_true_ppo' -and $cmd -match '(?i)--variant\s+wm_rl' -and $cmd -match '(?i)--row-id\s+wm_rl')
    })
    if ($writers.Count -gt 0) {
        $details = ($writers | ForEach-Object { "PID=$($_.ProcessId) CMD=$($_.CommandLine)" }) -join ' | '
        throw "BLOCKED: target writer already observed: $details"
    }
}
function Assert-FrozenAssets {
    Require-File $Python 'CC4 Python runtime'
    Require-Sha $Runner $ExpectedRunnerSha 'true-PPO runner' | Out-Null
    Assert-WorktreeSnapshot
    Require-Sha $Mandatory $ExpectedSha.Mandatory 'mandatory requirements' | Out-Null
    Require-Sha $Taskbook $ExpectedSha.Taskbook 'taskbook' | Out-Null
    Require-Sha $ReplayManifest $ExpectedSha.ReplayManifest 'replay manifest' | Out-Null
    Require-Sha $ReplayTrain $ExpectedSha.ReplayTrain 'train replay' | Out-Null
    Require-Sha $ReplayValidation $ExpectedSha.ReplayValidation 'validation replay' | Out-Null
    Require-Sha $ModelManifest $ExpectedSha.ModelManifest 'reward model manifest' | Out-Null
    Require-Sha $Prior $ExpectedSha.Prior 'prior audit asset (not loaded by WM-RL)' | Out-Null
    Require-Sha $PriorCacheManifest $ExpectedSha.PriorCacheManifest 'prior cache audit asset (not loaded by WM-RL)' | Out-Null
    Require-Sha $PriorCacheBinding $ExpectedSha.PriorCacheBinding 'prior cache binding audit asset (not loaded by WM-RL)' | Out-Null
    Require-Sha $WorldModel $ExpectedSha.WorldModel 'absolute world model' | Out-Null
    Require-Sha $RewardModel $ExpectedSha.RewardModel 'Full-Reward predictor' | Out-Null
    Require-Sha $CalibrationRaw $ExpectedSha.CalibrationRaw 'calibration raw states' | Out-Null
    Require-Sha $CalibrationInputManifest $ExpectedSha.CalibrationInputManifest 'fitted calibration input manifest' | Out-Null
    $cal = Get-Content -Raw -LiteralPath $CalibrationInputManifest | ConvertFrom-Json
    if ($cal.status -ne 'PASS' -or $cal.split -ne 'calibration' -or $cal.episode_ticks -ne 500 -or
        $cal.provider_calls -ne 0 -or $cal.policy_updated -ne $false -or $cal.raw_jsonl_sha256 -ne $ExpectedSha.CalibrationRaw) {
        throw 'BLOCKED: fitted calibration input manifest is not PASS/500/provider0/read-only'
    }
    Require-Contains $Runtime 'FormalPPOVariant.WM_RL, False, True, 6, "deterministic_non_llm_K6_H4"' 'WM-RL variant identity'
    Require-Contains $Runtime 'non_llm_candidate_plans(state, agent_name=agent_name)' 'fixed non-LLM candidate generator'
    Require-Contains $Runtime 'self.spec.uses_world_model' 'world-model decision gate'
    Require-Contains $Runtime 'world_model_evaluator.evaluate(state, plans, projection_context=projection_context)' 'frozen WM rollout'
}
function Get-ContractEvidence([object]$Resources) {
    return [ordered]@{
        schema = 'cc4_v3_table2_wm_rl_launcher_plan_v1'; status = 'PASS'; stage = $Stage
        variant = $Variant; table_id = 'table2'; row_id = $RowId; repeat_index = 1; policy_seed = 51001
        reward_mode = 'Full-Reward'; reward_channel = 'real_response_reward'; candidate_rollout_channel = 'Full-Reward_predicted_return'
        candidate_semantics = 'deterministic_non_llm_K6_H4'; candidate_generator = 'non_llm_candidate_plans'; candidate_count = 6; horizon = 4
        uses_llm_prior = $false; uses_world_model = $true; llm_or_online_provider_in_candidate_generation = $false
        online_provider_allowed = $false; provider_calls = 0; cache_only = $false
        train_seeds = '1000..1031'; calibration_seeds = '3000..3007'; validation_seeds = '2000..2007'; test_seeds = '4000..4099'; episode_ticks = 500
        environment_interaction_steps = 16000; formal_result_eligible = $true
        worktree = $Worktree; git_head = $ExpectedHead; runner = $Runner; runner_sha256 = $ExpectedRunnerSha
        target = $Target; log_root = $LogRoot; resources = $Resources
        target_exists = (Test-Path -LiteralPath $Target); log_root_exists = (Test-Path -LiteralPath $LogRoot)
        prior_sha256_audit = $ExpectedSha.Prior; prior_cache_manifest_sha256_audit = $ExpectedSha.PriorCacheManifest; prior_cache_binding_sha256_audit = $ExpectedSha.PriorCacheBinding
        world_model_sha256 = $ExpectedSha.WorldModel; reward_model_sha256 = $ExpectedSha.RewardModel
        calibration_raw_sha256 = $ExpectedSha.CalibrationRaw; fitted_calibration_manifest_sha256 = $ExpectedCalibrationInputSha
        canonical_normalizer_sha256_required = $ExpectedNormalizerSha
    }
}
function Get-RunnerArgs([string]$RunStage) {
    $args = @('-B', '-m', 'formal_experiments.ours.run_table23_true_ppo', '--project-root', $ProjectRoot,
        '--mode', 'formal', '--stage', $RunStage, '--variant', $Variant, '--reward-mode', 'Full-Reward',
        '--policy-seed', '51001', '--world-model', $WorldModel, '--reward-model', $RewardModel,
        '--replay-manifest', $ReplayManifest, '--replay-train-jsonl', $ReplayTrain,
        '--replay-validation-jsonl', $ReplayValidation, '--model-manifest', $ModelManifest,
        '--table-id', 'table2', '--row-id', $RowId, '--source', 'final_reward_20260917',
        '--canonical-run-id', 'wm_rl_trueppo_20260922_repeat1', '--device', 'cuda', '--out', $Target)
    if ($RunStage -eq 'calibration') { $args += @('--calibration-jsonl', $CalibrationRaw, '--calibration-manifest', $CalibrationInputManifest) }
    return $args
}
function Assert-CliHelp {
    $pushed = $false
    try {
        Push-Location -LiteralPath $ProjectRoot; $pushed = $true
        & $Python -B -m formal_experiments.ours.run_table23_true_ppo --help *> $null
        if ($LASTEXITCODE -ne 0) { throw 'BLOCKED: true-PPO CLI help failed' }
    }
    finally { if ($pushed) { Pop-Location } }
}
function Assert-CalibrationTarget {
    Require-File $Lock 'writer lock evidence'
    $lockPayload = Get-Content -Raw -LiteralPath $Lock | ConvertFrom-Json
    if ($lockPayload.status -ne 'RELEASED') { throw "BLOCKED: calibration lock is not RELEASED ($($lockPayload.status))" }
    $normalizer = Join-Path $Target 'normalizer.json'; $manifest = Join-Path $Target 'calibration_manifest.json'
    Require-Sha $normalizer $ExpectedNormalizerSha 'v3 canonical normalizer' | Out-Null
    Require-File $manifest 'fitted calibration manifest'
    $payload = Get-Content -Raw -LiteralPath $manifest | ConvertFrom-Json
    if ($payload.status -ne 'PASS' -or $payload.split -ne 'calibration' -or $payload.episode_ticks -ne 500 -or
        $payload.provider_calls -ne 0 -or $payload.policy_updated -ne $false -or $payload.manifest_sha256 -ne $ExpectedCalibrationInputSha -or
        $payload.raw_jsonl_sha256 -ne $ExpectedSha.CalibrationRaw -or $payload.normalizer_file_sha256 -ne $ExpectedNormalizerSha -or
        $payload.normalizer.calibration_manifest_sha256 -ne $ExpectedCalibrationInputSha -or $payload.normalizer.policy_updated -ne $false) {
        throw 'BLOCKED: calibration target identity/provenance is not PASS or v3-canonical'
    }
}
function Invoke-Stage([string[]]$RunnerArgs, [object]$Resources) {
    if (-not (Test-Path -LiteralPath $LogRoot)) { New-Item -ItemType Directory -LiteralPath $LogRoot | Out-Null }
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
    $base = Join-Path $LogRoot ($Stage + '_' + $stamp)
    $stdout = $base + '.stdout.log'; $stderr = $base + '.stderr.log'; $pidPath = $base + '.pid'; $exitPath = $base + '.exitcode'; $planPath = $base + '.plan.json'
    $launcherSha = Sha $LauncherPath
    $plan = Get-ContractEvidence $Resources
    $plan.launcher = $LauncherPath; $plan.launcher_sha256 = $launcherSha; $plan.runner_args = $RunnerArgs; $plan.started_utc = [DateTime]::UtcNow.ToString('o')
    [IO.File]::WriteAllText($planPath, ($plan | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
    $child = Start-Process -FilePath $Python -ArgumentList $RunnerArgs -WorkingDirectory $ProjectRoot -NoNewWindow -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    [IO.File]::WriteAllText($pidPath, [string]$child.Id, [Text.UTF8Encoding]::new($false))
    $child.WaitForExit(); $exitCode = [int]$child.ExitCode
    [IO.File]::WriteAllText($exitPath, [string]$exitCode, [Text.UTF8Encoding]::new($false))
    if ($exitCode -ne 0) { throw "BLOCKED: $Stage exited $exitCode; see $stdout and $stderr" }
    if ($Stage -eq 'calibration') { Assert-CalibrationTarget }
    Write-Host ("PASS: $Stage completed; target=$Target; stdout=$stdout; runner_sha256=$ExpectedRunnerSha")
}

if ($Execute -and $DryRun) { throw 'BLOCKED: choose either -Execute or -DryRun' }
if (-not $Execute -and -not $DryRun) {
    Write-Host "INERT_PLAN: no stage executed. Use -DryRun -Stage $Stage for read-only gates or -Execute -Stage $Stage to run one stage."
    Write-Host "target=$Target"
    exit 0
}

Assert-FrozenAssets
Assert-NoTargetWriter
$targetExists = Test-Path -LiteralPath $Target -PathType Container
if ($Stage -eq 'train' -and -not $targetExists) {
    if (-not $DryRun) { throw "BLOCKED: train requires a completed calibration target: $Target" }
    $resources = [ordered]@{ skipped = 'target_absent'; free_ram_mib = $null; cpu_percent = $null; gpu_free_mib = $null; gpu_util_percent = $null }
    $plan = Get-ContractEvidence $resources; $plan.status = 'BLOCKED'; $plan.start_allowed = $false; $plan.reason = 'calibration target is absent'; $plan.would_write = $false; $plan.formal_target_created = $false; $plan.formal_log_created = $false
    $plan.runner_args = Get-RunnerArgs 'train'
    Write-Output ($plan | ConvertTo-Json -Depth 12)
    exit 0
}
$resources = Assert-ResourceGate
if ($Stage -eq 'calibration') {
    if ($targetExists) { throw "BLOCKED: calibration requires a fresh absent target: $Target" }
    if ($DryRun) {
        if (Test-Path -LiteralPath $LogRoot) { throw "BLOCKED: calibration log root already exists: $LogRoot" }
        $plan = Get-ContractEvidence $resources; $plan.start_allowed = $true; $plan.would_write = $false; $plan.formal_target_created = $false; $plan.formal_log_created = $false
        $plan.runner_args = Get-RunnerArgs 'calibration'; $plan.cli_help_checked = $true
        Assert-CliHelp
        Write-Output ($plan | ConvertTo-Json -Depth 12); exit 0
    }
    if (Test-Path -LiteralPath $LogRoot) { throw "BLOCKED: refusing pre-existing launcher log root: $LogRoot" }
    Invoke-Stage (Get-RunnerArgs 'calibration') $resources; exit 0
}

Assert-CalibrationTarget
if ($DryRun) {
    $plan = Get-ContractEvidence $resources; $plan.start_allowed = $true; $plan.would_write = $false; $plan.formal_target_created = $false; $plan.formal_log_created = $false
    $plan.runner_args = Get-RunnerArgs 'train'; $plan.cli_help_checked = $true
    Assert-CliHelp
    Write-Output ($plan | ConvertTo-Json -Depth 12); exit 0
}
Invoke-Stage (Get-RunnerArgs 'train') $resources
