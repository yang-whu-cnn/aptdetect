<##
.SYNOPSIS
  Fail-closed, stage-aware CARL-CC4 repeat-1 launcher.

.DESCRIPTION
  The default invocation is inert.  -DryRun performs read-only identity,
  resource, device and CLI-help checks and never creates a training target or
  launcher log.  -Execute -Stage train is the only mutating path prepared here;
  it runs the pinned CARL CLI on frozen train/validation replay and writes the
  training target and logs under the main checkout.  Validation selection is
  produced by that CLI.  Test consumes the taskbook-disclosed H=4 adaptation;
  repeat-1 remains PROVISIONAL and is not a five-repeat paper-row PASS.

  No command in this file deletes, resets, cleans, restores or overwrites an
  existing target/log.  The dedicated implementation worktree is D:\w\carl;
  the mutable output roots are in the main checkout.

.EXAMPLE
  .\table1_carl_cc4_20260922_repeat1.ps1
  .\table1_carl_cc4_20260922_repeat1.ps1 -Stage preflight -DryRun
  .\table1_carl_cc4_20260922_repeat1.ps1 -Stage train -DryRun
  .\table1_carl_cc4_20260922_repeat1.ps1 -Stage train -Execute
##>
[CmdletBinding()]
param(
    [ValidateSet('preflight', 'train', 'validation', 'test')]
    [string]$Stage,
    [switch]$Execute,
    [switch]$DryRun,
    [switch]$Resume
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepositoryRoot = 'D:\paper\github-me\aptdetect'
$MainProjectRoot = Join-Path $RepositoryRoot 'chapter2_region_detection'
$WorktreeRoot = 'D:\w\carl'
$WorktreeProject = Join-Path $WorktreeRoot 'chapter2_region_detection'
$Python = Join-Path $MainProjectRoot '.venv_cc4\Scripts\python.exe'
$Target = Join-Path $WorktreeProject 'outputs\formal_v3\training\carl_cc4\repeat_01'
$TestTarget = Join-Path $MainProjectRoot 'outputs\formal_v3\table1\carl_cc4\repeat_1_retry3_identity_fix'
$LogRoot = Join-Path $MainProjectRoot 'outputs\launcher_logs\table1_carl_cc4_20260922_repeat1_train_scientific_fix_2a0e53e9'
$TestLogRoot = Join-Path $MainProjectRoot 'outputs\launcher_logs\table1_carl_cc4_20260922_repeat1_test_identity_fix_265bdd0b'
$Runner = Join-Path $WorktreeProject 'baselines\carl_cc4\formal_cli.py'
$FormalTraining = Join-Path $WorktreeProject 'baselines\carl_cc4\formal_training.py'
$Buffer = Join-Path $WorktreeProject 'baselines\carl_cc4\buffers.py'
$Scm = Join-Path $WorktreeProject 'baselines\carl_cc4\scm.py'
$Reward = Join-Path $WorktreeProject 'baselines\carl_cc4\reward.py'
$Policy = Join-Path $WorktreeProject 'baselines\carl_cc4\policy.py'
$Augmentation = Join-Path $WorktreeProject 'baselines\carl_cc4\augmentation.py'
$Config = Join-Path $WorktreeProject 'configs\formal_v3\methods\carl_cc4.yaml'
$EvaluatorRepeat = Join-Path $WorktreeProject 'formal_experiments\evaluation\run_method_repeat.py'
$EvaluatorEpisode = Join-Path $WorktreeProject 'formal_experiments\evaluation\run_method_episode.py'
$ManifestValidator = Join-Path $WorktreeProject 'formal_experiments\common\run_manifest.py'
$ReplayManifest = Join-Path $MainProjectRoot 'docs\FINAL_REWARD_REPLAY_MANIFEST.json'
$ModelManifest = Join-Path $MainProjectRoot 'docs\FINAL_REWARD_MODEL_MANIFEST.json'
$Mandatory = Join-Path $MainProjectRoot 'docs\USER_MANDATORY_REQUIREMENTS.md'
$Taskbook = Join-Path $RepositoryRoot 'CC4_V3_0917_FINAL_EXPERIMENT_TASKBOOK.md'
$Handoff = Join-Path $MainProjectRoot 'docs\CHATGPT_HANDOFF_FINAL_EXPERIMENTS.md'
$TrainReplay = Join-Path $MainProjectRoot 'outputs\formal_replay_final_20260917\train.jsonl'
$ValidationReplay = Join-Path $MainProjectRoot 'outputs\formal_replay_final_20260917\validation.jsonl'
$WorldModel = Join-Path $MainProjectRoot 'outputs\world_model_final_20260917\a4_5b\world_model_absolute.pt'
$Paper = Join-Path $RepositoryRoot 'CARL.pdf'
$ExpectedHead = '265bdd0bd93990e1561c85fb6bfa9cb5ad07371c'
$Expected = [ordered]@{
    mandatory = '5450bc5d692f4441497737f291121923693d15f6a27f92b5f55d0c24d91026a8'
    taskbook = '6d862e9475c92a668358f69d9e09e9b32b7d23cd83679dcd4bd0d3847eae00a5'
    handoff = '9b8b146fc24c4ceefeb8b925439535bb3fc6b239828274403178cd557af2c75e'
    replay_manifest = 'b7aae1a9189afad9717be7ebfff97ef2016bb25634ae8e8c83c254987653f8ee'
    model_manifest = '0a8a2c5c196dee74eda94cca24e4904cf93badf9008bf6ddcc27e6b854503931'
    train_replay = 'ba608ed7bd63d6a6c28a259738ca35e6122bc2f0a6cf7f427df1f2b3252c8cee'
    validation_replay = '7e23294678167078704c9603932b014d893fbd35e06b5a145259aef16c23588d'
    world_model = '3b86593aa8adda3e0e173bfb700c543bd88641d23cf3e73ddc3992284c8f900f'
    paper = '42e8ef7d912007a2454002300852a54d1dfffad2d9b1099958f7c8d6a84cfae1'
    formal_cli = 'ad1b25c966eb70da2df35e34c8bef01e451cb490d9825533bfcdbd53c78adf96'
    formal_training = '656ebf1b4efb94feb25b43b7e7fd13990a9bb70a8035fd33e0d1249c8471ddcb'
    buffer = '73828f1e86b4b911a71461c2f2e708705e3ca1100caed862a3d260311cb6e7b7'
    scm = 'c67b83ff93fc1163acd267288cd3865fe9721de7332cb91bc9dcb71aa29a0862'
    reward = 'c32183ffeb8a678a13c9e1a162ec34bf23c1437a7c7ce12c256ffc65cfa1a4d0'
    policy = 'bfcbc1dd4516e1dec2db34ab39d146b47927c2e4569815bd5c530a0f09fecbf3'
    augmentation = '9755269f7fef5fe19d7069d0d8dca87c7098fd80e89ca55fcdef4526b3c9e104'
    config = '79ab790f02f596bfc2975f38f44facb8d05c1ca2b3d947df3feaa791a50c5120'
    evaluator_repeat = '84803fe8cf095938d486f301d4b9fd7ef30b4ea3b7ca32636bdd44284cd93b50'
    evaluator_episode = '6665973b06e4695aabfbfac267a42e9954f81ded20aeb03c3e45fb55cb0a75b6'
    manifest_validator = '12b301d390ac1f8e49112eefb21ebf80b14baba09ce2ab71152fefb05bf2e943'
    fresh_checkpoint = '87ac6842d2101b468d870766d0aa9b6ba922d41745a2841d4e449f427c26c066'
    fresh_selection = 'c53712271c0617666ce7d6648f503844b141fba3f24f2aff50566dc26cc41978'
    fresh_training_manifest = '2ff13c8f4fdb4717da16c7ecfab8136b415262a0d989d9358ce167b885baa3b9'
    fresh_training_code_commit = '911c531ad915a980be68275d5401cc2e1bf83f44'
}

function Require-File([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "BLOCKED: missing ${Label}: $Path" }
}
function Sha([string]$Path) {
    Require-File $Path 'SHA-bound artifact'
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Require-Sha([string]$Path, [string]$ExpectedSha, [string]$Label) {
    $actual = Sha $Path
    if ($actual -ne $ExpectedSha) { throw "BLOCKED: ${Label} SHA drift: ${actual} (expected ${ExpectedSha})" }
}
function Require-Worktree {
    Require-File $Runner 'CARL formal CLI'
    $head = (git -C $WorktreeRoot rev-parse HEAD).Trim()
    if ($head -ne $ExpectedHead) { throw "BLOCKED: CARL worktree HEAD drift: $head" }
    $status = @(git -C $WorktreeRoot status --porcelain=v1 --untracked-files=all)
    if ($status.Count -ne 0) { throw "BLOCKED: CARL worktree is dirty: $($status -join ' | ')" }
}
function Require-Identity {
    Require-Worktree
    Require-Sha $Mandatory $Expected.mandatory 'mandatory requirements'
    Require-Sha $Taskbook $Expected.taskbook 'taskbook'
    Require-Sha $Handoff $Expected.handoff 'final-experiment handoff'
    Require-Sha $ReplayManifest $Expected.replay_manifest 'replay manifest'
    Require-Sha $ModelManifest $Expected.model_manifest 'model manifest'
    Require-Sha $TrainReplay $Expected.train_replay 'train replay'
    Require-Sha $ValidationReplay $Expected.validation_replay 'validation replay'
    Require-Sha $WorldModel $Expected.world_model 'frozen world model'
    Require-Sha $Paper $Expected.paper 'CARL paper'
    Require-Sha $Runner $Expected.formal_cli 'CARL formal CLI'
    Require-Sha $FormalTraining $Expected.formal_training 'CARL formal training'
    Require-Sha $Buffer $Expected.buffer 'CARL real/synthetic buffer'
    Require-Sha $Scm $Expected.scm 'CARL SCM'
    Require-Sha $Reward $Expected.reward 'CARL reward mapping'
    Require-Sha $Policy $Expected.policy 'CARL policy'
    Require-Sha $Augmentation $Expected.augmentation 'CARL imagination gate'
    Require-Sha $Config $Expected.config 'CARL formal config'
    Require-Sha $EvaluatorRepeat $Expected.evaluator_repeat 'CARL formal repeat evaluator'
    Require-Sha $EvaluatorEpisode $Expected.evaluator_episode 'CARL formal episode evaluator'
    Require-Sha $ManifestValidator $Expected.manifest_validator 'formal manifest validator'
    Require-File $Python 'CC4 Python runtime'
}
function Require-StaticContract {
    $source = (Get-Content -Raw -LiteralPath $Runner).ToLowerInvariant()
    foreach ($needle in @('synthetic_rollouts_per_real_start', 'adapted_truncated', 'world_model_frozen', 'hidden_truth_policy_input', 'test_seeds_used', 'validation_evidence_schema', 'validation_evidence_manifest', 'candidate_schedule', 'carl_cc4_adapted_training_manifest_v3')) {
        if (-not $source.Contains($needle)) { throw "BLOCKED: CARL CLI lacks contract evidence: $needle" }
    }
    $bufferSource = (Get-Content -Raw -LiteralPath $Buffer).ToLowerInvariant()
    if (-not $bufferSource.Contains('synthetic_rollouts_per_real') -or -not $bufferSource.Contains('real_start_count')) {
        throw 'BLOCKED: CARL buffer lacks explicit 8:1 real-start ratio gate'
    }
    $scmSource = (Get-Content -Raw -LiteralPath $Scm).ToLowerInvariant()
    foreach ($needle in @('dag_parents', 'intervene_action')) {
        if (-not $scmSource.Contains($needle)) { throw "BLOCKED: CARL SCM lacks contract evidence: $needle" }
    }
    if (-not $source.Contains('hidden_truth_policy_input')) { throw 'BLOCKED: CARL checkpoint does not bind hidden-truth policy isolation' }
    $rewardSource = (Get-Content -Raw -LiteralPath $Reward).ToLowerInvariant()
    if (-not $rewardSource.Contains('cfg.beta_comp * int(incident_change)')) {
        throw 'BLOCKED: CAICS reward mapping lacks signed beta_comp compromise-change term'
    }
    $config = Get-Content -Raw -LiteralPath $Config
    foreach ($needle in @('synthetic_rollouts_per_real_rollout: 8', 'effective_imagination_horizon: 4', 'formal_result_eligible: true', 'paper_row_eligible: false', 'repeat_status: PROVISIONAL', 'truncation_reason: frozen_shared_world_model_validated_only_to_H4', 'evaluation_search: false')) {
        if (-not $config.Contains($needle)) { throw "BLOCKED: CARL config lacks frozen contract: $needle" }
    }
    $repeatSource = (Get-Content -Raw -LiteralPath $EvaluatorRepeat).ToLowerInvariant()
    foreach ($needle in @('carl_validation_bundle_frozen_before_sha256', 'carl_validation_bundle_frozen_after_sha256', 'frozen_before/frozen_after mismatch', 'source_training_target', 'source_training_bundle_sha256')) {
        if (-not $repeatSource.Contains($needle)) { throw "BLOCKED: CARL repeat evaluator lacks evidence freeze gate: $needle" }
    }
    $episodeSource = (Get-Content -Raw -LiteralPath $EvaluatorEpisode).ToLowerInvariant()
    foreach ($needle in @('provider_calls', 'test_seeds_used', 'online_provider_fallback')) {
        if (-not $episodeSource.Contains($needle)) { throw "BLOCKED: CARL episode evaluator lacks test isolation metadata: $needle" }
    }
}
function Assert-ResourceGate {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $ram = [math]::Round([double]$os.FreePhysicalMemory / 1024.0, 2)
    $cpu = [math]::Round([double](Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue, 2)
    if ($ram -lt 4096.0 -or $cpu -ge 85.0) { throw "BLOCKED: resource gate failed (free_ram_mib=$ram, cpu_percent=$cpu; require RAM>=4096 and CPU<85)" }
    return [ordered]@{ free_ram_mib = $ram; cpu_percent = $cpu; selected_device = 'cpu'; thresholds = 'RAM>=4096MiB; CPU<85%; device=cpu' }
}
function Assert-DeviceGate {
    $probe = @(& $Python -B -c "import torch; print(torch.__version__); print('cpu=ready')" 2>&1)
    if ($LASTEXITCODE -ne 0 -or ($probe -join "`n") -notmatch 'cpu=ready') { throw 'BLOCKED: CPU torch device probe failed' }
    return [ordered]@{ selected_device = 'cpu'; probe = ($probe -join ' '); gpu_required = $false; online_provider_allowed = $false }
}
function Assert-CliHelp {
    Push-Location -LiteralPath $WorktreeProject
    try { & $Python -B -m baselines.carl_cc4.formal_cli --help *> $null; if ($LASTEXITCODE -ne 0) { throw 'BLOCKED: CARL formal CLI --help failed' } }
    finally { Pop-Location }
}
function Get-Writers {
    $needles = @($Target, $TestTarget, $LogRoot, $TestLogRoot) | ForEach-Object { [regex]::Escape($_) }
    $pattern = ($needles -join '|')
    @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine -match $pattern -and $_.Name -match '^(python|pwsh|powershell)(\.exe)?$' })
}
function Assert-UniqueWriter {
    $writers = @(Get-Writers)
    if ($writers.Count -ne 0) { throw "BLOCKED: CARL target/log already has writer(s): $($writers.ProcessId -join ',')" }
}
function Assert-NoOverwrite([string]$Path, [string]$Label) {
    if (Test-Path -LiteralPath $Path) { throw "BLOCKED: refusing to overwrite existing ${Label}: $Path (use -Resume only for an exact train prefix)" }
}
function Assert-FreshTrainingBundle {
    $checkpoint = Join-Path $Target 'checkpoint.pt'
    $selection = Join-Path $Target 'validation_selection.json'
    $trainingManifest = Join-Path $Target 'training_manifest.json'
    Require-Sha $checkpoint $Expected.fresh_checkpoint 'fresh CARL checkpoint'
    Require-Sha $selection $Expected.fresh_selection 'fresh CARL validation selection'
    Require-Sha $trainingManifest $Expected.fresh_training_manifest 'fresh CARL training manifest'
    $manifest = Get-Content -Raw -LiteralPath $trainingManifest | ConvertFrom-Json
    if ($manifest.code_commit -ne $Expected.fresh_training_code_commit -or $manifest.policy_seed -ne 51001 -or $manifest.test_seeds_used -ne $false) { throw 'BLOCKED: fresh CARL training provenance identity drift' }
    if (($manifest.training_episode_seeds -join ',') -ne ((1000..1031) -join ',') -or ($manifest.validation_episode_seeds -join ',') -ne ((2000..2007) -join ',')) { throw 'BLOCKED: fresh CARL train/validation seed identity drift' }
    if ($manifest.checkpoint_sha256 -ne $Expected.fresh_checkpoint -or $manifest.validation_selection_sha256 -ne $Expected.fresh_selection) { throw 'BLOCKED: fresh CARL manifest SHA binding drift' }
    if ($manifest.formal_result_eligible -ne $true -or $manifest.paper_row_eligible -ne $false -or $manifest.repeat_status -ne 'PROVISIONAL' -or $manifest.effective_imagination_horizon -ne 4 -or $manifest.adaptation -ne 'ADAPTED_TRUNCATED') { throw 'BLOCKED: fresh CARL adapted protocol identity drift' }
    $selectionPayload = Get-Content -Raw -LiteralPath $selection | ConvertFrom-Json
    if ($selectionPayload.schema -ne 'carl_cc4_validation_selection_v2' -or $selectionPayload.policy_seed -ne 51001 -or $selectionPayload.git_dirty -ne $false -or $selectionPayload.test_seeds_used -ne $false -or ($selectionPayload.validation_episode_seeds -join ',') -ne ((2000..2007) -join ',')) { throw 'BLOCKED: fresh CARL selection identity drift' }
    return $manifest
}
function Get-RunnerArgs {
    $args = @('-B', '-m', 'baselines.carl_cc4.formal_cli', '--repeat-index', '1', '--train-replay', $TrainReplay,
        '--validation-replay', $ValidationReplay, '--world-model', $WorldModel, '--output', $Target, '--device', 'cpu')
    if ($Resume) { $args += '--resume' }
    return $args
}
function Write-StageLogs([string[]]$RunnerArgs, [object]$Resources) {
    if (Test-Path -LiteralPath $LogRoot) { throw "BLOCKED: refusing to overwrite launcher log root: $LogRoot" }
    New-Item -ItemType Directory -Path $LogRoot | Out-Null
    $stdout = Join-Path $LogRoot 'train.stdout.log'; $stderr = Join-Path $LogRoot 'train.stderr.log'
    $plan = [ordered]@{ schema = 'cc4_carl_repeat1_launcher_plan_v1'; stage = 'train'; method = 'carl_cc4'; repeat_index = 1; policy_seed = 51001; train_seeds = '1000..1031'; validation_seeds = '2000..2007'; test_seeds = '4000..4099'; ticks = 500; synthetic_rollouts_per_real = 8; requested_horizon = 256; effective_horizon = 4; imagination_gate = 'ADAPTED_TRUNCATED'; formal_result_eligible = $true; paper_row_eligible = $false; repeat_status = 'PROVISIONAL'; truncation_reason = 'frozen_shared_world_model_validated_only_to_H4'; target = $Target; logs = @($stdout, $stderr); runner = $Runner; runner_sha256 = $Expected.formal_cli; resources = $Resources; runner_args = $RunnerArgs; started_utc = [DateTime]::UtcNow.ToString('o') }
    [IO.File]::WriteAllText((Join-Path $LogRoot 'train.plan.json'), ($plan | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
    $process = Start-Process -FilePath $Python -ArgumentList $RunnerArgs -WorkingDirectory $WorktreeProject -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -Wait
    [IO.File]::WriteAllText((Join-Path $LogRoot 'train.exitcode'), [string]$process.ExitCode, [Text.UTF8Encoding]::new($false))
    if ($process.ExitCode -ne 0) { throw "BLOCKED: CARL train exit=$($process.ExitCode); logs retained in $LogRoot" }
}

$plan = [ordered]@{ schema = 'cc4_carl_repeat1_launcher_plan_v1'; status = 'INERT_PLAN'; method = 'carl_cc4'; row = 'CARL-CC4 (adapted)'; repeat_index = 1; policy_seed = 51001; train_seeds = '1000..1031'; validation_seeds = '2000..2007'; test_seeds = '4000..4099'; episode_ticks = 500; synthetic_rollouts_per_real = 8; requested_horizon = 256; effective_horizon = 4; imagination_gate = 'ADAPTED_TRUNCATED'; formal_result_eligible = $true; paper_row_eligible = $false; repeat_status = 'PROVISIONAL'; truncation_reason = 'frozen_shared_world_model_validated_only_to_H4'; worktree = $WorktreeRoot; head = $ExpectedHead; target = $Target; test_target = $TestTarget; log_root = $LogRoot; test_log_root = $TestLogRoot; train_command = "$($MyInvocation.MyCommand.Path) -Stage train -Execute"; validation_command = "$($MyInvocation.MyCommand.Path) -Stage validation -DryRun"; test_command = "$($MyInvocation.MyCommand.Path) -Stage test -DryRun"; note = 'No Stage is inert. Dry-run never creates target/log. H=4 adaptation is taskbook-disclosed; repeat-1 is PROVISIONAL and not a five-repeat paper-row PASS.' }

if ([string]::IsNullOrWhiteSpace($Stage)) { $plan | ConvertTo-Json -Depth 8; exit 0 }
if ($Execute -and $DryRun) { throw 'BLOCKED: choose either -Execute or -DryRun' }
try {
    Require-Identity; Require-StaticContract; Assert-UniqueWriter; Assert-CliHelp
    if ($Stage -eq 'preflight') {
        $resource = Assert-ResourceGate; $device = Assert-DeviceGate
        $plan.status = 'DRY_RUN_PASS'; $plan.stage = $Stage; $plan.resource_gate = $resource; $plan.device_gate = $device; $plan.would_write = $false
        $plan | ConvertTo-Json -Depth 10; exit 0
    }
    if ($Stage -eq 'train') {
        if ($Resume) {
            if (-not (Test-Path -LiteralPath (Join-Path $Target 'progress.pt') -PathType Leaf)) { throw "BLOCKED: -Resume requires existing exact-prefix progress.pt: $Target" }
        } else { Assert-NoOverwrite $Target 'CARL training target' }
        $resource = Assert-ResourceGate; $device = Assert-DeviceGate
        if ($DryRun) { $plan.status = 'DRY_RUN_PASS'; $plan.stage = $Stage; $plan.resource_gate = $resource; $plan.device_gate = $device; $plan.runner_args = Get-RunnerArgs; $plan.would_write = $false; $plan | ConvertTo-Json -Depth 10; exit 0 }
        if (-not $Execute) { $plan.status = 'INERT_STAGE'; $plan | ConvertTo-Json -Depth 8; exit 0 }
        Write-StageLogs (Get-RunnerArgs) $resource; $plan.status = 'PASS'; $plan.stage = $Stage; $plan.resource_gate = $resource; $plan.device_gate = $device; $plan | ConvertTo-Json -Depth 10; exit 0
    }
    if ($Stage -eq 'validation') {
        $manifest = Assert-FreshTrainingBundle
        $requiredReason = 'frozen_shared_world_model_validated_only_to_H4'
        if ($manifest.formal_result_eligible -ne $true -or $manifest.paper_row_eligible -ne $false -or $manifest.repeat_status -ne 'PROVISIONAL' -or $manifest.effective_imagination_horizon -ne 4 -or $manifest.adaptation -ne 'ADAPTED_TRUNCATED' -or $manifest.truncation_reason -ne $requiredReason) { throw 'BLOCKED: CARL adapted training manifest identity drift' }
        $selection = Join-Path $Target 'validation_selection.json'
        if (-not (Test-Path -LiteralPath $selection -PathType Leaf)) { throw "BLOCKED: CARL validation selection absent: $selection" }
        $plan.status = 'DRY_RUN_PASS'; $plan.stage = $Stage; $plan.reason = 'read-only validation selection accepted; no additional validation run'; $plan.would_write = $false; $plan.training_manifest = (Join-Path $Target 'training_manifest.json'); $plan.validation_selection = $selection; $plan | ConvertTo-Json -Depth 8; exit 0
    }
    if ($Stage -eq 'test') {
        $trainingManifestPath = Join-Path $Target 'training_manifest.json'
        $manifest = Assert-FreshTrainingBundle
        $requiredReason = 'frozen_shared_world_model_validated_only_to_H4'
        if ($manifest.formal_result_eligible -ne $true -or $manifest.paper_row_eligible -ne $false -or $manifest.repeat_status -ne 'PROVISIONAL' -or $manifest.adaptation -ne 'ADAPTED_TRUNCATED' -or $manifest.effective_imagination_horizon -ne 4 -or $manifest.truncation_reason -ne $requiredReason) { throw 'BLOCKED: CARL adapted training manifest identity drift' }
        if (-not (Test-Path -LiteralPath (Join-Path $Target 'checkpoint.pt') -PathType Leaf) -or -not (Test-Path -LiteralPath (Join-Path $Target 'validation_selection.json') -PathType Leaf)) { throw 'BLOCKED: CARL frozen checkpoint/selection absent' }
        Assert-NoOverwrite $TestTarget 'CARL formal test target'
        $resource = Assert-ResourceGate; $device = Assert-DeviceGate
        $testSeeds = (4000..4099) -join ','
        $testArgs = @('-B', '-m', 'formal_experiments.evaluation.run_method_repeat', '--method', 'carl_cc4', '--run-mode', 'formal', '--repeat-index', '1', '--policy-seed', '51001', '--episode-seeds', $testSeeds, '--ticks', '500', '--device', 'cpu', '--out', $TestTarget)
        if ($DryRun) { $plan.status = 'DRY_RUN_PASS'; $plan.stage = $Stage; $plan.resource_gate = $resource; $plan.device_gate = $device; $plan.runner_args = $testArgs; $plan.test_log_root = $TestLogRoot; $plan.formal_result_eligible = $true; $plan.paper_row_eligible = $false; $plan.repeat_status = 'PROVISIONAL'; $plan.would_write = $false; $plan | ConvertTo-Json -Depth 10; exit 0 }
        if (-not $Execute) { $plan.status = 'INERT_STAGE'; $plan.stage = $Stage; $plan | ConvertTo-Json -Depth 8; exit 0 }
        if (Test-Path -LiteralPath $TestLogRoot) { throw "BLOCKED: refusing to overwrite test launcher log root: $TestLogRoot" }
        New-Item -ItemType Directory -Path $TestLogRoot | Out-Null
        $stdout = Join-Path $TestLogRoot 'test.stdout.log'; $stderr = Join-Path $TestLogRoot 'test.stderr.log'
        $testPlan = [ordered]@{ schema = 'cc4_carl_repeat1_launcher_plan_v1'; stage = 'test'; method = 'carl_cc4'; repeat_index = 1; policy_seed = 51001; episode_seeds = '4000..4099'; ticks = 500; requested_horizon = 256; effective_horizon = 4; imagination_gate = 'ADAPTED_TRUNCATED'; formal_result_eligible = $true; paper_row_eligible = $false; repeat_status = 'PROVISIONAL'; truncation_reason = $requiredReason; target = $TestTarget; logs = @($stdout, $stderr); runner = $EvaluatorRepeat; runner_sha256 = $Expected.evaluator_repeat; resources = $resource; runner_args = $testArgs; started_utc = [DateTime]::UtcNow.ToString('o') }
        [IO.File]::WriteAllText((Join-Path $TestLogRoot 'test.plan.json'), ($testPlan | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
        $env:CC4_CARL_ARTIFACT_ROOT = $RepositoryRoot; $env:CC4_CARL_PROJECT_ROOT = $MainProjectRoot; $env:CC4_CARL_TRAINING_TARGET = $Target
        $process = Start-Process -FilePath $Python -ArgumentList $testArgs -WorkingDirectory $WorktreeProject -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -Wait
        [IO.File]::WriteAllText((Join-Path $TestLogRoot 'test.exitcode'), [string]$process.ExitCode, [Text.UTF8Encoding]::new($false))
        if ($process.ExitCode -ne 0) { throw "BLOCKED: CARL formal test exit=$($process.ExitCode); logs retained in $TestLogRoot" }
        $plan.status = 'PASS'; $plan.stage = $Stage; $plan.resource_gate = $resource; $plan.device_gate = $device; $plan.test_log_root = $TestLogRoot; $plan.target = $TestTarget; $plan.formal_result_eligible = $true; $plan.paper_row_eligible = $false; $plan.repeat_status = 'PROVISIONAL'; $plan | ConvertTo-Json -Depth 10; exit 0
    }
} catch { Write-Error $_; exit 2 }
