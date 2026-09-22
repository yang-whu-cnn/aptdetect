<#
.SYNOPSIS
  Stage-aware, non-overwriting TERLA-A4 repeat-1 launcher.

The implementation is executed from the clean pinned a553 worktree.  This
script lives in the dirty main checkout only as an inert, auditable handoff:
without -Stage it never runs Python; with -DryRun it performs read-only gates
and never creates a formal target or launcher log.
#>
[CmdletBinding()]
param(
    [ValidateSet('preflight', 'train', 'validation', 'test')]
    [string]$Stage,
    [switch]$Execute,
    [switch]$DryRun,
    [switch]$Resume,
    [switch]$AllowReviewedTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$MainRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$RepositoryRoot = (Resolve-Path (Join-Path $MainRoot '..')).Path
$CandidateRoot = 'C:\Users\25453\.codex\worktrees\a553\aptdetect'
$CandidateProject = Join-Path $CandidateRoot 'chapter2_region_detection'
$Python = Join-Path $MainRoot '.venv_cc4\Scripts\python.exe'
$SourceDir = Join-Path $MainRoot 'outputs\rebuild_20260920\terla_a4_resume_51001_attempt2'
$Migration = Join-Path $CandidateProject 'docs\TERLA_A4_POLICY51001_LINEAGE_MIGRATION_20260922.json'
$TrainTarget = Join-Path $MainRoot 'outputs\formal_v3\table1\terla_a4\repeat_1_trainonly_v1'
$EvalTarget = Join-Path $MainRoot 'outputs\formal_v3\table1\terla_a4\repeat_1_eval_v1'
$TestTarget = Join-Path $MainRoot 'outputs\formal_v3\table1\terla_a4\repeat_1'
$LogRoot = if ($Stage -eq 'test') {
    Join-Path $MainRoot 'outputs\launcher_logs\terla_a4_repeat1_test_20260922_v1'
} else {
    Join-Path $MainRoot 'outputs\launcher_logs\terla_a4_repeat1_stage_20260922'
}
$PolicySeed = 51001
$Repeat = 1
$MinRamGiB = 4.5

$ExpectedHead = 'fa5f02a82dd370a3bbcb095219bbbd033a52e9b3'
$Expected = [ordered]@{
    Mandatory = '5450bc5d692f4441497737f291121923693d15f6a27f92b5f55d0c24d91026a8'
    Taskbook = '6d862e9475c92a668358f69d9e09e9b32b7d23cd83679dcd4bd0d3847eae00a5'
    CandidateTaskbook = 'ff0366ad729297c5f4a6e8fcbe09674f20fb89631f4d7128f7df35f6039bdc20'
    StagedTraining = 'b57b091aab2ab2c7a91acd279b0b5edb5cea6ff43edaef065e248125c487daa0'
    StagedEvaluation = '50d76f09b873bbfb002ccd7ae489c1a6b116e0e846b5cd58aacba5a850a7197f'
    EpisodeRunner = 'fa5a8013347b917f20727acc37a5a204d17fe7dee74478afc3eabaa226a4a9b8'
    Migration = 'ac25d6a6b3cfff93c37c7fe0f7b1f0d83e0b60034f2e77228cf335c098dd5632'
    Config = 'a70edd3000c02d3f75af04d81a9d80d79ed3ce44d1b20c0015c82bbd46b75b7e'
    Model = 'c6c3322813e0628db063d8915584df316e2e9262356d403bd0a437f9d3488d24'
    Graph = '6d5f629b09fb0e400c3bfb0778716c8ccf9e50fd349898bc0afbab8ad8cd05e3'
    Reward = '028040484178554e71792be0f09164b732de3f5879c4c46140063618baf3c0c9'
    Runtime = 'e43b95640ec2cdabd26283122f4cce08039d4a37fe03131d1dc0b7b198ff9d27'
    Targeting = 'ccc8d483d3becb1d26c884f34a2cf900bbe7fc0eee5c09bfc16fecbc726dde46'
    SourceCheckpoint = '7c5e823f741ae805f751ccfa02444931f07e92f7687bcd39ca5000304c4ab783'
    SourceOptimizer = '5a153cf8f26ad1259f08743b6800f7576b88a25edb01dcc268e6801917e791f3'
    SourceManifest = 'a54ba4e6df63a7f7b03e78044e192240d77242fc239fc0db1cb8f38eaf915921'
    SourceResume = '66514c476150c74c2419419121371d8d3dd5aa0e57f031e7d28c0e2d52b596c8'
    SourceStep08 = '169fba50b253b3c62a279f0175cd09769261e3ffaa48d5a2d925033cc27c1efb'
    SourceStep16 = '5aca65c44862298f60f43c93b1cf4b03ad0f88ee765ec89d7ecca8081e531864'
    SourceStep24 = '4e7ff041c518a408da68b509f48b0ab77365d1b7984f398b95d1828212b31c58'
    SourceStep32 = '7d174dc2884baca8be146055d526a3396bd451abe320c6e21b1a7ab8b82dcf91'
    SourceDecisions = 'cd8132688067ed022174c7e790d3046982f595893888f2f5eb1f82f2450ede3f'
}

function Sha([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Require-Sha([string]$Path, [string]$ExpectedSha, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "BLOCKED: missing $Label`: $Path"
    }
    $actual = Sha $Path
    if ($actual -ne $ExpectedSha) {
        throw "BLOCKED: $Label SHA drift: $actual (expected $ExpectedSha)"
    }
}

function Assert-CandidateIdentity {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "BLOCKED: missing Python runtime: $Python"
    }
    if (-not (Test-Path -LiteralPath $CandidateRoot -PathType Container)) {
        throw "BLOCKED: pinned TERLA worktree is missing: $CandidateRoot"
    }
    $head = (git -C $CandidateRoot rev-parse HEAD).Trim()
    if ($head -ne $ExpectedHead) { throw "BLOCKED: candidate HEAD drift: $head" }
    $dirty = @(git -C $CandidateRoot status --porcelain=v1 --untracked-files=all)
    if ($dirty.Count -ne 0) { throw "BLOCKED: candidate worktree is dirty: $($dirty -join ' | ')" }
    Require-Sha (Join-Path $MainRoot 'docs\USER_MANDATORY_REQUIREMENTS.md') $Expected.Mandatory 'mandatory requirements'
    Require-Sha (Join-Path $RepositoryRoot 'CC4_V3_0917_FINAL_EXPERIMENT_TASKBOOK.md') $Expected.Taskbook 'taskbook'
    Require-Sha (Join-Path $CandidateRoot 'CC4_V3_0917_FINAL_EXPERIMENT_TASKBOOK.md') $Expected.CandidateTaskbook 'candidate taskbook'

    $candidateFiles = [ordered]@{
        StagedTraining = 'baselines\terla_a4\staged_training.py'
        StagedEvaluation = 'baselines\terla_a4\staged_evaluation.py'
        EpisodeRunner = 'formal_experiments\evaluation\run_method_episode.py'
        Migration = 'docs\TERLA_A4_POLICY51001_LINEAGE_MIGRATION_20260922.json'
        Config = 'baselines\terla_a4\config.yaml'
        Model = 'baselines\terla_a4\model.py'
        Graph = 'baselines\terla_a4\graph.py'
        Reward = 'baselines\terla_a4\reward.py'
        Runtime = 'baselines\terla_a4\runtime.py'
        Targeting = 'baselines\terla_a4\targeting.py'
    }
    foreach ($key in $candidateFiles.Keys) {
        Require-Sha (Join-Path $CandidateProject $candidateFiles[$key]) $Expected[$key] "candidate $key"
    }
    $sourceFiles = [ordered]@{
        SourceCheckpoint = 'policy_51001.pt'
        SourceOptimizer = 'policy_51001.optimizer.pt'
        SourceManifest = 'policy_51001.training.json'
        SourceResume = 'policy_51001.resume.pt'
        SourceStep08 = 'policy_51001.step_08.pt'
        SourceStep16 = 'policy_51001.step_16.pt'
        SourceStep24 = 'policy_51001.step_24.pt'
        SourceStep32 = 'policy_51001.step_32.pt'
        SourceDecisions = 'policy_51001.decisions.jsonl'
    }
    foreach ($key in $sourceFiles.Keys) {
        Require-Sha (Join-Path $SourceDir $sourceFiles[$key]) $Expected[$key] "source $($sourceFiles[$key])"
    }
}

function Get-ResourceGate {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $freeRamGiB = [double]$os.FreePhysicalMemory / 1MB
    $cpu = [double](Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue
    return [ordered]@{
        pass = ($freeRamGiB -ge $MinRamGiB -and $cpu -lt 90.0)
        free_ram_gib = [math]::Round($freeRamGiB, 3)
        cpu_percent = [math]::Round($cpu, 1)
        minimum_free_ram_gib = $MinRamGiB
        maximum_cpu_percent = 90.0
        device = 'cpu'
    }
}

function Assert-NoTargetWriter([string]$Target) {
    $token = [IO.Path]::GetFullPath($Target).TrimEnd('\')
    $writers = @(Get-CimInstance -ClassName Win32_Process | Where-Object {
        $cmd = [string]$_.CommandLine
        $cmd.IndexOf($token, [StringComparison]::OrdinalIgnoreCase) -ge 0
    })
    if ($writers.Count -gt 0) {
        throw "BLOCKED: target writer already observed: $($writers.ProcessId -join ',')"
    }
}

function Assert-StageSafety {
    if ($Stage -eq 'train') {
        if (Test-Path -LiteralPath $TrainTarget) {
            if (-not $Resume) { throw "BLOCKED: train target exists; use -Resume after identity review: $TrainTarget" }
            if (-not (Test-Path -LiteralPath (Join-Path $TrainTarget 'stage_state.json') -PathType Leaf)) {
                throw "BLOCKED: existing train target has no stage_state.json: $TrainTarget"
            }
        } elseif ($Resume) {
            throw "BLOCKED: -Resume requires an existing train target: $TrainTarget"
        }
        Assert-NoTargetWriter $TrainTarget
    } elseif ($Stage -eq 'validation') {
        if (Test-Path -LiteralPath $EvalTarget) { throw "BLOCKED: validation target exists; refusing overwrite: $EvalTarget" }
        Assert-NoTargetWriter $EvalTarget
    } elseif ($Stage -eq 'test') {
        if (-not $AllowReviewedTest) { throw 'BLOCKED: test requires -AllowReviewedTest and separate validation review' }
        if (-not (Test-Path -LiteralPath $EvalTarget -PathType Container)) { throw "BLOCKED: validation target is absent: $EvalTarget" }
        $statePath = Join-Path $EvalTarget 'stage_state.json'
        if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) { throw "BLOCKED: validation state is absent: $statePath" }
        $state = Get-Content -Raw -LiteralPath $statePath | ConvertFrom-Json
        if ($state.stage -ne 'validation' -or $state.status -ne 'COMPLETE' -or
            (($state.completed_seeds -join ',') -ne '2000,2001,2002,2003,2004,2005,2006,2007')) {
            throw 'BLOCKED: test requires a complete, untouched validation stage'
        }
        if (Test-Path -LiteralPath $TestTarget) { throw "BLOCKED: test target exists; refusing overwrite: $TestTarget" }
        Assert-NoTargetWriter $EvalTarget
        Assert-NoTargetWriter $TestTarget
    }
}

function Candidate-Args([string]$RunStage, [string]$Target) {
    if ($RunStage -eq 'train') {
        $args = @('-B', '-m', 'baselines.terla_a4.staged_training', '--stage', 'train',
            '--stop-after', 'train', '--policy-seed', "$PolicySeed", '--repeat', "$Repeat",
            '--output-dir', $Target)
        if ($Resume) { $args += '--resume' }
        return $args
    }
    return @('-B', '-m', 'baselines.terla_a4.staged_evaluation', '--stage', $RunStage,
        '--source-dir', $SourceDir, '--migration', $Migration, '--target-dir', $Target,
        '--policy-seed', "$PolicySeed", '--repeat', "$Repeat", '--device', 'cpu',
        '--min-ram-gib', "$MinRamGiB") + $(if ($RunStage -eq 'test') {
            @('--validation-dir', $EvalTarget)
        } else { @() })
}

function Invoke-Candidate([string[]]$Arguments, [int[]]$AllowedExitCodes) {
    if (Test-Path -LiteralPath $LogRoot) { throw "BLOCKED: refusing pre-existing launcher log root: $LogRoot" }
    New-Item -ItemType Directory -Path $LogRoot | Out-Null
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
    $stdout = Join-Path $LogRoot "$Stage`_$stamp.stdout.log"
    $stderr = Join-Path $LogRoot "$Stage`_$stamp.stderr.log"
    $planPath = Join-Path $LogRoot "$Stage`_$stamp.plan.json"
    $plan = [ordered]@{ launcher = $PSCommandPath; launcher_sha256 = Sha $PSCommandPath
        candidate_head = $ExpectedHead; runner_args = $Arguments; source_dir = $SourceDir
        target = if ($Stage -eq 'train') { $TrainTarget } elseif ($Stage -eq 'test') { $TestTarget } else { $EvalTarget }
        validation_target = $EvalTarget; test_target = $TestTarget
        started_utc = [DateTime]::UtcNow.ToString('o') }
    $plan | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $planPath -Encoding utf8NoBOM
    $child = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $CandidateProject `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    $child.WaitForExit()
    $exitCode = [int]$child.ExitCode
    $exitPath = Join-Path $LogRoot "$Stage`_$stamp.exitcode"
    [string]$exitCode | Set-Content -LiteralPath $exitPath -Encoding ascii
    if ($AllowedExitCodes -notcontains $exitCode) {
        throw "BLOCKED: candidate stage exited $exitCode; logs retained under $LogRoot"
    }
    return [ordered]@{ exit_code = $exitCode; stdout = $stdout; stderr = $stderr; plan = $planPath }
}

function Read-OnlyCandidatePreflight([string]$Target) {
    $args = @('-B', '-m', 'baselines.terla_a4.staged_evaluation', '--stage', 'preflight',
        '--source-dir', $SourceDir, '--migration', $Migration, '--target-dir', $Target,
        '--policy-seed', "$PolicySeed", '--repeat', "$Repeat", '--device', 'cpu',
        # The launcher resource gate below is authoritative for a dry-run.  A
        # small candidate threshold lets identity/target checks run and report
        # BLOCKED resources without making the dry-run itself fail early.
        '--min-ram-gib', $(if ($Execute) { "$MinRamGiB" } else { '0.1' }), '--dry-run')
    Push-Location $CandidateProject
    try {
        & $Python @args
        if ($LASTEXITCODE -ne 0) { throw "BLOCKED: candidate preflight exit=$LASTEXITCODE" }
    } finally { Pop-Location }
}

function Read-OnlyCandidateStage([string]$RunStage, [string]$Target) {
    $args = Candidate-Args $RunStage $Target
    $ramIndex = [Array]::IndexOf([string[]]$args, '--min-ram-gib')
    if ($ramIndex -ge 0) { $args[$ramIndex + 1] = '0.1' }
    $args += '--dry-run'
    if ($RunStage -eq 'test') { $args += '--allow-reviewed-test' }
    Push-Location $CandidateProject
    try {
        & $Python @args
        if ($LASTEXITCODE -ne 0) { throw "BLOCKED: candidate $RunStage dry-run exit=$LASTEXITCODE" }
    } finally { Pop-Location }
}

$plan = [ordered]@{
    launcher = 'terla_a4_repeat1_stage_launcher_20260922'
    status = 'INERT_PLAN'
    table_id = 'table1'; method = 'terla_a4'; method_label = 'TERLA-A4 (adapted)'
    repeat_index = $Repeat; policy_seed = $PolicySeed
    train_seeds = '1000..1031'; validation_seeds = '2000..2007'; test_seeds = '4000..4099'
    ticks_per_episode = 500; train_target = $TrainTarget; validation_target = $EvalTarget; test_target = $TestTarget
    candidate_worktree = $CandidateRoot; candidate_head = $ExpectedHead
    main_taskbook_sha256 = $Expected.Taskbook; candidate_taskbook_sha256 = $Expected.CandidateTaskbook
    source_lineage = $SourceDir; migration = $Migration; log_root = $LogRoot
    no_online_provider = $true; device = 'cpu'; formal_result_eligible = $false
    note = 'repeat_1 is provisional; no repeat_2..5 and no long stage is started by default'
}

if ([string]::IsNullOrWhiteSpace($Stage)) {
    $plan | ConvertTo-Json -Depth 8
    exit 0
}
if ($Execute -and $DryRun) { throw 'BLOCKED: choose either -Execute or -DryRun' }
if ($Stage -eq 'preflight' -and $Execute) { throw 'BLOCKED: preflight is read-only; omit -Execute' }
if ($Stage -eq 'test' -and -not $AllowReviewedTest) { throw 'BLOCKED: test requires -AllowReviewedTest' }

Assert-CandidateIdentity
Assert-StageSafety
$resource = Get-ResourceGate
$plan.status = if ($DryRun) { 'DRY_RUN' } else { 'INERT_PLAN' }
$plan.stage = $Stage; $plan.resume = [bool]$Resume; $plan.allow_reviewed_test = [bool]$AllowReviewedTest
$plan.resource_gate = $resource
$plan.would_write = [bool]$Execute

if (-not $Execute) {
    if ($Stage -eq 'train') {
        Push-Location $CandidateProject
        try {
            $trainPreflightArgs = @('-B', '-m', 'baselines.terla_a4.staged_training', '--stage', 'preflight',
                '--policy-seed', "$PolicySeed", '--repeat', "$Repeat", '--output-dir', $TrainTarget)
            if ($Resume) { $trainPreflightArgs += '--resume' }
            & $Python @trainPreflightArgs
            if ($LASTEXITCODE -ne 0) { throw "BLOCKED: train preflight exit=$LASTEXITCODE" }
        } finally { Pop-Location }
    } elseif ($Stage -eq 'validation') {
        Read-OnlyCandidateStage $Stage $EvalTarget
    } elseif ($Stage -eq 'test') {
        Read-OnlyCandidateStage $Stage $TestTarget
    } else {
        Read-OnlyCandidatePreflight $EvalTarget
    }
    $plan | ConvertTo-Json -Depth 8
    exit 0
}
if (-not $resource.pass) { throw "BLOCKED: resource gate failed: $($resource | ConvertTo-Json -Compress)" }

$before = @{}
$validationBefore = $null
if ($Stage -eq 'test') {
    foreach ($path in @((Join-Path $SourceDir 'policy_51001.pt'), (Join-Path $SourceDir 'policy_51001.optimizer.pt'),
        (Join-Path $SourceDir 'policy_51001.training.json'))) { $before[$path] = Sha $path }
    $validationBefore = Get-ChildItem -LiteralPath $EvalTarget -File -Recurse |
        Sort-Object FullName | ForEach-Object {
            [ordered]@{ path = $_.FullName.Substring($EvalTarget.Length).TrimStart('\'); sha256 = Sha $_.FullName }
        } | ConvertTo-Json -Compress
}
$target = if ($Stage -eq 'train') { $TrainTarget } elseif ($Stage -eq 'test') { $TestTarget } else { $EvalTarget }
$result = if ($Stage -eq 'train') { Invoke-Candidate (Candidate-Args 'train' $target) @(3) }
          elseif ($Stage -eq 'validation') { Invoke-Candidate (Candidate-Args 'validation' $target) @(0) }
          else { Invoke-Candidate ((Candidate-Args 'test' $target) + @('--allow-reviewed-test')) @(0) }
if ($Stage -eq 'test') {
    foreach ($path in $before.Keys) { if ((Sha $path) -ne $before[$path]) { throw "BLOCKED: source asset changed during test: $path" } }
    $validationAfter = Get-ChildItem -LiteralPath $EvalTarget -File -Recurse |
        Sort-Object FullName | ForEach-Object {
            [ordered]@{ path = $_.FullName.Substring($EvalTarget.Length).TrimStart('\'); sha256 = Sha $_.FullName }
        } | ConvertTo-Json -Compress
    if ($validationAfter -ne $validationBefore) { throw "BLOCKED: validation artifacts changed during test" }
}
$plan.status = 'PASS'; $plan.result = $result
$plan | ConvertTo-Json -Depth 8
