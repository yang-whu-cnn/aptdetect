[CmdletBinding()]
param(
    [string]$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$chapter2 = Join-Path $repo 'chapter2_region_detection'
$catalog = Join-Path $repo 'experiment'

function Ensure-Directory([string]$Path) {
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
}

function Copy-Tree([string]$Source, [string]$Destination) {
    if (-not (Test-Path -LiteralPath $Source)) { return }
    Ensure-Directory $Destination
    Get-ChildItem -LiteralPath $Source -File -Recurse -Force |
        Where-Object {
            $_.Extension -ne '.pyc' -and
            $_.FullName -notmatch '[\\/]__pycache__[\\/]' -and
            $_.FullName -notmatch '[\\/]\.pytest_cache[\\/]' -and
            $_.Name -notmatch '(?i)(^|_)pilot([_.]|$)' -and
            $_.Name -notmatch '(?i)(^|_)smoke([_.]|$)'
        } | ForEach-Object {
        $relative = [IO.Path]::GetRelativePath($Source, $_.FullName)
        $target = Join-Path $Destination $relative
        Ensure-Directory (Split-Path -Parent $target)
        Copy-Item -LiteralPath $_.FullName -Destination $target -Force
    }
}

function Copy-Files([string[]]$Sources, [string]$Destination) {
    Ensure-Directory $Destination
    foreach ($source in $Sources) {
        if (Test-Path -LiteralPath $source) {
            Copy-Item -LiteralPath $source -Destination $Destination -Force
        }
    }
}

function Write-Json([string]$Path, $Value) {
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $Path -Encoding utf8
}

function Write-Text([string]$Path, [string]$Text) {
    Ensure-Directory (Split-Path -Parent $Path)
    Set-Content -LiteralPath $Path -Value $Text -Encoding utf8
}

function Get-Sha256([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

# Copy compact sidecars only. Raw traces, checkpoints and journals remain at
# their canonical source paths and are never duplicated in this catalog.
$canonicalMetadataPattern = '^(\.true_ppo_writer\.lock\.json|aggregate\.json|backup_manifest(_v2)?\.json|backup_manifest(_v2)?\.sha256|calibration_manifest\.json|config\.resolved\.yaml|eligibility_report\.json|evaluation_manifest\.json|frozen_manifest\.json|journal_index\.json|manifest\.json|metrics\.json|normalizer\.json|policy_spec\.json|prepare_test_authorization\.json|preflight_snapshot\.json|provenance.*\.json|.*coverage.*\.json|.*hash.*\.(json|sha256)|resume_index\.json|resume_state\.json|source_manifest\.json|stage_state\.json|test_summary\.json|train_summary\.json|training_manifest\.json|validation\.json|validation_selection\.json|validation_records\.json|writer_lease\.json)$'

function Copy-CanonicalMetadata {
    param(
        [string]$Source,
        [string]$Destination,
        [bool]$Active = $false
    )
    Ensure-Directory $Destination
    $summary = [ordered]@{
        source = $Source
        source_exists = (Test-Path -LiteralPath $Source)
        active_source = $Active
        copied_files = @()
        skipped_large_files = @()
    }
    if (-not $summary.source_exists -or $Active) {
        Write-Json (Join-Path $Destination 'source_binding.json') ([ordered]@{
            schema_version = 1
            source = $Source
            source_exists = [bool]$summary.source_exists
            active_source = $Active
            note = if ($Active) { 'Source is actively written; catalog keeps a stable binding only and does not copy files.' } else { 'Canonical source is absent; binding retained for fail-closed audit.' }
        })
        Write-Json (Join-Path $Destination 'source_metadata_hashes.json') $summary
        return $summary
    }
    $sourceItem = Get-Item -LiteralPath $Source
    $allFiles = if ($sourceItem.PSIsContainer) { @(Get-ChildItem -LiteralPath $Source -File -Recurse -Force) } else { @($sourceItem) }
    foreach ($file in $allFiles) {
        if ($file.Name -notmatch $canonicalMetadataPattern) { continue }
        $relative = if ($sourceItem.PSIsContainer) { [IO.Path]::GetRelativePath($Source, $file.FullName) } else { $file.Name }
        if ($file.Length -gt 10MB) {
            $summary.skipped_large_files += [ordered]@{ relative_path = $relative; bytes = [int64]$file.Length }
            continue
        }
        $target = Join-Path $Destination $relative
        Ensure-Directory (Split-Path -Parent $target)
        Copy-Item -LiteralPath $file.FullName -Destination $target -Force
        $summary.copied_files += [ordered]@{ relative_path = $relative; bytes = [int64]$file.Length; sha256 = (Get-Sha256 $file.FullName) }
    }
    Write-Json (Join-Path $Destination 'source_metadata_hashes.json') $summary
    return $summary
}

function Write-MethodReadme {
    param([string]$MethodDirectory, $Method, [bool]$Alias = $false)
    $aliasText = if ($Alias) { "`nThis entry is a logical alias. It never starts a second run and never copies raw trajectories." } else { '' }
    $text = @"
# $($Method.name)

Catalog entry for the formal Chapter 2 CC4/A4 experiment. The canonical
execution source remains under `chapter2_region_detection/`; this directory is
an auditable, compact catalog of code/configuration/launcher bindings and
result metadata. Raw traces and checkpoints are intentionally not duplicated.

- Table: `$($Method.table)`
- Baseline identity: `$($Method.baseline)`
- Status: `$($Method.status)`
- Canonical source: `$($Method.result)`
- Protocol: repeat 1 sprint, test seeds `4000..4099`, 500 ticks/episode;
  DCA retains its immutable five-repeat exception.
- Admission: only a verified `eligibility_report.json` with `passed=true` is
  paper-eligible; repeat-1-only entries remain provisional.
- Note: $($Method.note)
$aliasText

    See `configs/`, `launchers/`, `manifests/`, `results/result_pointer.json`, and
`results/metadata/` for frozen bindings and compact evidence copied from the
canonical source.
"@
    Write-Text (Join-Path $MethodDirectory 'README.md') $text
    Write-Text (Join-Path $MethodDirectory 'docs\catalog_notes.md') $text
}

function Write-MethodBindings {
    param([string]$MethodDirectory, $Method, [bool]$Alias = $false)
    $configDir = Join-Path $MethodDirectory 'configs'
    $launcherDir = Join-Path $MethodDirectory 'launchers'
    $manifestDir = Join-Path $MethodDirectory 'manifests'
    Ensure-Directory $configDir; Ensure-Directory $launcherDir; Ensure-Directory $manifestDir
    Write-Json (Join-Path $configDir 'resolved_config_binding.json') ([ordered]@{
        schema_version = 1
        method = $Method.name
        source_config = if ($Method.config) { "chapter2_region_detection/configs/formal_v3/methods/$($Method.config)" } else { 'shared formal runner configuration' }
        canonical_result = $Method.result
        status = $Method.status
        alias = $Alias
        note = if ($Alias) { 'Non-executing alias; configuration is inherited from the physical Table 2 LWM-RL entry.' } else { 'Catalog binding; do not edit to tune a result.' }
    })
    Write-Json (Join-Path $manifestDir 'catalog_manifest_binding.json') ([ordered]@{
        schema_version = 1
        method = $Method.name
        source = $Method.result
        status = $Method.status
        canonical = [bool]$Method.canonical
        alias_of = $Method.alias_of
        raw_artifacts = 'retained at canonical source; only compact sidecars are copied here'
    })
    if ($Alias) {
        Write-Text (Join-Path $launcherDir 'ALIAS_NON_EXECUTING.md') @"
# Non-executing alias

`$($Method.name)` aliases `$($Method.alias_of)`. Do not launch a second run.
The canonical physical command and result are inherited from the Table 2 LWM-RL
entry; this file exists to make the alias boundary explicit and auditable.
"@
    } elseif ($Method.launcher) {
        Write-Text (Join-Path $launcherDir 'launcher_binding.md') @"
# Launcher binding

Canonical launcher: `chapter2_region_detection/scripts/$($Method.launcher)`
This catalog copy is documentation only; launch from the canonical worktree
after the normal preflight, identity, unique-writer and backup gates pass.
"@
    } else {
        Write-Text (Join-Path $launcherDir 'REPRODUCE_FORMAL_REPEAT1.md') @"
# Formal repeat-1 launcher declaration

This entry has no standalone shell wrapper in the canonical tree. The formal
runner is `chapter2_region_detection/formal_experiments/evaluation/run_method_repeat.py`
for baseline methods, or `chapter2_region_detection/formal_experiments/ours/run_table23_true_ppo.py`
for Table 2/3 variants. Reproduce only after preflight and frozen-identity
checks, using method `$($Method.baseline)$($Method.name)` with seeds `4000..4099`
and `500` ticks. This catalog file is deliberately a declaration, not an
automatic execution trigger.
"@
    }
}

function Add-ResultSnapshot {
    param(
        [string]$MethodDirectory,
        [string]$Source,
        [string]$Status,
        [bool]$Canonical,
        [string]$Note,
        [string]$AliasOf = '',
        [bool]$Active = $false
    )

    $results = Join-Path $MethodDirectory 'results'
    $metadata = Join-Path $results 'metadata'
    Ensure-Directory $metadata
    $exists = Test-Path -LiteralPath $Source
    $files = @()
    $bytes = [int64]0
    if ($Active) {
        # These are compact catalog copies from an earlier build, not source
        # artifacts. Remove them so an active pointer cannot expose stale
        # metadata while the canonical writer is still changing the result.
        Get-ChildItem -LiteralPath $metadata -File -Force -ErrorAction SilentlyContinue | Remove-Item -Force
        Write-Json (Join-Path $metadata 'active_source_binding.json') ([ordered]@{
            schema_version = 1
            source = $Source
            status = $Status
            active_writer = $true
            note = 'Stable binding only; no source metadata copied until the writer releases its lock.'
        })
    }
    if ($exists -and -not $Active) {
        $sourceItem = Get-Item -LiteralPath $Source
        if ($sourceItem.PSIsContainer) {
            $allFiles = @(Get-ChildItem -LiteralPath $Source -File -Recurse -Force)
        } else {
            $allFiles = @($sourceItem)
        }
        $bytes = [int64](($allFiles | Measure-Object -Property Length -Sum).Sum)
        $files = @($allFiles | ForEach-Object {
            $relative = if ($sourceItem.PSIsContainer) {
                [IO.Path]::GetRelativePath($Source, $_.FullName)
            } else {
                $_.Name
            }
            [ordered]@{ relative_path = $relative; bytes = [int64]$_.Length }
        })

        $safeNames = @(
            '.true_ppo_writer.lock.json', 'aggregate.json', 'calibration_manifest.json',
            'config.resolved.yaml', 'eligibility_report.json', 'evaluation_manifest.json',
            'journal_index.json', 'manifest.json', 'metrics.json', 'normalizer.json',
            'policy_spec.json', 'prepare_test_authorization.json', 'preflight_snapshot.json', 'resume_index.json',
            'resume_state.json', 'stage_state.json', 'test_summary.json',
            'train_summary.json', 'training_manifest.json', 'validation.json',
            'validation_selection.json', 'validation_records.json', 'writer_lease.json',
            'backup_manifest.json', 'backup_manifest_v2.json', 'backup_manifest.sha256',
            'backup_manifest_v2.sha256', 'source_manifest.json', 'frozen_manifest.json'
        )
        foreach ($file in $allFiles) {
            if (($safeNames -contains $file.Name) -and $file.Length -le 10MB) {
                $relativeParent = if ($sourceItem.PSIsContainer) {
                    [IO.Path]::GetRelativePath($Source, $file.DirectoryName)
                } else { '.' }
                $destination = if ($relativeParent -eq '.') { $metadata } else { Join-Path $metadata $relativeParent }
                Ensure-Directory $destination
                Copy-Item -LiteralPath $file.FullName -Destination $destination -Force
            }
        }
    }

    $pointer = [ordered]@{
        schema_version = 1
        generated_at = (Get-Date).ToUniversalTime().ToString('o')
        source = $Source
        source_exists = $exists
        status = $Status
        canonical = $Canonical
        alias_of = $AliasOf
        note = $Note
        file_count = $files.Count
        total_bytes = $bytes
        files = $files
        inventory_note = if ($Active) { 'Inventory intentionally omitted while canonical writer is active.' } else { 'Compact metadata copied; raw artifacts remain at source.' }
    }
    Write-Json (Join-Path $results 'result_pointer.json') $pointer
}

Ensure-Directory $catalog

$shared = Join-Path $catalog '_shared'
Copy-Tree (Join-Path $chapter2 'formal_experiments\common') (Join-Path $shared 'code\formal_experiments\common')
Copy-Tree (Join-Path $chapter2 'formal_experiments\evaluation') (Join-Path $shared 'code\formal_experiments\evaluation')
Copy-Tree (Join-Path $chapter2 'formal_experiments\ours') (Join-Path $shared 'code\formal_experiments\ours')
Copy-Tree (Join-Path $chapter2 'shared') (Join-Path $shared 'code\shared')
Copy-Tree (Join-Path $chapter2 'src') (Join-Path $shared 'code\src')
Copy-Tree (Join-Path $chapter2 'tests') (Join-Path $shared 'tests')
Copy-Tree (Join-Path $chapter2 'configs\formal_v3') (Join-Path $shared 'configs\formal_v3')
Copy-Files @(
    (Join-Path $chapter2 'requirements.txt'),
    (Join-Path $chapter2 'requirements_cc4_freeze.txt')
) (Join-Path $shared 'manifests')
$latestHandoff = Get-ChildItem -LiteralPath (Join-Path $chapter2 'docs') -File -Filter 'SESSION_HANDOFF_*.md' | Sort-Object LastWriteTime | Select-Object -Last 1
Copy-Files @(
    (Join-Path $repo 'CC4_V3_0917_FINAL_EXPERIMENT_TASKBOOK.md'),
    (Join-Path $chapter2 'docs\USER_MANDATORY_REQUIREMENTS.md'),
    (Join-Path $chapter2 'docs\CHATGPT_HANDOFF_FINAL_EXPERIMENTS.md'),
    (Join-Path $chapter2 'docs\TRUE_PPO_FINAL_IMPLEMENTATION_PLAN_20260921.md'),
    (Join-Path $chapter2 'docs\EXPERIMENT_BACKUP_PROTOCOL.md'),
    (Join-Path $chapter2 'docs\FINAL_REWARD_MODEL_MANIFEST.json'),
    (Join-Path $chapter2 'docs\FINAL_REWARD_REPLAY_MANIFEST.json'),
    (Join-Path $chapter2 'docs\AGGREGATION_READINESS_REPEAT1_20260922.md'),
    (Join-Path $chapter2 'docs\CARL_CC4_REPEAT1_READINESS_20260922.md'),
    (Join-Path $catalog 'CLEANUP_REPORT.md'),
    (Join-Path $chapter2 'outputs\formal_v3\aggregate_tables\repeat1_provisional_v4_20260922T2258\provisional_report.json'),
    (Join-Path $chapter2 'outputs\formal_v3\aggregate_tables\repeat1_provisional_v4_20260922T2258\provisional_tables.md'),
    (Join-Path $chapter2 'outputs\formal_v3\aggregate_tables\repeat1_provisional_v4_20260922T2258\table1.csv'),
    (Join-Path $chapter2 'outputs\formal_v3\aggregate_tables\repeat1_provisional_v4_20260922T2258\table2.csv'),
    (Join-Path $chapter2 'outputs\formal_v3\aggregate_tables\repeat1_provisional_v4_20260922T2258\table3.csv'),
    $(if ($latestHandoff) { $latestHandoff.FullName })
) (Join-Path $shared 'docs')

# Keep a compact, explicit index of frozen dependencies used by the formal
# runners. This records bindings without copying model/checkpoint payloads.
Write-Json (Join-Path $shared 'manifests\frozen_asset_bindings.json') ([ordered]@{
    schema_version = 1
    generated_at = (Get-Date).ToUniversalTime().ToString('o')
    reward_model_manifest = 'chapter2_region_detection/docs/FINAL_REWARD_MODEL_MANIFEST.json'
    replay_manifest = 'chapter2_region_detection/docs/FINAL_REWARD_REPLAY_MANIFEST.json'
    prior_prototypes = 'chapter2_region_detection/outputs/priorrl_cc4/prototypes/frozen_prototypes.json'
    prior_coverage = 'chapter2_region_detection/outputs/priorrl_cc4/prototypes/frozen_prototype_coverage.json'
    prior_provenance = 'chapter2_region_detection/outputs/priorrl_cc4/prototypes/frozen_prototype_provenance.json'
    world_model = 'chapter2_region_detection/outputs/world_model_final_20260917/a4_5b/world_model_absolute.pt'
    note = 'Bindings only; large model and replay payloads remain at canonical paths.'
})

$launchers = @(
    'table1_carl_cc4_20260922_repeat1.ps1',
    'terla_a4_repeat1_stage_launcher_20260922.ps1',
    'table2_llm_rl_trueppo_20260922_repeat1.ps1',
    'table2_wm_rl_trueppo_20260922_repeat1.ps1',
    'table3_delay_only_trueppo_20260922_repeat1.ps1',
    'table3_fail_only_trueppo_20260922_repeat1.ps1'
) | ForEach-Object { Join-Path $chapter2 "scripts\$_" }
Copy-Files $launchers (Join-Path $shared 'launchers')

$methods = @(
    [ordered]@{ table='table1'; name='uamcts_cc4_adapted'; baseline='uamcts_cc4'; config=''; launcher=''; result='outputs\formal_v3\table1\uamcts_cc4\repeat_1_fresh_support_v1_retry1'; status='RUNNING_TEST_DO_NOT_MOVE'; canonical=$false; active=$true; note='Fresh-support retry1 is actively writing; old failed v1 is retained as audit evidence.' },
    [ordered]@{ table='table1'; name='rsmbrl_cc4'; baseline='rsmbrl_cc4'; config='rsmbrl_cc4.yaml'; launcher=''; result='outputs\formal_v3\dependencies\backups\rsmbrl_repeat1_pass_20260921T230914Z_v2'; status='PROVISIONAL_REPEAT1_PASS'; canonical=$true; active=$false; note='Immutable local backup of the external canonical repeat1 result.' },
    [ordered]@{ table='table1'; name='carl_cc4_adapted'; baseline='carl_cc4'; config='carl_cc4.yaml'; launcher='table1_carl_cc4_20260922_repeat1.ps1'; result='outputs\formal_v3\table1\carl_cc4\repeat_1_retry2_scientific_fix'; status='PROVISIONAL_REPEAT1_PASS'; canonical=$true; active=$false; note='Scientific-fix retry2 is the canonical repeat1 candidate.' },
    [ordered]@{ table='table1'; name='dca_cc4_adapted'; baseline='dca_cc4'; config='dca_cc4.yaml'; launcher=''; result='outputs\formal_v3\table1\dca_cc4'; status='FORMAL_FIVE_REPEAT_PASS_IMMUTABLE'; canonical=$true; active=$false; note='Five-repeat PASS row; never overwrite or rerun.' },
    [ordered]@{ table='table1'; name='priorrl_ppo_cc4'; baseline='priorrl_cc4'; config='priorrl_ppo_cc4.yaml'; launcher=''; result='outputs\formal_v3\table1\priorrl_ppo_cc4\repeat_1'; status='PROVISIONAL_REPEAT1_PASS'; canonical=$true; active=$false; note='Canonical repeat1 only; old alpha/partial outputs excluded.' },
    [ordered]@{ table='table1'; name='terla_a4'; baseline='terla_a4'; config='terla_a4.yaml'; launcher='terla_a4_repeat1_stage_launcher_20260922.ps1'; result='outputs\formal_v3\table1\terla_a4\repeat_1'; status='PARTIAL_BLOCKED_ADMISSION'; canonical=$false; active=$false; note='Raw test exists; generic eligibility is blocked by adapted display-name compatibility.' },
    [ordered]@{ table='table2'; name='rl_only'; baseline=''; config=''; launcher=''; result='outputs\formal_v3\table2\rl_only_trueppo_v2\repeat_1'; status='PROVISIONAL_REPEAT1_PASS'; canonical=$true; active=$false; note='True-PPO repeat1 source and aggregate PASS.' },
    [ordered]@{ table='table2'; name='llm_rl'; baseline=''; config=''; launcher='table2_llm_rl_trueppo_20260922_repeat1.ps1'; result='outputs\formal_v3\table2\llm_rl_trueppo_20260922_repeat1\repeat_1'; status='PROVISIONAL_REPEAT1_PASS'; canonical=$true; active=$false; note='True-PPO repeat1 source and aggregate PASS.' },
    [ordered]@{ table='table2'; name='wm_rl'; baseline=''; config=''; launcher='table2_wm_rl_trueppo_20260922_repeat1.ps1'; result='outputs\formal_v3\table2\wm_rl_trueppo_20260922_repeat1\repeat_1'; status='RUNNING_TEST_DO_NOT_MOVE'; canonical=$false; active=$true; note='Active writer at catalog time; pointer and binding only until completion.' },
    [ordered]@{ table='table2'; name='lwm_rl'; baseline=''; config=''; launcher=''; result='outputs\formal_v3\table2\lwm_rl_trueppo_v3\repeat_1'; status='PROVISIONAL_REPEAT1_PASS'; canonical=$true; active=$false; note='True-PPO repeat1 source and aggregate PASS.' },
    [ordered]@{ table='table3'; name='delay_only'; baseline=''; config=''; launcher='table3_delay_only_trueppo_20260922_repeat1.ps1'; result='outputs\formal_v3\table3\delay_only_trueppo_20260922_repeat1'; status='RUNNING_VALIDATION_DO_NOT_MOVE'; canonical=$false; active=$true; note='Validation writer active; catalog stores stable binding only until validation/test complete.' },
    [ordered]@{ table='table3'; name='fail_only'; baseline=''; config=''; launcher='table3_fail_only_trueppo_20260922_repeat1.ps1'; result='outputs\formal_v3\table3\fail_only_trueppo_20260922_repeat1'; status='PROVISIONAL_REPEAT1_PASS'; canonical=$true; active=$false; note='True-PPO repeat1 completed with eligibility passed=true; remains provisional until the formal repeat requirement is met.' }
)

$oursFiles = @(
    'run_table23_formal.py', 'table2_variants.py', 'variant_training.py',
    'ppo_core.py', 'ppo_training.py', 'llm_prior_v2.py', 'prior_cache.py',
    'reward_ablation.py', 'reward_artifact_contract.py', 'reward_provenance.py'
) | ForEach-Object { Join-Path $chapter2 "formal_experiments\ours\$_" }

foreach ($method in $methods) {
    $methodDirectory = Join-Path $catalog "$($method.table)\$($method.name)"
    foreach ($part in @('code','configs','launchers','results','manifests','docs')) {
        Ensure-Directory (Join-Path $methodDirectory $part)
    }
    if ($method.baseline) {
        Copy-Tree (Join-Path $chapter2 "baselines\$($method.baseline)") (Join-Path $methodDirectory "code\$($method.baseline)")
        if ($method.name -eq 'rsmbrl_cc4') {
            Copy-Tree (Join-Path $chapter2 'baselines\ug_cem_apt') (Join-Path $methodDirectory 'code\ug_cem_apt')
            Copy-Tree (Join-Path $repo 'mbrl-lib-uncertainty_guided_planning\uncertainty_guided_planning') (Join-Path $methodDirectory 'code\upstream\uncertainty_guided_planning')
        }
    } else {
        Copy-Files $oursFiles (Join-Path $methodDirectory 'code')
    }
    if ($method.config) {
        Copy-Files @((Join-Path $chapter2 "configs\formal_v3\methods\$($method.config)")) (Join-Path $methodDirectory 'configs')
    }
    if ($method.launcher) {
        Copy-Files @((Join-Path $chapter2 "scripts\$($method.launcher)")) (Join-Path $methodDirectory 'launchers')
    }
    Write-MethodReadme -MethodDirectory $methodDirectory -Method $method
    Write-MethodBindings -MethodDirectory $methodDirectory -Method $method
    $sourcePath = Join-Path $chapter2 $method.result
    $sourceSummary = Copy-CanonicalMetadata -Source $sourcePath -Destination (Join-Path $methodDirectory 'manifests\canonical') -Active ([bool]$method.active)
    Add-ResultSnapshot -MethodDirectory $methodDirectory -Source $sourcePath -Status $method.status -Canonical ([bool]$method.canonical) -Note $method.note -Active ([bool]$method.active)
    Write-Json (Join-Path $methodDirectory 'manifests\source_metadata_summary.json') $sourceSummary
}

$table1Alias = Join-Path $catalog 'table1\lwm_rl'
foreach ($part in @('code','configs','launchers','results','manifests','docs')) { Ensure-Directory (Join-Path $table1Alias $part) }
$table1AliasInfo = [ordered]@{ table='table1'; name='lwm_rl'; baseline=''; config=''; launcher=''; result='outputs\formal_v3\table2\lwm_rl_trueppo_v3\repeat_1'; status='PROVISIONAL_ALIAS'; canonical=$false; active=$false; alias_of='..\..\table2\lwm_rl'; note='Table 1 logical alias of the Table 2 LWM-RL physical run; no duplicate run.' }
Write-MethodReadme -MethodDirectory $table1Alias -Method $table1AliasInfo -Alias $true
Write-MethodBindings -MethodDirectory $table1Alias -Method $table1AliasInfo -Alias $true
Write-Text (Join-Path $table1Alias 'code\ALIAS_CODE_BINDING.md') 'Logical alias only: implementation code is inherited from table2/lwm_rl; no second executable is maintained here.'
Copy-CanonicalMetadata -Source (Join-Path $chapter2 $table1AliasInfo.result) -Destination (Join-Path $table1Alias 'manifests\canonical') -Active $false | Out-Null
Add-ResultSnapshot -MethodDirectory $table1Alias -Source (Join-Path $chapter2 $table1AliasInfo.result) -Status 'PROVISIONAL_ALIAS' -Canonical $false -AliasOf '..\..\table2\lwm_rl' -Note $table1AliasInfo.note

$fullReward = Join-Path $catalog 'table3\full_reward_alias'
foreach ($part in @('code','configs','launchers','results','manifests','docs')) { Ensure-Directory (Join-Path $fullReward $part) }
$fullRewardInfo = [ordered]@{ table='table3'; name='full_reward_alias'; baseline=''; config=''; launcher=''; result='outputs\formal_v3\table2\lwm_rl_trueppo_v3\repeat_1'; status='PROVISIONAL_ALIAS'; canonical=$false; active=$false; alias_of='..\..\table2\lwm_rl'; note='Full-Reward is the Table 2 LWM-RL physical run; no separate execution or copied raw results.' }
Write-MethodReadme -MethodDirectory $fullReward -Method $fullRewardInfo -Alias $true
Write-MethodBindings -MethodDirectory $fullReward -Method $fullRewardInfo -Alias $true
Write-Text (Join-Path $fullReward 'code\ALIAS_CODE_BINDING.md') 'Logical alias only: implementation code is inherited from table2/lwm_rl; no second executable is maintained here.'
Copy-CanonicalMetadata -Source (Join-Path $chapter2 $fullRewardInfo.result) -Destination (Join-Path $fullReward 'manifests\canonical') -Active $false | Out-Null
Add-ResultSnapshot -MethodDirectory $fullReward -Source (Join-Path $chapter2 $fullRewardInfo.result) -Status 'PROVISIONAL_ALIAS' -Canonical $false -AliasOf '..\..\table2\lwm_rl' -Note $fullRewardInfo.note

# Machine-readable coverage and provenance indexes make omissions visible when
# the catalog is rebuilt after a run changes state.
$indexEntries = @()
$allCatalogMethods = @($methods) + @($table1AliasInfo) + @($fullRewardInfo)
foreach ($entry in $allCatalogMethods) {
    $entryDir = Join-Path $catalog "$($entry.table)\$($entry.name)"
    $parts = [ordered]@{}
    $missing = @()
    foreach ($part in @('code','configs','launchers','manifests','results','docs')) {
        $partDir = Join-Path $entryDir $part
        $partFiles = if (Test-Path -LiteralPath $partDir) { @(Get-ChildItem -LiteralPath $partDir -File -Recurse -Force) } else { @() }
        $parts[$part] = [ordered]@{ present = (Test-Path -LiteralPath $partDir); file_count = $partFiles.Count; bytes = [int64](($partFiles | Measure-Object -Property Length -Sum).Sum) }
        if ($partFiles.Count -eq 0) { $missing += $part }
    }
    $hashFile = Join-Path $entryDir 'manifests\canonical\source_metadata_hashes.json'
    $hashes = @()
    if (Test-Path -LiteralPath $hashFile) {
        try { $hashes = @((Get-Content -LiteralPath $hashFile -Raw | ConvertFrom-Json).copied_files | Where-Object { $_.sha256 } | ForEach-Object { [ordered]@{ relative_path = $_.relative_path; bytes = $_.bytes; sha256 = $_.sha256 } }) } catch { $hashes = @() }
    }
    $reason = if ($missing.Count -gt 0) { if ($entry.active) { 'Active writer: compact source binding retained while result remains immutable.' } else { 'Catalog rebuild should restore non-empty required categories.' } } elseif ($entry.active -and $hashes.Count -eq 0) { 'Active writer: source SHA sidecars intentionally deferred until the lock is released.' } else { '' }
    $indexEntries += [ordered]@{
        table = $entry.table
        method = $entry.name
        source = $entry.result
        status = $entry.status
        canonical = [bool]$entry.canonical
        alias_of = $entry.alias_of
        missing_reason = $reason
        required_parts = $parts
        sha256 = $hashes
    }
}
Write-Json (Join-Path $catalog 'method_index.json') ([ordered]@{
    schema_version = 1
    generated_at = (Get-Date).ToUniversalTime().ToString('o')
    required_categories = @('code','configs','launchers','manifests','results','docs')
    methods = $indexEntries
})

$oversized = @(Get-ChildItem -LiteralPath $catalog -File -Recurse -Force | Where-Object { $_.Length -gt 10MB } | ForEach-Object {
    [ordered]@{ path = [IO.Path]::GetRelativePath($catalog, $_.FullName); bytes = [int64]$_.Length }
})
Write-Json (Join-Path $catalog 'coverage_report.json') ([ordered]@{
    schema_version = 1
    generated_at = (Get-Date).ToUniversalTime().ToString('o')
    table_counts = [ordered]@{ table1 = 7; table2 = 4; table3 = 3; total = 14 }
    required_categories = @('code','configs','launchers','manifests','results','docs')
    all_methods_have_all_categories = (@($indexEntries | Where-Object { @($_.required_parts.Values | Where-Object { -not $_.present -or $_.file_count -eq 0 }).Count -gt 0 }).Count -eq 0)
    oversized_files = $oversized
    entries = $indexEntries
})

$catalogManifest = [ordered]@{
    schema_version = 1
    generated_at = (Get-Date).ToUniversalTime().ToString('o')
    repository_root = $repo
    git_head = (git -C $repo rev-parse HEAD).Trim()
    tables = [ordered]@{
        table1 = @('uamcts_cc4_adapted','rsmbrl_cc4','carl_cc4_adapted','dca_cc4_adapted','priorrl_ppo_cc4','terla_a4','lwm_rl')
        table2 = @('rl_only','llm_rl','wm_rl','lwm_rl')
        table3 = @('delay_only','fail_only','full_reward_alias')
    }
    policy = [ordered]@{
        repeat_scope = 'repeat1 sprint; DCA immutable five-repeat exception'
        raw_results = 'kept at canonical source paths; catalog stores metadata and pointers'
        admission = 'only eligibility PASS outputs may enter final tables; all repeat1-only rows remain PROVISIONAL'
    }
}
Write-Json (Join-Path $catalog 'catalog_manifest.json') $catalogManifest

Write-Host "Experiment catalog generated at $catalog"
