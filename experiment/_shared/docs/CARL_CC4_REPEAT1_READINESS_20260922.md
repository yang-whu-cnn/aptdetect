# CARL-CC4 (adapted) repeat-1 readiness audit (2026-09-22)

Status: **READY / TRAINING_COMPLETE; TEST_NOT_STARTED** (repeat-1 is PROVISIONAL).
The taskbook permits the disclosed adapted protocol: requested imagination
H=256, frozen-world-model validated/effective H=4, with explicit truncation
disclosure. No H=256 supplement, repeat2-5 run, or formal test was created by
this audit. The corrected hidden outer CPU train writer completed 32/32.

## Frozen identity

- Dedicated implementation worktree: `D:\w\carl`
- HEAD: `6459a142640a3e1baaa746e38a5419688b26b1a7`
- Worktree: clean dedicated CARL patch; launcher pins the post-patch HEAD and
  all CARL/evaluator SHA256 values
- Policy seed: `51001` (repeat 1)
- Training seeds: `1000..1031`, 500 ticks
- Validation seeds: `2000..2007`, 500 ticks
- Test seeds: `4000..4099`, 500 ticks (not opened by the training CLI)
- Standard CAICS-to-CC4 mapping: `alpha_comp=.025`, `beta_comp=2`,
  A4 costs `no-op/analyse=0`, `remove/restore=1`, isolate terms zero;
  normal 500-tick termination has no terminal bonus/penalty.
- Policy input: finite Blue-observable D27 only; hidden labels are offline
  reward/SCM labels and are not policy inputs.
- Training architecture: frozen five-member absolute-state WM, causal reward
  SCM/reward intervention, real buffer plus exactly 8 synthetic rollouts per
  real start, duration-aware PPO; no decision-time MCTS/CEM/model rollout.

## Adapted protocol and gates

The frozen WM has validated horizon H=4 only. The implementation requests the
paper H=256 horizon but records `ADAPTED_TRUNCATED`, effective H=4, and
`formal_result_eligible=true`, while retaining `paper_row_eligible=false` and
`repeat_status=PROVISIONAL`. This is the taskbook-disclosed adapted protocol;
it may execute the complete repeat-1 train -> validation selection -> test
chain but cannot be promoted to a five-repeat paper row. No H=256 experiment,
reward-field substitution, or hidden-truth boundary substitution is allowed.

The CARL patch uses the signed standard CAICS compromise-change term
`-beta_comp * incident_change`; therefore a clearance (`-1`) contributes the
positive `beta_comp` improvement. Regression coverage checks both the numeric
value and the SCM intervention response. The launcher binds this patched source
and rejects SHA drift.

## Artifact SHA256

| Artifact | SHA256 |
|---|---|
| `baselines/carl_cc4/formal_cli.py` (D:\w\carl) | `79978a597e00adb4e30f5675ec75b426d566ded7b85b78e45c04ae0b5fb630a3` |
| `baselines/carl_cc4/formal_training.py` (D:\w\carl) | `0075f4c81749cec4b2a371f92ba7380475fe42b452ec9e94151f489b77da0959` |
| `baselines/carl_cc4/buffers.py` (D:\w\carl) | `73828f1e86b4b911a71461c2f2e708705e3ca1100caed862a3d260311cb6e7b7` |
| `baselines/carl_cc4/scm.py` (D:\w\carl) | `f1fa6a190ba4110c74182315a29b534ea26c1bf97b1f1eb4f108bbb8ca4cea4b` |
| `baselines/carl_cc4/reward.py` (D:\w\carl) | `c32183ffeb8a678a13c9e1a162ec34bf23c1437a7c7ce12c256ffc65cfa1a4d0` |
| `baselines/carl_cc4/augmentation.py` (D:\w\carl) | `9755269f7fef5fe19d7069d0d8dca87c7098fd80e89ca55fcdef4526b3c9e104` |
| `configs/formal_v3/methods/carl_cc4.yaml` (D:\w\carl) | `79ab790f02f596bfc2975f38f44facb8d05c1ca2b3d947df3feaa791a50c5120` |
| final train replay | `ba608ed7bd63d6a6c28a259738ca35e6122bc2f0a6cf7f427df1f2b3252c8cee` |
| final validation replay | `7e23294678167078704c9603932b014d893fbd35e06b5a145259aef16c23588d` |
| frozen absolute WM | `3b86593aa8adda3e0e173bfb700c543bd88641d23cf3e73ddc3992284c8f900f` |
| `CARL.pdf` | `42e8ef7d912007a2454002300852a54d1dfffad2d9b1099958f7c8d6a84cfae1` |
| `formal_experiments/evaluation/run_method_repeat.py` (D:\w\carl) | `1896593d1102d9ee82d03529e9ebda2ccd5cf9533eaf079e6469a534956ce78c` |
| `formal_experiments/evaluation/run_method_episode.py` (D:\w\carl) | `60ca319e1c16b548bd2433041227b81531de91e04b9c32715602d8347816cdb4` |
| `formal_experiments/common/run_manifest.py` (D:\w\carl) | `12b301d390ac1f8e49112eefb21ebf80b14baba09ce2ab71152fefb05bf2e943` |
| repeat-1 launcher | `46df063b65485e6b584ff3426199826f37b595766fbc968d33d402d067fe7cc8` |

The mandatory requirements, taskbook, final handoff, replay manifest and model
manifest were also SHA-checked by the launcher against the frozen handoff
values.

## Launcher and checks

Launcher: `scripts/table1_carl_cc4_20260922_repeat1.ps1`.

```powershell
.\table1_carl_cc4_20260922_repeat1.ps1
.\table1_carl_cc4_20260922_repeat1.ps1 -Stage preflight -DryRun
.\table1_carl_cc4_20260922_repeat1.ps1 -Stage train -DryRun
.\table1_carl_cc4_20260922_repeat1.ps1 -Stage validation -DryRun
.\table1_carl_cc4_20260922_repeat1.ps1 -Stage test -DryRun
```

The inert invocation produces `INERT_PLAN`. Preflight/train dry-runs bind the
patched signed-change reward and disclosed H4 contract; validation remains
read-only and requires the training manifest; test requires the frozen
checkpoint plus validation selection and runs all 100 frozen test seeds. Real mutable paths
are fixed to:

- training target: `D:\paper\github-me\aptdetect\chapter2_region_detection\outputs\formal_v3\training\carl_cc4\repeat_01`
- launcher logs: `D:\paper\github-me\aptdetect\chapter2_region_detection\outputs\launcher_logs\table1_carl_cc4_20260922_repeat1_train_attempt2`
- formal test target (fresh retry, not created): `...\outputs\formal_v3\table1\carl_cc4\repeat_1_retry1`

Final train evidence: `training_manifest.json`, `checkpoint.pt`, and
`validation_selection.json` exist; seeds are train `1000..1031` and validation
`2000..2007`, checkpoint ticks are 500, `test_seeds_used=false`, and the
manifest is `formal_result_eligible=true`, `paper_row_eligible=false`,
`repeat_status=PROVISIONAL`. Checkpoint SHA is
`a018b472e39fe21dcc2883c84a84cf0c6ff0b61c092d72faeeb7f318d8abd3bc`, selection
SHA is `38ab430090a9de99cadc23eeacf094d551448e9842a0c19661656c6f6ae2ee21`.

Preflight resource/device evidence before patch completion: free RAM 4993 MiB,
CPU 18%, Python/Torch `2.14.0+cu130` CPU probe passed; the runner is explicitly
CPU-only and provider calls are forbidden. The live resource gate must be
rechecked immediately before the hidden outer train writer starts.
`git diff --check` passed. The first hidden launch was stopped before training
because a PowerShell automatic-variable collision passed empty CLI args; its
plan/logs are retained at the suffixed `...repeat1_outer` and unsuffixed
`...repeat1` paths. The corrected launch uses the fresh non-overwriting
`...repeat1_train_attempt2` log root. The corrected launch exited 0 and left no
CARL writer. Validation dry-run was read-only (`would_write=false`); formal
test remains explicitly unstarted.

## Targeted tests

On the pinned clean CARL worktree, the CARL-specific suite passed:

```text
47 tests, OK
tests/test_carl_cc4.py
tests/test_carl_formal_training.py
tests/test_carl_formal_cli.py
```

These tests cover signed standard reward terms, SCM intervention, seed/source
buffer isolation, exact 8:1 ratio, D27 hidden-truth isolation, frozen
evaluation policy, H4 disclosure, validation split isolation, resume identity,
frozen WM batching and CLI metadata.
