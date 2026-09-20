# Experiment output backup protocol

This protocol makes every stopped, partial, or completed experiment output
recoverable without allowing a backup to become a paper result by accident.
The implementation is `formal_experiments.evaluation.backup_outputs` and uses
only Python's standard library.

## Boundaries and status gates

- A source must be a strict child of
  `chapter2_region_detection/outputs`. The outputs directory itself is not a
  valid source. Every component and every entry in the tree is checked and
  symlinks, junctions, and Windows reparse points are rejected.
- The only permitted backup root is `C:\aptdetect_experiment_backups`.
  `--backup-root` exists for explicit selection of that root and for isolated
  tests; an arbitrary destination is rejected. Existing backups are never
  overwritten or deleted.
- Normally `run_status.json` must contain exactly one of `STOPPED`, `PARTIAL`,
  or `PASS` in its `status` field. Missing, unknown, and `RUNNING` statuses
  fail closed. The only fallback is a Table 2/3 train-only safe stop with an
  `orchestrator_report.json` that exactly declares `status=STOPPED`,
  `execution_status=STOPPED`, `result_state=PARTIAL`,
  `expected_safe_stop=true`, and `formal_result_eligible=false`, together with
  `exit_code.txt` equal to `3`. This fallback remains non-paper-eligible.
- A `PASS` source must contain `manifest.json` with
  `formal_result_eligible: true` and `eligibility_report.json` with
  `passed: true`. `STOPPED` and `PARTIAL` backups always record
  `formal_result_eligible: false`, even if an input file claims otherwise.

## Commands

From the repository root:

```text
python -m chapter2_region_detection.formal_experiments.evaluation.backup_outputs dry-run --source chapter2_region_detection/outputs/<run>
python -m chapter2_region_detection.formal_experiments.evaluation.backup_outputs create --source chapter2_region_detection/outputs/<run>
python -m chapter2_region_detection.formal_experiments.evaluation.backup_outputs verify --backup-dir C:\aptdetect_experiment_backups\<name>.finalized
python -m chapter2_region_detection.formal_experiments.evaluation.backup_outputs restore --backup-dir C:\aptdetect_experiment_backups\<name>.finalized --restore-target chapter2_region_detection/outputs/<new-run>
```

`--run-id <directory-name>` may replace `--source`; it resolves only to a
single child of `outputs`. `--expect-status` adds an explicit status assertion.
`--max-bytes` rejects a source larger than the supplied limit. `--backup-dir`
on `create` selects a name under the backup root; the tool appends
`.finalized` (or changes a supplied `.partial` suffix) and refuses a collision.

`dry-run` reads and hashes the source and checks the space gate but does not
create the backup root or any output. `verify` checks the sidecar, manifest,
all payload paths, counts, byte totals, and SHA-256 values. `restore` requires
a target that does not exist and is a strict child of `outputs`.

## Atomic copy and audit record

Creation uses a unique `<id>.partial/payload` directory. The source is
snapshotted before copying, the payload is snapshotted after copying, and the
source is snapshotted again. Any source mutation or payload mismatch fails
closed. A failed `.partial` directory is retained as evidence for main-window
review; the tool never deletes or cleans it automatically. The final manifest
records, for both source snapshots and the payload,
the relative path of every file, file count, total bytes, and SHA-256 digest.

The staging directory receives `backup_manifest.json` and its SHA-256 sidecar;
both are flushed with `fsync` before a same-volume non-overwriting rename to
`<id>.finalized`. A low-space hard gate rejects free space below
`source_bytes + 2 GiB`. Free space below 20 GiB or 15% of the volume is only a
warning and is recorded in the manifest; the tool never performs cleanup.

Restore copies into `<target>.restore.partial`, fully verifies the copied
payload, and then performs a non-overwriting rename. It never replaces an
existing target. A failed restore staging directory is likewise retained for
main-window review and is never automatically deleted.

## Recovery and paper eligibility

Use `verify` before consuming a backup. A `STOPPED` or `PARTIAL` backup is an
audit/recovery artifact only and must not be entered into a paper table. Only
a `PASS` backup whose two source gates were satisfied at creation is marked
paper-eligible; verification rechecks that invariant.
