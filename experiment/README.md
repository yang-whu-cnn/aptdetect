# APTDetect formal experiment catalog

This directory is the compact, auditable entry point for the Chapter 2 CC4 v3
experiments. It groups the production code, frozen configuration, launcher, and
result metadata by comparison method.

## Layout

- `_shared/`: common runners, evaluation/admission code, tests, formal configs,
  requirements, manifests, and mandatory experiment documents.
- `table1/`: UAMCTS, RSMBRL, CARL, DCA, PriorRL, TERLA, plus the logical LWM-RL
  alias used by Table 1.
- `table2/`: RL-Only, LLM-RL, WM-RL, and LWM-RL.
- `table3/`: Delay-Only, Fail-Only, and the Full-Reward alias.

Each method has `code/`, `configs/`, `launchers/`, `results/`, `manifests/`, and
`docs/`. `results/result_pointer.json` records the canonical source, status,
file inventory, and byte count. Small immutable result metadata is copied under
`results/metadata/`; large trajectories, journals, and checkpoints remain at
their canonical source paths and are not duplicated here.

## Safety rules

- DCA repeat 1–5 is immutable and must never be overwritten or rerun.
- A single-repeat PASS is still labelled `PROVISIONAL` until the full formal
  repeat requirement is met.
- Table 1 LWM-RL and Table 3 Full-Reward are aliases of the Table 2 LWM-RL
  physical run; they must not trigger separate executions.
- `RUNNING_TEST_DO_NOT_MOVE` denotes an active writer. Do not copy, move,
  truncate, or clean its source directory until the writer releases its lock.
- The canonical source trees under `chapter2_region_detection/` remain the
  execution source of truth. Rebuild this catalog with
  `chapter2_region_detection/scripts/build_experiment_catalog.ps1` after a
  formal run reaches a new stable state.
