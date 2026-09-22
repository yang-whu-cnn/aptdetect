# RSMBRL-CC4

CC4 reproduction of Webster and Flach, *Risk Sensitive Model-Based Reinforcement Learning using Uncertainty Guided Planning* (NeurIPS 2021).

- Preserved: PETS-style ensemble planning, categorical CEM, and the paper objective `sum reward - beta * epistemic uncertainty`.
- Adapted: continuous controls become A4 categorical sequences; horizon is H=4; the frozen shared five-member world model and final-reward predictor are used.
- Paper defaults retained where applicable: 5 CEM iterations, population 200, elite ratio 0.3, and alpha 0.1. The paper's 12-particle TS-infinity propagation is **not implemented** by the shared deterministic ensemble evaluator; the resolved config marks this difference explicitly instead of exposing an unused parameter. This is a CC4 reproduction, not a bit-level reproduction.

## Local artifact gate

Run `python -m baselines.rsmbrl_cc4.artifact_preflight` before constructing a
planner. The gate verifies the final-reward manifest and checkpoint hashes,
D27/A4, absolute target mode, five ensemble members, H=4, Full-Reward model
normalizers, and a read-only normalizer bundle produced only from calibration
seeds 3000..3007. The historical `outputs/ug_cem_v2/step6/` online bundle is
not eligible because it was bound to old model paths and enables evaluation-time
EMA updates. The required RSMBRL artifact path is
`outputs/rsmbrl_cc4/calibration/rsmbrl_normalizers_frozen.pt`.
- Formal runtime also requires the immutable
  `rsmbrl_normalizers_frozen.sidecar.json`. It binds the normalizer, shared
  world/reward checkpoints, exact train (1000..1031) and validation
  (2000..2007) replay files and hashes, bundle schema, and a clean Git state.
  The replay files are audited for exact seed coverage, so any test-seed row,
  missing sidecar, missing field, or SHA mismatch fails closed. The sidecar's
  `formal_result_eligible=false` describes the calibration artifact itself;
  only the separately validated repeat output may become a formal result.
- The frozen normalizer bundle is format v2 and is cryptographically bound to
  the shared D27 semantic projection version/SHA. Pre-projection format-v1
  bundles fail closed. Calibration rollouts use root tick and immutable root
  context from calibration-only seeds 3000..3007.
- `beta=0.1` remains the paper-fixed value and is not selected or tuned on the
  calibration split; the bundle records `paper_fixed_no_validation_tuning`.
- Evaluation mode never updates uncertainty normalization. Only the first action is executed before replanning.
- `beta=0` is diagnostic CEM and is not a Table 1 method.

Source paper SHA256: `df594e5b3d2bd36e06d94d96e3cde0c549fa1894a9fe5d26ecd6c4552c91d4d7`.
Upstream: `sradicwebster/mbrl-lib`, uncertainty-guided branch, commit `9f97859594f2b0547e01193a8758936090b0b2ec`.

## Development pilot (2026-09-19)

The CPU pilot used policy seeds 61001/61002 and the same episode seeds as the DCA pilot (3200..3209). All 10 episodes completed exactly 100 post-reset environment transitions and ended at controller tick 100. The run produced 3,978 decisions: requested/executed counts were Sleep 3,091, Analyse 783, Remove 85, Restore 19, with zero fallback and zero errors. The root action mask was recorded for every decision.

Both repeat-level normalizer hashes were bitwise identical before and after evaluation for every Blue agent. Planning consumed 276.57 seconds and episode wall time totalled 295.58 seconds on CPU. The report is development/provisional only and sets `formal_result_eligible=false`; it is stored under `outputs/rsmbrl_cc4/pilot_2x5x100/` with resolved configuration, decision and episode JSONL, artifact hashes, and output hashes.
