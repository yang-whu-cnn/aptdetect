# PriorRL-PPO-CC4

`development_pilot.py` currently provides the strict seed/budget orchestration,
observable-state boundary, formal reward-channel boundary, and PPO core only. It
does **not** claim a completed real CybORG pilot. The concrete CybORG episode
callbacks will be connected and run only after the frozen prototype artifact is
available; until then all pilot outputs are synthetic-test-only and
`formal_result_eligible=false`.

`cc4_callback.py` now supplies the concrete shared CC4 environment callback and
has passed a development-only 20-tick train plus 20-tick deterministic-evaluation
wiring smoke using `UniformFrozenRetriever`. This validates environment/reward/
duration/mask wiring only; it is not the approved 2 x 5 x 100 prototype pilot.

CC4 PPO adaptation of Yan et al., *Efficient Reinforcement Learning with Large Language Model Priors*.

- Preserved: the variational objective uses forward `KL(pi || p_LLM)` (paper Eq. 2/16), added to the environment-return objective.
- Adapted: D27 feeds a compact A4 PPO actor/critic. Six cached H=4 plans are collapsed to an A4 distribution from first-action weighted frequency, then epsilon-smoothed.
- Final-paper reward and duration-aware discounting are used.
- Cache miss fails closed. Training and test prohibit online LLM calls. No world model is imported or used.
- `alpha_KL=0` is the RL-Only diagnostic identity; formal alpha selection uses validation only.

## Cache-readiness blocker

The current cache identity is exact float32 D27 + public agent + split + model/prompt provenance. A read-only audit of the deterministic 2,000-state train selection found only 2,488/43,297 train replay rows covered (5.75%), 139/10,796 final validation replay rows (1.29% semantic overlap before split isolation), and 16/240 legacy validation-bank states (6.67% semantic overlap). Split isolation means train entries cannot satisfy validation/test lookups even when D27 bytes match. The currently present legacy validation cache uses a legacy agent sentinel and different generation configuration; zero entries are compatible with the new identity. Therefore online PPO with exact lookup is blocked and should be expected to fail closed almost immediately.

The exact-lookup audit motivated the explicitly approved, frozen prototype
adaptation below; no implicit nearest-neighbour or quantized fallback exists.
See `docs/PRIORRL_CACHE_COVERAGE_AUDIT.md`.

## Approved prototype retrieval adaptation

`prototype_retrieval.py` implements the approved offline adaptation: prototypes
and per-agent D27 scalers are fit from train states with verified train cache
entries only; lookup is agent-local weighted standardized L2; weights, distance
version, SHA-based tie-break and maximum-radius semantics are frozen in the
artifact. Validation states may only freeze the radius (a declared nearest
distance maximum, i.e. quantile 1.0). An initial 0.99 development gate was
rejected before producing results because it caused fail-closed misses on a
train-seed trajectory. Test/evaluation loads the immutable artifact read-only.
Cache/prototype misses and distances above the radius fail closed. This layer
returns an A4 prior and does not import or call a world model. PPO continues to
use forward `KL(pi || p_LLM)` from `policy.py`.

The first-version local source snapshot remains value-based DQN/CQL and direct
LLM-generation code. Its separately added CC4 metrics script uses a heuristic
prior plus an online linear Q scorer and hidden-truth-derived bookkeeping; it is
not an implementation suitable for the final observable-only PPO protocol.
Therefore this repository preserves the paper's prior-regularization direction
but discloses PPO/D27/A4/prototype retrieval as CC4 adaptations.

Source paper SHA256: `40ab37985422db1a439a22757f30a368b04363b8a9a9b3d59dec3488085e1c71`.
Upstream: `yanxue7/RL-LLM-Prior`, commit `13b99c9bba5462b9c84c2a433b4a5ddec046177e`; the public snapshot is value-based, so this PPO path is a disclosed adaptation.

## Formal training and evaluation gate

`formal_training.py` implements the frozen two-stage protocol.  A single tuning
initialization trains each declared alpha on train seeds 1000--1031 and ranks it
only on validation seeds 2000--2007 (mean Full-Reward, smallest-alpha tie
break).  The selected alpha is snapshotted into each of five independently
trained repeat directories.  Every training episode is 500 ticks; checkpoints
include optimizer state and can resume only with the same protocol identity.
The training manifests explicitly attest that test seeds were unused, online
LLM calls were zero, and no world model was used.

The shared Table-1 runner loads only a checkpoint, its training manifest, the
snapshotted alpha-selection artifact, and the frozen prototype artifact whose
hashes agree.  Evaluation is deterministic and read-only.  Missing or
mismatched artifacts and prototype misses fail closed.  This implementation
does not make the method formal-eligible by itself: alpha selection, five
complete checkpoints, a clean frozen code snapshot, and 5 x 100 x 500 validated
test outputs must all exist before aggregation.
