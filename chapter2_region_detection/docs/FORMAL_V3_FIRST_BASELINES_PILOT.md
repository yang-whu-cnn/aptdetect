# Formal v3 first-baseline pilot readiness

Scope: DCA-CC4 (adapted), RSMBRL-CC4, and PriorRL-PPO-CC4. This work does not change the W0 evaluator, `.venv`, or the prior-cache builder.

## Frozen pilot protocol

- CC4 / `FiniteStateRedAgent`, D27 or equivalent Blue-visible evidence, A4, durations 1/2/3/5.
- Development pilot only: 2 independent repeats x 5 episodes x 100 ticks. It must not be reported as a paper result.
- Formal run remains 5 repeats x 100 episodes x 500 ticks and is blocked until W0 and method-specific prerequisites pass.
- No online LLM calls. Test/calibration separation and read-only evaluation state are mandatory.

## Method audit

### DCA-CC4 (adapted)

The paper defines alert DBSCAN/temporal sequencing, abductive reasoning and tabular model/value iteration over attacker actions. It is not PPO and its actions are not defender responses. The CC4 adapter implements deterministic mixed-distance DBSCAN, temporal ordering, a finite tabular attack-path model, value iteration and Dyna replay; path values participate in target ranking. Its tokenizer rejects common hidden-truth fields and selects targets only from observable alert hosts. The attacker reward is reserved for path ranking, not defender learning. The original power-grid abductive rule base is not implementable from CC4 observations and remains an explicitly disclosed modality adaptation rather than a preserved component.

Initial component gate: deterministic mapping smoke passed; the end-to-end wiring status is recorded below.

#### Executed development pilot (2026-09-19)

The baseline-specific runner now consumes real `FiniteStateRedAgent` Blue observations, translates only visible `Processes`/`Connections`/`Files` rows to alert tokens, and delegates every A4 action/target decision to the shared `CybORGActionAdapter`. It never reads the environment controller state or Red sessions.

- Wiring smoke: seed 3199, 1 episode x exactly 20 post-reset environment transitions; controller terminal tick 20, 78 decisions, 43 raw visible events, zero fallback, complete.
- Development pilot: seeds 3200..3209, 2 repeats x 5 episodes x exactly 100 post-reset environment transitions; all 10 episodes reached controller terminal tick 100, with 2,873 decisions and 3,234 raw visible events.
- Requested A4 counts: no-op 929, analyse 1,763, remove 168, restore 13.
- Executed counts: Sleep 929, Analyse 1,763, Remove 168, Restore 13. No fallback or runtime exception occurred in the corrected exact-transition run.
- Both reports and every JSONL record set `formal_result_eligible=false`. The audit bundles include resolved config plus decision/event line counts and SHA256 hashes under `outputs/dca_cc4_v3/`.

DCA development pilot gate: **passed for wiring, completeness, observable-input isolation, shared resolver use, audit logging, and fallback semantics**. It remains ineligible for formal paper results because this is the short development protocol and the original power-grid abductive rules remain a disclosed adaptation gap.

The runner hard-gates both `environment_steps == requested_ticks` and `controller_tick_end == requested_ticks`; CC4 is constructed with `scenario_steps=requested_ticks+1`. The earlier 99-transition artifacts were overwritten and are not valid pilot evidence.

### RSMBRL-CC4

The existing categorical CEM and uncertainty implementation are reused. The new wrapper uses the paper objective exactly: expected return minus `beta * uncertainty`; it does not inherit the historical iteration-decayed penalty. H=4, A4, ensemble=5 and evaluation-time normalization freeze are enforced. Paper CEM defaults are resolved where implemented. The paper's 12-particle TS-infinity propagation is not implemented by the shared deterministic ensemble evaluator and is explicitly recorded as an adaptation gap.

Pilot gate: deterministic objective/core smoke passes. End-to-end CC4 pilot remains blocked on resolving the calibration normalizer bundle and shared frozen model artifacts on the pilot host.

### PriorRL-PPO-CC4

The policy is D27 -> PPO A4. The six cached plan priors are collapsed by weighted first-action frequency. The loss uses forward `KL(pi || p_LLM)` and alpha zero is exactly the RL-only loss. The adapter has no generation fallback and fails closed on cache miss. The module contains no world-model dependency.

Pilot gate: deterministic loss/network/cache smoke passes. Training pilot remains blocked until the offline prior cache has sufficient exact D27/agent coverage; no API may be called to fill misses during the pilot.

## Compute planning estimate

These are planning estimates, not measured benchmarks. DCA is CPU-only and should be negligible relative to environment time. The implemented RSMBRL path evaluates approximately `population 200 x CEM iterations 5 x H4 x ensemble 5 = 20,000` ensemble-state predictions per decision; it does not multiply by the paper's unimplemented 12-particle setting. Batch rollout on one GPU is strongly recommended and is expected to dominate pilot cost. PriorRL PPO is small; environment collection and exact-cache coverage dominate. Before formal launch, measure one 100-tick episode per method and extrapolate from observed wall time; do not infer the 5x100x500 budget from unit-test timings.
