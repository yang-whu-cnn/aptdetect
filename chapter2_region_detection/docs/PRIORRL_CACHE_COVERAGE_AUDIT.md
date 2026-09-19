# PriorRL exact-cache coverage audit

Date: 2026-09-19. This was a read-only audit; provider calls = 0 and no training was run.

## Finding

The taskbook's approximate 2,000 representative-state cache is incompatible with a runtime that requires exact float32 D27 + agent identity and fails closed on every cache miss. The selected set is useful as a stratified teacher sample, but it is not a runnable exact cache for online PPO.

Measured against the frozen deterministic selection:

- Train replay: 2,488/43,297 decision rows covered (5.746%); 2,000/41,079 unique `(agent, exact D27)` states covered (4.869%).
- Final validation replay: 139/10,796 rows have the same `(agent, exact D27)` as a selected train state (1.288%); unique coverage is 1.227%. These are semantic overlaps only: the cache identity includes `split`, so a train entry still cannot satisfy a validation lookup.
- Legacy 240-state validation bank: 16/240 semantic overlaps (6.667%). All feature-17=1 cells have zero overlap. Again, split isolation prevents reuse.
- Compatible entries currently present under `outputs/lwm_rl_final_20260917/prior_cache_final`: 0/2,000 selected train states and 0 compatible final-validation states. Existing files use legacy validation identity/provenance and cannot be silently reused.

Coverage was computed per agent/action/evidence cell in `outputs/priorrl_cc4/cache_coverage_audit.json`. Most populated train cells cover roughly 3.5%-7%; unusually high rates for `blue_agent_4`, evidence=0 reflect very small exhausted cells, not broad coverage.

An online PPO rollout changes actions and therefore visits D27 byte patterns not present in the behavior replay. The replay coverage above is an optimistic upper bound, not a guarantee. Exact lookup would almost certainly fail during initial collection and cannot satisfy the development pilot gate.

## Options requiring an explicit protocol decision

### A. Exact pre-generation

Generate every exact train state and separate validation/test entries. Train alone requires 41,079 unique teacher calls rather than 2,000. Final validation has roughly ten thousand additional unique states, and online PPO/test can still create unseen states. This is expensive and still does not close the online state space; generating test priors after observing test states also requires a carefully predeclared, label-blind workflow to avoid adaptive test leakage.

### B. Frozen train-only retrieval (recommended minimum adaptation)

Treat the 2,000 cached states as prototypes. Freeze, before validation/test:

1. a train-only feature scaler;
2. agent-local distance metric and feature weights;
3. deterministic nearest-prototype tie-break;
4. maximum admissible distance/radius;
5. retrieval version and prototype-set SHA256.

At runtime, map D27 to one prototype, then perform an exact read of that prototype's cache entry. Missing prototype entries or distances above the frozen radius still fail closed. This preserves offline-only operation but changes `p_LLM(a|s)` to an explicitly disclosed approximation `p_LLM(a|prototype(s))`. Validation may measure coverage and choose the radius; final test must not alter it. Main risks are aliasing states with materially different incident evidence and agent/action imbalance. A leakage test must prove that fitting uses train only and that validation/test merely query the frozen index.

### C. Frozen semantic quantization

Define an auditable D27 binning scheme from train/calibration only and generate one prior per `(agent, quantized state)`. This gives stable keys and bounded cache size, but bin edges can erase meaningful distinctions and require regeneration because the prompt/provenance must state the quantized representation. As with retrieval, out-of-domain bins fail closed.

## Recommendation

PriorRL remains blocked pending an explicit choice between B and C. B is the smallest engineering change and reuses the 2,000 prototypes, but it is a substantive method adaptation and should not be implemented silently. Exact expansion (A) is not a practical solution for online PPO and does not by itself eliminate misses.
