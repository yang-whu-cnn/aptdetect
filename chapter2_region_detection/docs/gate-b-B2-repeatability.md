# Gate B2.6 — Multi-LLM Repeatability

日期：2026-09-17  
状态：**PASS / CLOSED**

---

## 1. Purpose

Measure stochastic stability of the frozen three-model prior protocol without using repeatability outcomes to tune prompts, temperature, K, H or model-specific settings.

Repeatability is descriptive evidence only. It does not select the final primary model.

## 2. Frozen input subset

Source：the already frozen 240-state validation bank.

```text
source_bank_sha256 = b42472e9a20f66b56164d846e8bd41caa2cae54f0e9a6558d7ebebff2c771e17
state_count = 30
replicates_per_state = 3
subset_sha256 = be063e3de7b19eee6da7728ce3b2365b10d73806112128d9be048014ad58ae49
```

Deterministic subset constraints：

```text
30 exact-unique D27 states
5 agents × 6 states
feature17 = 15 zero / 15 one
8 validation seeds covered
no hidden/reward/future/test fields
```

## 3. Independent-generation semantics

For every `(model,state)`：

```text
replicate 0
replicate 1
replicate 2
```

are three distinct cache identities and therefore three distinct live provider transactions on the first formal run.

Repeatability uses a dedicated cache namespace：

```text
outputs/lwm_rl_v2/prior_cache_repeatability/
```

It does not reuse the 240-state prior-quality cache as replicate evidence. Later reruns may hit the same replicate-specific keys to avoid duplicate billing.

## 4. Metrics

Per state/model：

- pairwise candidate-set Jaccard；
- pairwise candidate overlap count；
- top-prior pair agreement；
- all-three top-prior agreement；
- top-prior preference variance；
- matched-plan preference variance；
- best-of-K frozen-WM value variance/range。

Reliability/efficiency：

- schema-valid generation rate；
- semantic-valid candidate rate；
- duplicate/fallback/retry rate；
- live calls；
- latency；
- estimated API cost。

## 5. Unit acceptance

User local execution：

```text
tests.test_gate_b2_repeatability
Ran 10 tests
OK
```

The tests cover deterministic balanced subset selection, exact replicate identity isolation, overlap/value-variance metrics, first-run three-live-call semantics, rerun cache hits, schema/fallback auditability, provider-failure handling and primary-model non-selection.

## 6. Formal reliability result

All three models：

```text
completed_states = 30/30
successful_generations = 90/90
failed_generations = 0
schema_valid_generation_rate = 1.0
semantic_valid_candidate_rate = 1.0
duplicate_candidate_rate = 0.0
fallback_candidate_rate = 0.0
retry_generation_rate = 0.0
```

## 7. Repeatability result

```text
Variant                    set Jaccard   overlap count   top pair agree   all-3 agree   best-of-K variance
---------------------------------------------------------------------------------------------------------
GPT-5.6 Sol                  0.434055       3.366667        0.733333        0.633333        0.134165
GPT-5.4 Mini                 0.293252       2.433333        0.500000        0.400000        0.918567
Gemini 3.5 Flash Lite       0.411077       3.277778        0.677778        0.533333        1.402401
```

Additional preference stability：

```text
GPT-5.6 Sol matched-plan preference variance mean      = 0.0005906624
GPT-5.4 Mini matched-plan preference variance mean     = 0.0005149787
Gemini 3.5 Flash Lite matched-plan preference variance = 0.0005084568
```

Interpretation：GPT-5.6 Sol is the most repeatable of the three under this protocol on set overlap, top-prior agreement and frozen-WM best-of-K value stability. GPT-5.4 Mini has the weakest candidate-set overlap/top agreement, while Gemini has larger best-of-K value variation despite moderate set overlap.

These are validation repeatability diagnostics, not real-environment performance claims.

## 8. Efficiency result

```text
Variant                    p50 latency   p95 latency   live calls   estimated cost
----------------------------------------------------------------------------------
GPT-5.6 Sol                  5.390 s       7.402 s        90        USD 1.133325
GPT-5.4 Mini                 2.833 s       3.960 s        90        USD 0.162754
Gemini 3.5 Flash Lite       2.279 s       2.728 s        90        USD 0.109167
```

## 9. Artifacts

```text
formal_experiments/evaluation/run_b2_repeatability.py
tests/test_gate_b2_repeatability.py
outputs/lwm_rl_v2/b2/repeatability/repeatability_summary.json
outputs/lwm_rl_v2/b2/repeatability/<model_alias>.json
```

## 10. Formal PASS

```text
30-state deterministic subset               PASS
3 independent generations/state/model       PASS
3 models × 90 generations                    PASS
schema/semantic validity                     PASS
provider failure hidden as fallback          NO
same frozen WM/reward evidence               PASS
primary_model_selected                       FALSE
```

**FINAL STATUS: PASS / CLOSED**

This closes Gate B2 as a whole. Next: Gate B3 provider-neutral posterior representation regression.
