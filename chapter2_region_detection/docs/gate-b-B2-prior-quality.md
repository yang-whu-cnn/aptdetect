# Gate B2 — Formal Multi-LLM Prior Quality

日期：2026-09-17  
状态：**SOURCE READY / LOCAL TEST + FORMAL 240-STATE RUN PENDING**

---

## 1. Preconditions

已关闭：

```text
B0.1 frozen WM final audit          PASS / CLOSED
B2.2 candidate registry             PASS / CLOSED
B2.3 validation state bank          PASS / CLOSED
B2.5 persistent cache               PASS / CLOSED
B2 live three-model capability      PASS / CLOSED
```

Frozen bank：

```text
records = 240
unique exact D27 = 240
bank_sha256 = b42472e9a20f66b56164d846e8bd41caa2cae54f0e9a6558d7ebebff2c771e17
```

Primary LLM remains unselected.

## 2. Formal variants

```text
Tier-H: openai/gpt-5.6-sol
Tier-M: openai/gpt-5.4-mini
Tier-L: google/gemini-3.5-flash-lite
Uniform: deterministic state-independent K=6 plan sample, prior=1/6
```

All LLM variants share：

```text
OFOX gateway
same 240 validation states
same prompt
A=4 / K=6 / H=4
temperature=0.2
json_schema request
same semantic parser/fallback
same frozen WM/reward evaluator
```

No model-specific prompt tuning or different K/H.

## 3. Source

```text
formal_experiments/evaluation/run_b2_prior_quality.py
tests/test_gate_b2_prior_quality.py
```

Shared evidence models：

```text
outputs/world_model_v2/a4_5b/world_model_absolute.pt
outputs/world_model_v2/a4_5c/response_reward_predictor.pt
```

Cache：

```text
outputs/lwm_rl_v2/prior_cache/validation/<model_alias>/...
```

## 4. Provider transaction semantics

For each cache miss：

```text
current D27 state
-> same formal prompt
-> exact registry model
-> temperature=0.2
-> json_schema
-> bounded retry only for rate-limit/network/5xx
-> raw response contract audit
-> frozen semantic parser
-> PriorBatch K6/H4
-> cache
```

Provider failures are not cached.

A successful API response that is semantically/schema invalid is not repeatedly regenerated without bound. Its raw quality is recorded, then deterministic parser fallback may complete the final K6 PriorBatch. Therefore raw-schema quality and final PriorBatch validity remain separate metrics.

Raw response itself is not stored; prompt/response SHA256 are stored.

## 5. Cache/resume behavior

Formal run is resumable：

```text
valid cache hit -> no provider call
miss            -> one logical provider transaction
interrupted run -> rerun; completed keys remain hits
```

Changing model, registry hash, prompt, temperature, split or exact D27 state creates a different key.

## 6. Reliability metrics

Per model：

- completed states / 240；
- API success/completion rate；
- raw schema-valid response rate；
- semantic-valid candidates before fallback；
- duplicate candidate rate；
- fallback candidate rate；
- retry-state rate；
- failure count；
- cache hits/misses；
- live API calls this run。

Formal Gate completion requires all 240 states to produce a valid final PriorBatch. Raw schema/fallback values are experimental outcomes, not thresholds used to remove a model after seeing results.

## 7. Diversity metrics

- unique plan ratio；
- per-position action distribution；
- per-position normalized action entropy；
- global A4 action coverage；
- all-no-op plan prevalence；
- all-restore plan prevalence。

## 8. Frozen-WM candidate-quality metrics

Every state/variant uses the same `SharedRolloutEvaluator`：

```text
M=5 frozen absolute WM
H=4 fixed-member deterministic mean rollout
gamma_tick=0.99
same response reward predictor
```

Report：

- top-prior predicted return；
- top-prior rank under predicted value；
- best-of-K predicted return；
- mean candidate predicted return；
- candidate value spread；
- mean/max ensemble uncertainty；
- prior preference vs predicted value Spearman。

These are model-based planning diagnostics. They must not be described as real-environment superiority.

Uniform has tied prior=1/6, so prior-value Spearman is explicitly undefined rather than forced to zero.

## 9. Efficiency metrics

From successful live response/cache metadata：

- latency p50/p95；
- prompt tokens；
- completion tokens；
- total estimated USD under frozen registry pricing snapshot；
- retry count；
- cache hit/miss。

Cost is an estimate under the frozen pricing snapshot, not the billing statement.

Using the single-state preflight token counts as a rough planning estimate for a fresh 240-state run gives approximately：

```text
Tier-H ≈ USD 2.91
Tier-M ≈ USD 0.40
Tier-L ≈ USD 0.28
-------------------
3-model total ≈ USD 3.59
```

Actual cost will differ with output length and provider accounting. Cache reruns do not intentionally repeat successful requests.

## 10. Outputs

```text
outputs/lwm_rl_v2/b2/llm_h_gpt56_sol/prior_quality.json
outputs/lwm_rl_v2/b2/llm_m_gpt54_mini/prior_quality.json
outputs/lwm_rl_v2/b2/llm_l_gemini35_flash_lite/prior_quality.json
outputs/lwm_rl_v2/b2/uniform_non_llm/prior_quality.json
outputs/lwm_rl_v2/b2/prior_quality_summary.json
```

Reports contain parsed plans, prior preferences, frozen-WM values/uncertainties, state hashes and non-secret API metadata; they do not contain API keys or raw provider responses.

## 11. Unit acceptance

Current prior-quality test suite：

```text
tests/test_gate_b2_prior_quality.py
```

10 tests cover：

1. verified registry + no primary selection；
2. frozen pricing cost calculation；
3. Spearman/tie behavior；
4. provider metadata + no raw-response serialization；
5. raw-invalid API success remains auditable with fallback；
6. bounded retry；
7. WM value/uncertainty/rank extraction；
8. cache hit constructs no provider client；
9. deterministic zero-API uniform baseline；
10. raw schema and fallback rates remain separate。

Because strict `additionalProperties=false` validation was strengthened after the first live preflight, also rerun：

```text
tests/test_gate_b2_multi_model_preflight.py
```

which is now 9 tests.

## 12. Formal PASS

Prior-quality generation phase PASS requires：

```text
all 3 LLM completed_states = 240
all 3 final PriorBatch valid
uniform completed_states = 240
no provider failure hidden as fallback
registry hash recorded
bank hash exactly frozen value
same WM/reward evaluator
no calibration/test data
primary_model_selected = false
```

No requirement that a particular LLM ranks first.

After PASS：

```text
30-state × 3 independent-generation repeatability
```

Only after repeatability closes does B2 complete and proceed to B3 regression/PPO preparation.
