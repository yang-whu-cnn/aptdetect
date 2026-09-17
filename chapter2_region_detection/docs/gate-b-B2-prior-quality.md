# Gate B2 — Formal Multi-LLM Prior Quality

日期：2026-09-17  
状态：**PASS / CLOSED**

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

Frozen validation bank：

```text
records = 240
unique exact D27 = 240
bank_sha256 = b42472e9a20f66b56164d846e8bd41caa2cae54f0e9a6558d7ebebff2c771e17
feature17 = 118 zero / 122 one
```

Primary LLM remains unselected.

## 2. Formal variants

```text
Tier-H: openai/gpt-5.6-sol
Tier-M: openai/gpt-5.4-mini
Tier-L: google/gemini-3.5-flash-lite
Uniform: deterministic state-independent K=6 plan sample, prior=1/6
```

All LLM variants shared：

```text
OFOX gateway
same 240 validation states
same prompt
A=4 / K=6 / H=4
temperature=0.2
json_schema request
same semantic parser/fallback
same frozen M=5 WM
same frozen response-reward predictor
```

No model-specific prompt tuning or different K/H.

## 3. Local/unit acceptance

Before the formal run the user executed：

```text
tests.test_gate_b2_multi_model_preflight : 9/9 PASS
tests.test_gate_b2_prior_quality         : 10/10 PASS
-----------------------------------------------
total                                    : 19/19 PASS
```

## 4. Formal reliability result

All three exact LLMs completed all states：

```text
completed_states              = 240/240
schema_valid_response_rate    = 1.0000
semantic_valid_candidate_rate = 1.0000
fallback_candidate_rate       = 0.0000
duplicate_candidate_rate      = 0.0000
failure_count                 = 0
```

Uniform completed 240/240 without API calls.

Therefore no provider failure was hidden by parser fallback.

## 5. Frozen-WM prior-quality result

Higher predicted response return is better (values are negative here).

```text
Variant                         top-prior value   best-of-6 value   prior/value Spearman   top-prior rank
-------------------------------------------------------------------------------------------------------
GPT-5.6 Sol                        -9.100476          -8.511746              0.064325             2.9500
GPT-5.4 Mini                       -9.974312          -8.793619             -0.050331             3.3708
Gemini 3.5 Flash Lite             -11.154970         -8.685761             -0.192815             3.5125
Uniform non-LLM                    -9.728248          -8.222777                 N/A              3.4333
```

Interpretation boundary：

1. GPT-5.6 Sol has the strongest alignment between its own prior ranking and the frozen-WM ranking among the three LLMs, but the correlation is still weak.
2. GPT-5.4 Mini and Gemini 3.5 Flash Lite show negative mean prior/value Spearman on this frozen validation bank.
3. Uniform attains the highest `best-of-6` frozen-WM value in this experiment. This means its fixed six-plan candidate set covers high-WM-value plans well; it does **not** establish superior real-CC4 performance.
4. These results support keeping WM foresight and PPO posterior correction separate from the raw LLM prior.
5. Prior-quality alone must not select the final primary LLM.

## 6. Diversity observations

All LLM variants maintained：

```text
unique_plan_ratio_mean = 1.0
global action coverage = all A4 actions
all_restore_plan_rate  = 0
```

Per-position action distributions differ materially by model and are retained in the formal JSON report for paper analysis.

## 7. Efficiency result

```text
Variant                         p50 latency      p95 latency      estimated cost / 240 states
----------------------------------------------------------------------------------------------
GPT-5.6 Sol                     5.107 s          42.546 s         USD 4.19560
GPT-5.4 Mini                    3.426 s           8.189 s         USD 0.43378
Gemini 3.5 Flash Lite          2.411 s           3.830 s         USD 0.28615
```

Cost is the frozen registry estimate, not the provider billing statement.

The large GPT-5.6 Sol p95 tail must be retained as part of the final performance/cost/latency trade-off.

## 8. Artifacts

```text
formal_experiments/evaluation/run_b2_prior_quality.py
tests/test_gate_b2_prior_quality.py

outputs/lwm_rl_v2/b2/llm_h_gpt56_sol/prior_quality.json
outputs/lwm_rl_v2/b2/llm_m_gpt54_mini/prior_quality.json
outputs/lwm_rl_v2/b2/llm_l_gemini35_flash_lite/prior_quality.json
outputs/lwm_rl_v2/b2/uniform_non_llm/prior_quality.json
outputs/lwm_rl_v2/b2/prior_quality_summary.json
```

Formal reports contain parsed plans, prior preferences, frozen-WM values/uncertainties, state hashes and non-secret API metadata. They do not contain API keys or raw provider responses.

## 9. Formal PASS

```text
3 LLM × 240 states complete           PASS
Uniform × 240 states complete         PASS
all final PriorBatch A4/K6/H4 valid   PASS
raw schema/fallback separately audited PASS
same frozen bank/WM/reward             PASS
no calibration/test data               PASS
primary_model_selected                 FALSE
```

**FINAL STATUS: PASS / CLOSED**

Next：B2.6 repeatability must close before the whole Multi-LLM Prior Study can close.
