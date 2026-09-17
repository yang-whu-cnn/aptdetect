# Gate B2 — Multi-LLM Prior Study

日期：2026-09-17  
状态：**PASS / CLOSED**

---

## 1. Purpose

B2 isolates LLM prior behavior before PPO training. It does not select the final primary LLM. The final model choice must wait for end-to-end validation with independently trained PPO policies.

Frozen structural contract：

```text
planner-visible D27 only
A=4
K=6
H=4
same prompt version
same parser/fallback
same frozen WM/reward evaluator
primary_model_selected = false
```

## 2. Candidate registry — PASS / CLOSED

Canonical registry：

```text
configs/llm_model_registry_v1.yaml
```

Candidates：

```text
Tier-H  llm_h_gpt56_sol
        openai/gpt-5.6-sol

Tier-M  llm_m_gpt54_mini
        openai/gpt-5.4-mini

Tier-L  llm_l_gemini35_flash_lite
        google/gemini-3.5-flash-lite
```

All three use the same OFOX OpenAI-compatible gateway and the same formal prompt/A/K/H/parser contract.

## 3. Validation state bank — PASS / CLOSED

Frozen outputs：

```text
outputs/lwm_rl_v2/b2/state_bank.jsonl
outputs/lwm_rl_v2/b2/state_bank_summary.json
```

Real result：

```text
record_count = 240
unique_exact_state_count = 240
8/8 validation seeds covered
5/5 agents covered
6 states per seed×agent cell
feature17: 0=118, 1=122
selection_config_sha256 = 00698b76e1adeeec9bba9eb3148bea1319052584a47ae8e4771792d6f10f4117
bank_sha256 = b42472e9a20f66b56164d846e8bd41caa2cae54f0e9a6558d7ebebff2c771e17
hidden/reward/future fields = false
```

## 4. Persistent cache — PASS / CLOSED

Formal cache key isolates：

```text
split
provider/model alias/exact ID
registry hash
prompt version
A/K/H
generation config
exact D27 float32 SHA256
```

Cache hit causes zero provider calls. Failed provider transactions are never cached as successful priors. Corrupt entries are quarantined rather than silently reused.

## 5. Three-model live capability — PASS / CLOSED

On one identical frozen bank state, all three exact models passed in one attempt：

```text
api_success = true
json_schema verified = true
temperature=0.2 accepted = true
raw semantic valid = 6/6
duplicates = 0
fallback = 0
final K6/H4 valid = true
```

Therefore no per-model request special case was introduced.

## 6. Formal 240-state prior quality — PASS / CLOSED

User local regression before run：

```text
19/19 tests PASS
```

Formal results：

```text
GPT-5.6 Sol:
  completed=240/240
  schema=1.0
  fallback=0
  best_of_6_value=-8.511746
  top_prior_value=-9.100476
  prior/value Spearman=0.064325
  cost≈USD 4.19560

GPT-5.4 Mini:
  completed=240/240
  schema=1.0
  fallback=0
  best_of_6_value=-8.793619
  top_prior_value=-9.974312
  prior/value Spearman=-0.050331
  cost≈USD 0.43378

Gemini 3.5 Flash Lite:
  completed=240/240
  schema=1.0
  fallback=0
  best_of_6_value=-8.685761
  top_prior_value=-11.154970
  prior/value Spearman=-0.192815
  cost≈USD 0.28615

Uniform non-LLM:
  completed=240/240
  best_of_6_value=-8.222777
```

Interpretation：prior scores are not reliable substitutes for WM foresight. Uniform best-of-6 superiority in frozen-WM diagnostics does not establish real-environment superiority.

## 7. Repeatability — PASS / CLOSED

User local unit acceptance：

```text
tests.test_gate_b2_repeatability
10/10 PASS
```

Frozen repeatability subset：

```text
state_count = 30
replicates_per_state = 3
source_bank_sha256 = b42472e9a20f66b56164d846e8bd41caa2cae54f0e9a6558d7ebebff2c771e17
subset_sha256 = be063e3de7b19eee6da7728ce3b2365b10d73806112128d9be048014ad58ae49
5 agents × 6 states
feature17 = 15/15
8 validation seeds covered
```

Each model executed 90/90 independent generation transactions with zero failures, schema-valid rate 1.0, semantic-valid rate 1.0, duplicate rate 0 and fallback rate 0.

Observed stability：

```text
Variant                    set Jaccard   top pair agree   all-3 agree   best-of-K value variance
------------------------------------------------------------------------------------------------
GPT-5.6 Sol                  0.434055        0.733333        0.633333           0.134165
GPT-5.4 Mini                 0.293252        0.500000        0.400000           0.918567
Gemini 3.5 Flash Lite       0.411077        0.677778        0.533333           1.402401
```

Repeatability costs for the fresh 90-call runs：

```text
GPT-5.6 Sol            USD 1.133325
GPT-5.4 Mini           USD 0.162754
Gemini Flash Lite      USD 0.109167
```

These stability values are experimental outcomes. They are not post-hoc thresholds and do not authorize prompt/temperature tuning.

## 8. Scientific interpretation

The combined B2 results support three claims to test later rather than assume now：

1. stronger raw LLM prior alignment is not guaranteed;
2. candidate diversity and repeated-generation stability differ materially by model;
3. WM foresight + PPO posterior correction is necessary to determine whether weaker/noisier priors can still produce competitive end-to-end policies.

No primary model is selected at B2.

## 9. Formal completion checklist

```text
registry frozen                          PASS
240-state validation bank                PASS
persistent cache                         PASS
3-model live capability                  PASS
3 LLM × 240 prior-quality reports        PASS
Uniform × 240 baseline                   PASS
30 states × 3 repeats × 3 LLM           PASS
provider failure hidden as fallback      NO
hidden/calibration/test leakage           NO
primary_model_selected                    FALSE
```

**FINAL STATUS: GATE B2 PASS / CLOSED**

Next：Gate B3 provider-neutral posterior representation regression, then Gate B4 PPO network/unit implementation.
