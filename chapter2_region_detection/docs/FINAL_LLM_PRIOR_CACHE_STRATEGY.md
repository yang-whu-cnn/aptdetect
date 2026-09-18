# Final LLM Prior Cache Strategy

## 1. Change of Execution Strategy

Date: 2026-09-18

The online LLM provisional stage with 2000 transitions was stopped manually.
The generated artifacts are retained as development evidence only and are not eligible for final paper tables.

Reason:

- Calling LLM for every decision transition introduces unnecessary latency and cost.
- The role of LLM in the final method is prior generation, not the policy execution engine.
- Repeated PPO experiments require deterministic and low-cost prior access.

The final execution strategy changes from online generation to offline prior construction.

---

## 2. Final Frozen Method

The final method remains:

```
LLM generates K=6 candidate response plans
        |
        v
World Model predicts future reward and uncertainty
        |
        v
PPO selects candidate plan
        |
        v
Execute first action and re-plan
```

The optimization only changes the deployment mechanism of LLM prior generation.

---

## 3. LLM Role

LLM is treated as a teacher prior generator.

Input:

- Current observable D27 state
- Agent information
- Available actions

Output:

- K=6 candidate response plans
- Prior preference score
- Structured action sequence

LLM must not access:

- hidden compromise truth
- future reward
- evaluation labels

---

## 4. Model Selection

The final prior generation model uses GPT-5.6 Sol as the teacher model.

Reason:

- Better structured planning ability
- Better long-horizon reasoning
- More reliable candidate generation

The model is used offline only.

Online training does not repeatedly call the LLM.

---

## 5. Prior Dataset Construction

New pipeline:

```
CC4 state sampling
        |
        v
State bank
        |
        v
GPT-5.6 Sol offline generation
        |
        v
prior_cache_final
```

Target size:

- Minimum: 1000 states
- Recommended: 2000 states
- Optional extension: 5000 states

---

## 6. Cache Structure

```
outputs/lwm_rl_final_20260917/prior_cache_final/

train.jsonl
validation.jsonl
manifest.json
```

Each record contains:

- state vector
- candidate plans
- prior scores
- model identifier
- prompt version
- generation metadata

---

## 7. Runtime Mode

Add offline mode:

```
state
 |
 v
prior cache retrieval
 |
 v
candidate plans
 |
 v
World Model
 |
 v
PPO
```

Cache miss behavior:

- Development mode: allow fallback LLM call
- Formal experiment mode: reject cache miss

This guarantees reproducibility.

---

## 8. Paper Experiment Compatibility

The final comparison remains unchanged.

Table 2:

- RL-Only
- LLM-RL
- WM-RL
- LWM-RL

Definitions:

```
RL-Only:
PPO without prior

LLM-RL:
LLM prior cache + PPO

WM-RL:
World Model + PPO

LWM-RL:
LLM prior cache + World Model + PPO
```

---

## 9. Implementation Plan

1. Add offline prior builder.
2. Extend prior cache format with final protocol version.
3. Add offline runtime switch.
4. Generate GPT-5.6 Sol prior dataset.
5. Audit cache quality.
6. Run formal PPO experiments.

---

## 10. Non-goals

The following are prohibited:

- Changing D27 state representation.
- Changing A4 action space.
- Changing final reward protocol.
- Using hidden attack labels during prior generation.
- Using old reward provisional results as final results.
