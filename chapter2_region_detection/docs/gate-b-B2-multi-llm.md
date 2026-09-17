# Gate B2 — Multi-LLM Prior Study

日期：2026-09-17  
状态：**B2.2 REGISTRY SOURCE READY / B2.3 STATE-BANK SOURCE READY / LOCAL REAL BANK PENDING**

---

## 1. Purpose

B2 不预先指定唯一主 LLM。正式 prior study 先冻结：

1. 三档 candidate model registry；
2. 完全相同的 validation state bank；
3. 完全相同的 A4/K6/H4/prompt/parser；
4. 后续相同 frozen WM/reward evaluator。

Primary LLM 只能在 prior-quality 与 end-to-end validation 完成后确定。

## 2. Candidate registry

Canonical file：

```text
configs/llm_model_registry_v1.yaml
```

统一 gateway/protocol：

```text
OFOX
https://api.ofox.ai/v1
OpenAI-compatible Chat Completions
OFOX_API_KEY
```

冻结候选：

```text
Tier-H  llm_h_gpt56_sol
        openai/gpt-5.6-sol

Tier-M  llm_m_gpt54_mini
        openai/gpt-5.4-mini

Tier-L  llm_l_gemini35_flash_lite
        google/gemini-3.5-flash-lite
```

三者当前都只是 candidate；`primary_model_selected=false`。

共同 generation request：

```text
A=4
K=6
H=4
temperature=0.2
prompt=lwm_rl_gate_b_v2_1
structured output requested=json_schema
```

`structured_output_verified=false` 与 `actual_temperature=null` 在 live preflight 前保持如此；不允许把“requested”伪装成“provider/model 已验证支持”。

Registry 还保存当前 pricing snapshot，后续效率实验必须记录实际 token/cost；价格变化时保留 registry freeze snapshot，不回写历史结果。

Retry contract：`max_attempts=3`，等价于首次请求 + 最多 2 次 retry；仅 transport/rate-limit/5xx 可 retry。

## 3. Registry acceptance

必须满足：

- exact 3 tiers；
- alias/model ID unique；
- common OFOX gateway/protocol；
- K6/H4/temp0.2 一致；
- pricing provenance 存在；
- no API key value；
- primary model 未选择；
- exact IDs live preflight 前允许 `pending-live`，但 prior-quality 正式运行前必须验证。

Implementation：

```text
formal_experiments/ours/model_registry.py
```

## 4. Validation state bank

Source：

```text
outputs/formal_replay_v2/validation.jsonl
seeds=2000..2007 only
agents=blue_agent_0..4
```

Builder：

```text
formal_experiments/evaluation/build_llm_validation_state_bank.py
```

Canonical outputs：

```text
outputs/lwm_rl_v2/b2/state_bank.jsonl
outputs/lwm_rl_v2/b2/state_bank_summary.json
```

Selection：

```text
8 seeds × 5 agents × 6 states/cell = 240 states
completed decision transitions only
exact D27 float32 SHA256 dedupe
seed/agent deterministic order
feature17=0/1 both covered when both exist in source cell
even deterministic sampling inside available strata
```

Global exact-D27 duplicates are forbidden. If a later seed×agent cell cannot supply six globally unique completed states, builder hard fails rather than silently reusing duplicate states.

State-bank record contains only：

- bank format version；
- validation split；
- source replay provenance；
- selection-config SHA256；
- seed/agent/decision/global tick；
- feature17 planner-visible flag；
- exact D27 float32 state + SHA256。

It does not save：

- hidden Red truth；
- incident labels；
- response/official reward；
- next/future state；
- calibration/test data。

## 5. State-bank acceptance

Real validation replay must produce：

```text
record_count = 240
unique_exact_state_count = 240
source_seeds = [2000..2007]
5 agents present
per_seed = 30 each
per_agent = 48 each
per_cell = 6 each
state_dim = 27
state_dtype = float32_le
pass = True
```

Deterministic rerun must reproduce identical：

```text
selection_config_sha256
bank_sha256
```

If source data cannot satisfy 240 globally unique states while retaining cell coverage, B2.3 stays OPEN and the shortage must be documented before changing the selection protocol. Thresholds/rules are not relaxed after seeing LLM results.

## 6. Tests

```text
tests/test_gate_b2_registry_state_bank.py
```

8 tests cover：

1. exact three-tier registry/no primary；
2. common A4/K6/H4/temp prior contract；
3. cost-tier gradient；
4. feature17 coverage；
5. exact-D27 dedupe；
6. exact 240 unique balanced planner-visible bank；
7. deterministic rerun hash；
8. exact float32 hash/no quantization。

## 7. Next after PASS

B2.2/B2.3 PASS 后：

```text
B2.5 cache local gate
-> B2 live preflight for each exact model
-> freeze actual parameter support
-> 3 LLM + uniform prior-quality generation/evaluation
-> repeatability subset
```

不在 state-bank 阶段选择 winner，不触碰 test seeds，不训练 PPO。
