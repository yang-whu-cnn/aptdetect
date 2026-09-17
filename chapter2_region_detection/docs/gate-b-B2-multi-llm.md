# Gate B2 — Multi-LLM Prior Study

日期：2026-09-17  
状态：**B2.2 REGISTRY PASS / CLOSED；B2.3 STATE BANK PASS / CLOSED；CURRENT = THREE-MODEL LIVE PREFLIGHT**

---

## 1. Purpose

B2 不预先指定唯一主 LLM。正式 prior study 先冻结：

1. 三档 candidate model registry；
2. 完全相同的 validation state bank；
3. 完全相同的 A4/K6/H4/prompt/parser；
4. 后续完全相同的 frozen WM/reward evaluator。

Primary LLM 只能在 prior-quality 与 end-to-end validation 完成后确定。

## 2. Candidate registry — CLOSED

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

冻结 candidate：

```text
Tier-H  llm_h_gpt56_sol
        openai/gpt-5.6-sol

Tier-M  llm_m_gpt54_mini
        openai/gpt-5.4-mini

Tier-L  llm_l_gemini35_flash_lite
        google/gemini-3.5-flash-lite
```

三者当前都只是 candidate：

```text
primary_model_selected = false
```

共同 generation contract：

```text
A=4
K=6
H=4
temperature requested=0.2
prompt=lwm_rl_gate_b_v2_1
structured output requested=json_schema
```

Live preflight 前仍保持：

```text
structured_output_verified=false
actual_temperature=null
```

不能把“requested”写成“已验证支持”。

Registry 保存 freeze-time pricing snapshot，用于后续 token/cost efficiency comparison；价格变化不回写历史实验。

Retry contract：首次请求 + 最多 2 次 retry，仅 transport/rate-limit/5xx 可 retry。Tiny capability preflight 本身每模型只做一次请求，不进行 silent rescue。

Registry implementation：

```text
formal_experiments/ours/model_registry.py
```

## 3. Validation state bank — CLOSED

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

Frozen selection：

```text
8 seeds × 5 agents × 6 states/cell = 240 states
completed decision transitions only
exact D27 float32 SHA256 global dedupe
seed/agent deterministic order
feature17=0/1 both covered when both exist in source cell
even deterministic sampling inside available strata
```

State bank 只保存 planner-visible D27 + provenance，不保存 hidden Red truth、incident labels、reward、next/future state、calibration/test data。

## 4. Local foundation tests — PASS

用户本地执行：

```text
tests.test_gate_b2_registry_state_bank
tests.test_prior_cache
```

合计：

```text
15/15 PASS
```

因此 registry/state-bank/cache 的 offline protocol tests 均通过。

## 5. Real state-bank result — PASS

真实 validation replay：

```text
record_count: 240
unique_exact_state_count: 240
source_seeds: [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007]
agents: [blue_agent_0, blue_agent_1, blue_agent_2, blue_agent_3, blue_agent_4]
feature17_counts: {'0': 118, '1': 122}
selection_config_sha256: 00698b76e1adeeec9bba9eb3148bea1319052584a47ae8e4771792d6f10f4117
bank_sha256: b42472e9a20f66b56164d846e8bd41caa2cae54f0e9a6558d7ebebff2c771e17
pass: True
```

这满足：

- exact 240 records；
- exact 240 globally unique D27 states；
- 8 validation seeds；
- 5 Blue agents；
- feature17 两类均有充分覆盖；
- frozen selection config hash / bank hash 已产生；
- no calibration/test leakage。

因此：

```text
B2.2 candidate registry : PASS / CLOSED
B2.3 validation bank    : PASS / CLOSED
```

## 6. Current Gate — three-model live capability preflight

下一步不是 PPO，也不是直接跑 240×3 API calls。

先对三个 exact model 在同一个 frozen state 上各做一次 tiny live request：

```text
same D27 state
same prompt
same A4/K6/H4
same requested temperature=0.2
same json_schema
```

验证：

- exact model ID 可调用；
- temperature 参数被 endpoint 接受；
- JSON-Schema request 可用；
- raw response schema-valid；
- usage token metadata 可读；
- local parser final PriorBatch valid；
- no raw prompt/response/key serialization。

Source / tests / acceptance：

```text
docs/gate-b-B2-model-preflight.md
formal_experiments/evaluation/run_gate_b2_model_preflight.py
tests/test_gate_b2_model_preflight.py
```

Capability preflight PASS 后才允许：

```text
freeze verified capability fields
-> 240-state × 3 LLM prior-quality generation/evaluation
-> uniform baseline
-> 30-state × 3 repeatability
```

## 7. Current conclusion

```text
B2.2 registry             PASS / CLOSED
B2.3 validation state bank PASS / CLOSED
B2.5 cache unit gate      PASS / CLOSED
B2 live model preflight   SOURCE READY / LOCAL + REAL PENDING
Primary LLM selected      NO
PPO started               NO
Test seeds touched        NO
```
