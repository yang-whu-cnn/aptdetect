# Gate B2 — Multi-LLM Prior Study

日期：2026-09-17  
状态：**B2.2 REGISTRY PASS / CLOSED；B2.3 STATE BANK PASS / CLOSED；B2.5 CACHE PASS / CLOSED；CURRENT = THREE-MODEL LIVE PREFLIGHT**

---

## 1. Purpose

B2 不预先指定唯一主 LLM。正式 prior study 先冻结：

1. 三档 candidate model registry；
2. 完全相同的 validation state bank；
3. 完全相同的 A4/K6/H4/prompt/parser；
4. 后续完全相同的 frozen WM/reward evaluator。

Primary LLM 只能在 prior-quality 与 end-to-end validation 完成后确定。

## 2. Candidate registry — PASS / CLOSED

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

Live preflight 前：

```text
structured_output_verified=false
actual_temperature=null
```

不能把 requested capability 写成 verified capability。

Registry implementation：

```text
formal_experiments/ours/model_registry.py
```

## 3. Validation state bank — PASS / CLOSED

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

### Real result

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

## 4. Offline foundation tests — PASS

用户本地执行：

```text
tests.test_gate_b2_registry_state_bank
tests.test_prior_cache
```

合计：

```text
15/15 PASS
```

因此：

```text
B2.2 candidate registry : PASS / CLOSED
B2.3 validation bank    : PASS / CLOSED
B2.5 prior cache        : PASS / CLOSED
```

## 5. Current Gate — three-model live capability preflight

下一步不是 PPO，也不是直接跑 240×3 API calls。

对三个 exact model 在**同一个 frozen bank state** 上各做 tiny live request：

```text
same D27 state
same prompt
same A4/K6/H4
same requested temperature=0.2
same json_schema request
```

验证：

- exact model ID 可调用；
- temperature parameter 被 endpoint 接受；
- JSON-Schema request 被 endpoint 接受；
- raw response 本身是合法 JSON/schema，不允许 parser fallback 掩盖 provider incompatibility；
- raw semantic-valid candidate count=6；
- local parser final PriorBatch valid；
- usage token metadata 尽可能读取；
- bounded retry 只用于 registry 声明的 rate-limit/network/5xx；
- auth/model-not-found/invalid-request 不 retry；
- primary model 仍保持未选择。

### Source

```text
formal_experiments/evaluation/run_b2_multi_model_preflight.py
```

### Unit tests

```text
tests/test_gate_b2_multi_model_preflight.py
```

8 tests 锁定：

1. exact six-candidate raw contract PASS；
2. malformed JSON raw contract FAIL；
3. wrong K / invalid action raw contract FAIL；
4. retry/error-class classification；
5. registry model/temp/json_schema/usage request contract；
6. retryable 5xx bounded retry；
7. invalid-request no retry；
8. parser fallback cannot mask raw contract failure。

### Real preflight PASS

三个模型都必须：

```text
api_success = True
structured_output_verified = True
temperature_parameter_accepted = True
semantic_valid_candidate_count = 6
final_prior_valid = True
pass = True
```

`fallback_count` 可作为诊断，但 raw schema/semantic contract 不得依赖 fallback 才成立。

若任一模型对 `json_schema` 或 temperature 明确返回 unsupported/invalid request：

```text
STOP
-> 记录真实 capability
-> 修订 registry / protocol
-> 三模型共同公平规则重新冻结
```

不得只给单个模型 silent special-case。

## 6. Public capability re-check (2026-09-17)

在进入 live preflight 前再次核对 OFOX 当前公开目录/文档：三个 exact IDs 仍存在；OpenAI-compatible Chat Completions 文档仍公开 `temperature` 与 `response_format`；Structured Output 文档仍描述 `json_schema`。

该公开信息只证明 gateway/documentation 层 capability，不能替代 exact-model live preflight。

## 7. After preflight

只有三模型 real preflight PASS 后才进入：

```text
freeze verified capability fields
-> 240-state × 3 LLM prior generation
-> same frozen WM/reward prior-quality evaluation
-> uniform baseline
-> 30-state × 3 independent repeatability
```

仍然不在 prior-quality 阶段选择最终论文 winner；primary LLM 必须等待 end-to-end validation。

## 8. Current conclusion

```text
B0.1 frozen WM audit      PASS / CLOSED
B2.2 registry             PASS / CLOSED
B2.3 validation state bank PASS / CLOSED
B2.5 cache                PASS / CLOSED
B2 live model preflight   SOURCE READY / 8 UNIT TESTS + REAL PENDING
Primary LLM selected      NO
PPO started               NO
Test seeds touched        NO
```
