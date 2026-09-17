# Gate B2 — Three-Model Live Capability Preflight

日期：2026-09-17  
状态：**PASS / CLOSED**

---

## 1. Purpose

在正式 240-state prior-quality 实验前，对 registry 中三个 exact model 使用同一个 frozen validation D27 state 做 tiny live capability verification。

本 Gate 不比较模型质量、不选择主模型、不训练 PPO、不使用 test seeds。

## 2. Frozen inputs

```text
registry  = configs/llm_model_registry_v1.yaml
bank      = outputs/lwm_rl_v2/b2/state_bank.jsonl
bank hash = b42472e9a20f66b56164d846e8bd41caa2cae54f0e9a6558d7ebebff2c771e17
A/K/H     = 4/6/4
prompt    = lwm_rl_gate_b_v2_1
temp      = 0.2
format    = json_schema
```

Common probe：

```text
state_sha256 = 2dd1353c3c10bd59def5d4e57ab11857d9d7ae55bb8814aefe2f020d23079de3
network_mode = direct_no_env_proxy
```

## 3. Source / tests

```text
formal_experiments/evaluation/run_b2_multi_model_preflight.py
tests/test_gate_b2_multi_model_preflight.py
```

用户本地先执行原 8-test preflight suite：

```text
8/8 PASS
```

随后 raw validator 进一步收紧为严格执行 JSON Schema `additionalProperties=false`：root 只能有 `candidates`，candidate 只能有 `actions/prior_score/reason`。对应 suite 现在为 9 tests，正式 240-state batch 前需回归一次。

## 4. Real live result

```text
[GATE B2 MULTI-MODEL LIVE PREFLIGHT]
network_mode: direct_no_env_proxy
probe_state_sha256: 2dd1353c3c10bd59def5d4e57ab11857d9d7ae55bb8814aefe2f020d23079de3

tier_h openai/gpt-5.6-sol:
  api=True
  structured=True
  temp_accepted=True
  raw_valid=6/6
  duplicates=0
  final=True
  fallback=0
  attempts=1
  usage={prompt_tokens:694, completion_tokens:289, total_tokens:983}
  pass=True

tier_m openai/gpt-5.4-mini:
  api=True
  structured=True
  temp_accepted=True
  raw_valid=6/6
  duplicates=0
  final=True
  fallback=0
  attempts=1
  usage={prompt_tokens:694, completion_tokens:258, total_tokens:952}
  pass=True

tier_l google/gemini-3.5-flash-lite:
  api=True
  structured=True
  temp_accepted=True
  raw_valid=6/6
  duplicates=0
  final=True
  fallback=0
  attempts=1
  usage={prompt_tokens:921, completion_tokens:349, total_tokens:1270}
  pass=True

primary_model_selected=False
pass=True
```

## 5. Close decision

全部三个 exact model 均在一次请求内通过：

- exact ID live access；
- requested temperature=0.2 accepted；
- JSON Schema request accepted；
- raw 6/6 candidates valid；
- duplicate=0；
- parser final K6/H4 valid；
- fallback=0；
- usage metadata available。

因此 registry capability fields 已更新：

```text
structured_output_verified=true
actual_temperature=0.2
```

这表示 endpoint 实测接受该参数与结构化请求，不声称能观察 provider 内部采样实现。

Primary model 仍未选择。

## 6. Next

进入：

```text
Gate B2 formal prior-quality
240 frozen validation states × 3 LLM
+ uniform non-LLM baseline
+ persistent cache
+ same frozen WM/reward evaluator
```

完成 prior-quality 后再做 30-state × 3-repeat repeatability。PPO 仍未开始。
