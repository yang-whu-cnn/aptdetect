# Gate B2 — Three-Model Live Capability Preflight

日期：2026-09-17  
状态：**SOURCE READY / LOCAL 10-TEST + REAL LIVE PREFLIGHT PENDING**

---

## 1. Position in v2.3

已完成的前置基础设施：

```text
B0.1 frozen WM audit                  PASS / CLOSED
B2.2 candidate registry               PASS / CLOSED (offline contract)
B2.3 validation state bank            PASS / CLOSED
B2.5 persistent prior cache unit gate PASS / CLOSED
```

用户本地验收：

```text
registry/state-bank + cache combined tests: 15/15 PASS

real validation state bank:
record_count             = 240
unique_exact_state_count = 240
feature17 counts          = 118 / 122
selection_config_sha256  = 00698b76e1adeeec9bba9eb3148bea1319052584a47ae8e4771792d6f10f4117
bank_sha256              = b42472e9a20f66b56164d846e8bd41caa2cae54f0e9a6558d7ebebff2c771e17
pass                     = True
```

当前 Gate 只验证 provider/model capability。通过后才允许对 240-state bank 做正式 prior-quality generation。

不训练 PPO，不运行 test seeds，不选择主模型。

## 2. Source

```text
formal_experiments/evaluation/run_gate_b2_model_preflight.py
tests/test_gate_b2_model_preflight.py
```

正式输出：

```text
outputs/lwm_rl_v2/b2/preflight/model_capability_preflight.json
```

## 3. Frozen candidate models

来自：

```text
configs/llm_model_registry_v1.yaml
```

```text
Tier-H: openai/gpt-5.6-sol
Tier-M: openai/gpt-5.4-mini
Tier-L: google/gemini-3.5-flash-lite
```

三者统一：

```text
provider gateway = OFOX
endpoint         = OpenAI-compatible Chat Completions
A                = 4
K                = 6
H                = 4
prompt           = lwm_rl_gate_b_v2_1
requested temp   = 0.2
structured output= json_schema
```

`primary_model_selected=false` 必须保持不变。

## 4. Common state

Preflight 不给三个模型不同输入。

从已冻结 240-state validation bank 按：

```text
episode_seed
agent_name
decision_index
global_tick_start
state_sha256
```

字典序选择唯一第一个 state。

三个模型必须使用完全相同的：

```text
D27 state
agent_name
prompt version
A/K/H
temperature
JSON schema
```

runner 在调用前重新验证：

- state bank summary PASS；
- 240 records；
- 240 exact-unique D27 states；
- per-row D27 SHA256；
- selection-config hash consistency；
- full bank SHA256。

因此 preflight 不接受被修改或损坏的 state bank。

## 5. One live request per model

Tiny preflight 每个 exact model 只做一次 live request。

本 Gate：

- 不使用 prior cache；
- 不做 model-specific prompt tuning；
- 不自动换模型；
- 不删除 temperature 参数来救单个模型；
- 不使用 fallback 掩盖 API failure；
- 不根据输出质量选择 winner。

如果 provider 返回 transient error，本次 preflight 记录 failure class；正式 240-state generation 的 bounded retry 仍按 registry policy 实现。

## 6. Two-layer response validation

### Layer 1 — raw provider response

不能仅看 `parse_prior_response()` 的最终 6x4 输出，因为 neutral fallback 可能把坏响应补齐。

因此先直接检查 provider message JSON：

```text
root keys exactly: candidates
candidate count exactly: 6
candidate keys exactly: actions, prior_score, reason
actions length exactly: 4
actions only: no_op/analyse/remove/restore
prior_score finite in [0,1]
reason string
```

同时记录：

- semantic-valid candidate count；
- unique valid plan count；
- duplicate valid plan count。

Duplicate 本身不违反 JSON schema，因此在 tiny capability Gate 中作为 diagnostic；后续 prior-quality 会正式统计 duplicate/fallback rate。

### Layer 2 — formal parser

同一 raw response 再经过 frozen local parser。

最终必须满足：

```text
plans shape = [6,4]
A4 IDs only
scores finite
prior finite/nonnegative/sum=1
```

记录 `fallback_count`。

## 7. Temperature capability

请求始终发送：

```text
temperature=0.2
```

若请求成功，则可记录为：

```text
requested_temperature_accepted = true
```

这只证明该 endpoint/model 接受该请求参数；不声称 provider 暴露了内部实际 sampling temperature。

若 provider 明确拒绝该参数，preflight FAIL，之后先做 registry amendment，不允许私下为该模型删参数继续实验。

## 8. Usage / latency / cost

同一次 response 读取：

```text
prompt/input tokens
completion/output tokens
total tokens
response.model
finish_reason
wall-clock request latency
```

Registry 当前三模型均声明 `usage_metadata_expected=true`，因此 tiny preflight 要求 input/output token usage 可读。

估算成本：

```text
input_tokens  * frozen input USD/M
+
output_tokens * frozen output USD/M
```

除以 1,000,000。

这是 registry pricing snapshot 下的估算，不替代 provider 最终账单。

## 9. Failure recording

异常报告只保存：

```text
failure_class
exception_type
```

不保存原始 exception message，避免 message 意外包含 credential/proxy details。

failure classes 复用：

```text
quota_blocked
auth_blocked
network_blocked
model_blocked
api_error
```

## 10. Privacy / audit

Preflight report 明确：

```text
raw_prompt_recorded   = false
raw_response_recorded = false
api_key_value_recorded= false
cache_used            = false
```

只保存 prompt/response SHA256 及解析后的非敏感 diagnostics。

## 11. PASS conditions

每个三模型都必须：

1. exact requested model ID 的 API request 成功；
2. `temperature=0.2` 请求被 endpoint 接受；
3. JSON-Schema request 成功；
4. raw response `schema_valid=true`；
5. usage input/output tokens 可读；
6. final PriorBatch valid；
7. no raw/key serialization。

总体：

```text
all 3 model pass == True
```

任何一个失败：Gate 保持 OPEN。先诊断 capability/registry，不开始 240-state prior-quality，不替换 winner，不训练 PPO。

## 12. Tests

```text
tests/test_gate_b2_model_preflight.py
```

10 mock tests 覆盖：

1. exact raw schema acceptance；
2. unknown action raw-schema rejection；
3. duplicate diagnostic + parser fallback distinction；
4. usage/cost extraction；
5. frozen state-bank integrity / deterministic common-state selection；
6. all three exact IDs / same state / temp0.2 / json_schema；
7. missing usage => FAIL；
8. invalid raw schema cannot be hidden by final fallback；
9. provider exception safe classification；
10. no raw prompt/response/API key/cache in report。

## 13. Local execution

```bash
cd /d/paper/github-me/aptdetect/chapter2_region_detection
source .venv_cc4/Scripts/activate

git fetch origin
git pull --ff-only origin ug-cem-apt

python -m unittest tests.test_gate_b2_model_preflight -v
```

Expected：

```text
Ran 10 tests
OK
```

真实 preflight 前：

```bash
export OFOX_API_KEY="<your key>"
```

如果本机环境代理有问题，可沿用项目专用网络控制：

```bash
export OFOX_DISABLE_ENV_PROXY=1
unset OFOX_PROXY_URL
```

或显式设置 `OFOX_PROXY_URL`，但两者互斥。

运行：

```bash
python -m formal_experiments.evaluation.run_gate_b2_model_preflight
```

Expected summary form：

```text
[GATE B2 MODEL CAPABILITY PREFLIGHT]
state_sha256: <same state for all models>
network_mode: ...
tier_h ... api=True schema=True usage=True ... pass=True
tier_m ... api=True schema=True usage=True ... pass=True
tier_l ... api=True schema=True usage=True ... pass=True
pass: True
```

真实结果 PASS 后，再更新 registry 的 verified capability fields，并进入 240-state prior-quality generation。
