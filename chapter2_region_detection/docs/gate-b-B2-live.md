# Gate B — B2-live OFOX GPT-5.6 Sol Prior Smoke

日期：2026-09-17  
状态：**OFOX MIGRATION SOURCE READY / LOCAL 15-TEST + LIVE API SMOKE PENDING**

---

## 1. Provider/model amendment

B2 原先尝试使用 Gemini-3.1 direct Google API。真实 smoke 已证明网络层可达，但 `gemini-3.1-pro-preview` 对当前项目返回 free-tier quota limit=0。

根据当前实际实验接入条件，正式 B2 provider/model 已通过：

```text
docs/gate-b-B2-ofox-amendment.md
```

修订为：

```text
provider protocol = OFOX OpenAI-compatible API
base_url          = https://api.ofox.ai/v1
model             = openai/gpt-5.6-sol
paper label       = GPT-5.6 Sol via OFOX gateway
```

Gemini-specific client 与两次 Gemini live failure 仅保留为历史审计记录，不再属于 active formal provider。

## 2. Frozen B2 contract that remains unchanged

provider/model 变化不改变：

```text
A = 4
K = 6
H = 4
temperature = 0.2
FormalState D = 27
```

动作固定：

```text
no_op / analyse / remove / restore
```

仍禁止：

- hidden compromise truth；
- future reward；
- attack labels；
- test information；
- method-specific host target generation。

host target 仍由后续 shared resolver 决定。

## 3. Authentication

正式 key 只允许：

```text
OFOX_API_KEY
```

客户端构造函数不接受 `api_key=` 参数；真实 key 只从环境变量读取并传给 OpenAI SDK。

禁止：

- 写入 Python；
- 写入 YAML；
- 写入 GitHub；
- 打印到日志；
- 保存到 smoke report。

report 只记录：

```text
key_source = OFOX_API_KEY
api_key_value_recorded = false
```

## 4. Formal client

active implementation：

```text
formal_experiments/ours/ofox_prior_client.py
```

使用：

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.ofox.ai/v1",
    api_key=<OFOX_API_KEY from environment>,
)
```

正式调用：

```text
client.chat.completions.create(...)
model = openai/gpt-5.6-sol
temperature = 0.2
```

历史文件：

```text
formal_experiments/ours/gemini_prior_client.py
```

仅保留审计，不再由 formal smoke/import path 使用。

## 5. Structured output

OFOX OpenAI-compatible Chat Completions 支持 `response_format.type=json_schema`。

正式 request：

```text
response_format:
  type = json_schema
  json_schema.name = lwm_rl_candidate_plans
  json_schema.schema = frozen B2 schema
```

schema 强制：

- exactly 6 candidates；
- each plan exactly 4 actions；
- action enum only no_op/analyse/remove/restore；
- prior_score number in [0,1]；
- reason string。

服务端 structured output 后仍必须进入本地 frozen semantic parser：

```text
parse_prior_response()
```

因此 duplicate handling、invalid-plan rejection、neutral deterministic fallback 不变。

## 6. Network modes

为了避免 Windows/Git Bash 隐式代理再次污染实验，OFOX client 支持：

```text
default:
  sdk_environment_proxy

OFOX_DISABLE_ENV_PROXY=1:
  direct_no_env_proxy

OFOX_PROXY_URL=<proxy-url>:
  explicit_proxy
```

`OFOX_PROXY_URL` 与 `OFOX_DISABLE_ENV_PROXY` 互斥。

直连/显式代理模式均通过自建 `httpx.Client(..., trust_env=False)` 交给 OpenAI SDK。

proxy diagnostics 只显示：

```text
scheme://host:port
```

不会保存 user/password。

## 7. Live smoke state

runner 不改路径：

```text
formal_experiments/evaluation/run_gate_b_b2_live_prior_smoke.py
```

默认仍使用 Step-6 state-only artifact：

```text
agent = blue_agent_0
seed = 3000
earliest deterministic planner-visible D27 state
```

这只是 API connectivity/contract smoke，不参与 prompt tuning、模型选择或 PPO training。

## 8. Report

输出：

```text
outputs/lwm_rl_v2/gate_b/b2_live_prior_report.json
```

记录：

- provider；
- base_url；
- model；
- temperature；
- key source name；
- network mode；
- parsed plans；
- raw prior scores；
- normalized prior preferences；
- unique plan count；
- fallback count；
- prompt/response SHA256。

不记录：

- API key value；
- raw prompt；
- raw response；
- proxy credentials。

## 9. Failure semantics

分类：

```text
quota_blocked
auth_blocked
network_blocked
model_blocked
api_error
```

真实 API 没成功返回时 hard fail；不会使用 fallback 掩盖 transport/auth/model/quota failure。

只有 API 成功后，semantic parser 才允许用 neutral deterministic fallback 补足不完整/重复 candidate。

## 10. Tests

`tests/test_gate_b_b2_live_prior.py` 继续保持 15 个 tests，但语义已全部迁移到 OFOX：

1. `OFOX_API_KEY` source / missing guard；
2. default/direct/explicit network mode；
3. explicit network client construction；
4. exact K/H/A/score JSON schema；
5. OpenAI-compatible json_schema wrapper；
6. mock Chat Completions exact model/temp/schema；
7. client constructor 无 api_key 参数；
8. empty output hard fail；
9. hashes / no key exposure；
10. report privacy + OFOX metadata；
11. parser fallback after successful API call；
12. failure-class separation；
13. proxy diagnostic redaction；
14. deterministic smoke-state selection；
15. missing state rejection。

同时原 B1–B3 15 tests 也必须重新跑，因为 provider/model constants/config 已正式变更。

## 11. Local sequence

同步后安装 OpenAI SDK：

```bash
python -m pip install -U openai
```

先跑 B1–B3 regression：

```bash
python -m unittest tests.test_gate_b_llm_prior_posterior_contract -v
```

目标：

```text
Ran 15 tests
OK
```

再跑 B2-live mock tests：

```bash
python -m unittest tests.test_gate_b_b2_live_prior -v
```

目标：

```text
Ran 15 tests
OK
```

## 12. Key setup

Git Bash 当前 session：

```bash
export OFOX_API_KEY="<your key>"
```

不要把真实 key 发到聊天或 commit。

如果 OFOX 可以直连，建议避免之前的系统代理：

```bash
export OFOX_DISABLE_ENV_PROXY=1
unset OFOX_PROXY_URL
```

如果必须使用本机代理：

```bash
unset OFOX_DISABLE_ENV_PROXY
export OFOX_PROXY_URL="http://127.0.0.1:<actual-port>"
```

## 13. Real live command

```bash
python -m formal_experiments.evaluation.run_gate_b_b2_live_prior_smoke
```

目标 summary：

```text
[GATE B B2-LIVE SUMMARY]
api_call_succeeded: True
key_source: OFOX_API_KEY
network_mode: ...
provider: ofox_openai_compatible
base_url: https://api.ofox.ai/v1
model: openai/gpt-5.6-sol
temperature: 0.2
plans_shape: [6, 4]
unique_plan_count: ...
fallback_count: ...
prior_sum: approximately 1.0
pass: True
```

## 14. Close condition

B2-live 只有在：

- B1–B3 amended 15/15 regression PASS；
- OFOX B2-live 15/15 tests PASS；
- real OFOX `openai/gpt-5.6-sol` API call PASS；
- final PriorBatch 6×4 legal + finite；
- report 不含 key/raw prompt/raw response；

之后才能 CLOSED。

关闭后才进入 B4 PPO network/training pipeline。

## 15. Historical Gemini attempt

历史 B2-live 曾经历：

1. system proxy TLS EOF；
2. `GEMINI_DISABLE_ENV_PROXY=1` 后 transport PASS；
3. Gemini 3.1 Pro free-tier quota limit=0。

这些结果证明旧 Gemini transport/path 的失败原因，但不再构成当前 OFOX formal Gate 条件。

## 16. Current conclusion

```text
B1/B3 invariant contract       : FROZEN
B2 provider/model amendment    : APPLIED
Formal model                   : openai/gpt-5.6-sol
Formal gateway                 : OFOX OpenAI-compatible
Formal key env                 : OFOX_API_KEY
OFOX client                    : SOURCE READY
JSON Schema + semantic parser  : SOURCE READY
Local B1-B3 regression         : PENDING
Local B2-live tests            : PENDING
Real OFOX response             : PENDING

FINAL STATUS: OFOX MIGRATION SOURCE READY / TEST + LIVE SMOKE PENDING
```
