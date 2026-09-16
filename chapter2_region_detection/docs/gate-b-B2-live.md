# Gate B — B2-live Gemini Prior Smoke

日期：2026-09-16  
状态：**NETWORK FIX READY / LOCAL 13-TEST + LIVE API SMOKE PENDING**

---

## 1. Purpose

B1–B3 已完成 formal contract；本子 Gate 只验证真实 Gemini 网络链路：

```text
environment API key
-> google-genai client
-> Gemini Interactions API
-> D27 planner-visible prompt
-> JSON-Schema structured output
-> frozen B2 parser
-> K=6 x H=4 four-action PriorBatch
```

不训练 PPO，不改 prompt，不调 temperature，不做 validation selection。

## 2. Authentication

代码不接受 `api_key=` 参数。

只允许 Google GenAI SDK 官方环境变量：

```text
GOOGLE_API_KEY   # if present, takes precedence
GEMINI_API_KEY
```

程序只记录环境变量名称，例如：

```text
key_source = GEMINI_API_KEY
```

绝不读取/打印/保存 key value。

## 3. SDK / API

正式 live adapter：

```text
formal_experiments/ours/gemini_prior_client.py
```

使用当前 Google GenAI SDK：

```text
from google import genai
client = genai.Client()
client.interactions.create(...)
```

API request 冻结：

```text
model = gemini-3.1-pro-preview
temperature = 0.2
store = false
response mime = application/json
JSON Schema = exact K=6 / H=4 / four-action enum / score [0,1]
```

`store=false` 显式关闭 Interactions API 默认 server-side Interaction storage。

## 4. Structured-output schema

服务端 schema 已约束：

- exactly 6 candidates；
- each candidate actions exactly 4；
- each action enum only no_op/analyse/remove/restore；
- prior_score number in [0,1]；
- reason string。

即使 structured output 合法，客户端仍必须再经过已冻结 `parse_prior_response()` 做语义验证、duplicate handling 与 fallback。

## 5. Live smoke state

runner：

```text
formal_experiments/evaluation/run_gate_b_b2_live_prior_smoke.py
```

默认从 Step-6 planner-visible state-only artifact 选：

```text
agent = blue_agent_0
seed = 3000
earliest deterministic state
```

这里只做 API connectivity / contract smoke，不参与 prompt tuning、model selection 或 PPO training。

该 state-only artifact 已在 Step 6 验证：不含 hidden truth / reward labels。

## 6. Failure semantics

真实 B2-live smoke：

- missing key -> hard fail；
- SDK missing -> hard fail；
- API transport/auth/quota/model error -> hard fail；
- empty response -> hard fail。

不会用 fallback 掩盖“API 根本没调用成功”。

API 成功后，如果模型内容经过 semantic parser 出现 duplicate/incomplete candidates，正式 neutral parser fallback 仍可补齐 K；报告会记录 `fallback_count`。

因此 live Gate 验证的是：

```text
API call succeeded
+ final formal PriorBatch valid
```

而不是用一次调用对 prior quality 做调参。

## 7. Privacy / report

正式 report：

```text
outputs/lwm_rl_v2/gate_b/b2_live_prior_report.json
```

只记录：

- key source name，不记录 value；
- model / temperature；
- agent / seed；
- parsed plans / raw prior scores / normalized preferences / sources；
- unique_plan_count / fallback_count；
- prompt SHA256 / response SHA256。

明确不记录：

- raw API key；
- raw prompt；
- raw response。

## 8. Tests

新增：

```text
tests/test_gate_b_b2_live_prior.py
```

原始 10 tests + 3 network tests，共 13 tests：

1. key env precedence / missing guard；
2. structured-output exact K/H/action/score schema；
3. mock Interactions request + `store=False`；
4. client constructor 不存在 api_key 参数；
5. empty response hard failure；
6. hashes present / key value not exposed；
7. report 不保存 raw prompt/response/key；
8. API success 后 parser fallback 仍可审计；
9. smoke state deterministic selection；
10. missing agent/seed state rejection。

mock tests 不需要 API key，也不会访问网络。

## 9. Local sequence

先安装当前官方 SDK：

```bash
python -m pip install -U google-genai
```

然后先跑无网络单测：

```bash
python -m unittest tests.test_gate_b_b2_live_prior -v
```

目标：

```text
Ran 13 tests
OK
```

之后在环境变量已配置的同一终端执行：

```bash
python -m formal_experiments.evaluation.run_gate_b_b2_live_prior_smoke
```

live summary 必须：

```text
api_call_succeeded: True
model: gemini-3.1-pro-preview
temperature: 0.2
plans_shape: [6, 4]
prior_sum: approximately 1.0
pass: True
```

`fallback_count` 可以记录为 0..6；它是 parser robustness 诊断，不作为单次 live connectivity Gate 的调参指标。

## 10. Close condition

B2-live 只有在：

- 13/13 mock tests PASS；
- real Gemini API call PASS；
- report 不含 key/raw prompt/raw response；

之后才能 CLOSED。

关闭后才进入 B4 PPO network / training pipeline。

## 11. Current conclusion

```text
B1-B3 formal contract : PASS / CLOSED
Gemini live client    : READY
Structured output     : READY
Key hygiene           : READY
Stateless store=false : READY
Mock tests            : READY (13)
Live API execution    : PENDING

FINAL STATUS: SOURCE READY / LIVE SMOKE PENDING
```


## 12. First live failure — proxy/TLS preflight

首次真实调用在 Gemini 响应前失败：

```text
httpcore._sync.http_proxy
-> start_tls
-> SSL: UNEXPECTED_EOF_WHILE_READING
-> google.genai ... APIConnectionError
```

该堆栈说明请求已经进入 HTTP proxy transport，失败发生在 TLS CONNECT/handshake 阶段；因此不能归因于 B2 parser、K/H/A contract、模型输出质量或 PPO。

Google GenAI SDK 默认 httpx client 使用环境代理发现。为使实验网络条件可审计，新增仅通过环境变量控制的 Gemini-specific network modes：

```text
default:
  sdk_environment_proxy

GEMINI_DISABLE_ENV_PROXY=1:
  direct_no_env_proxy

GEMINI_PROXY_URL=<proxy-url>:
  explicit_proxy
```

`GEMINI_PROXY_URL` 与 `GEMINI_DISABLE_ENV_PROXY` 互斥。

explicit/direct 模式均向 `google.genai.types.HttpOptions(client_args=...)` 传递 `trust_env=False`，从而避免 Windows/Git Bash 的隐式代理污染；explicit 模式再单独传 `proxy=`。

新增 `masked_detected_proxies()` 仅显示：

```text
scheme://hostname:port
```

会删除 user/password，不记录 proxy credential。

runner 网络异常现在会给出：

- `network_mode`；
- redacted detected proxy endpoints；
- direct / explicit proxy 修复提示。

report 成功后额外记录 `network_mode`，仍不记录 proxy URL 或 credentials。

新增 3 tests：

11. network mode default/direct/explicit + mutual exclusion；
12. HttpOptions direct/explicit proxy contract；
13. proxy diagnostic userinfo redaction。

因此 B2-live 当前共 13 tests。首次 SSL failure 作为网络 preflight 历史保留；13 tests + real API PASS 前 Gate 仍保持 OPEN。