# Gate B — B2 Provider/Model Amendment: OFOX + GPT-5.6 Sol

日期：2026-09-17  
状态：**FORMAL AMENDMENT / IMPLEMENTATION IN PROGRESS**

---

## 1. Purpose

本文件是 `UG_CEM_APT_REPRODUCTION_PLAN.md` 中 Gate B / B2 的窄范围正式修订。

原 B2 provider/model 冻结：

```text
provider = Google Gemini API
model    = Gemini-3.1 / gemini-3.1-pro-preview
```

由于实际 Gemini 3.1 Pro API 项目在 B2-live 中返回 free-tier quota limit=0，现根据实验实际接入条件，将正式 LLM prior provider/model 改为：

```text
provider protocol = OFOX OpenAI-compatible API
base_url          = https://api.ofox.ai/v1
model             = openai/gpt-5.6-sol
paper label       = GPT-5.6 Sol via OFOX gateway
```

该变更属于实验实现模型变更，论文必须按实际执行模型报告；不得继续把正式主实验描述为 Gemini-3.1。

## 2. What changes

只改变 B2 的 LLM provider/model transport：

```text
Gemini-3.1 direct Google API
        ↓ superseded
OFOX OpenAI-compatible gateway
+ openai/gpt-5.6-sol
```

正式 authentication：

```text
OFOX_API_KEY
```

API key 只允许来自环境变量，不写入 Python/YAML/GitHub/report。

## 3. What remains frozen

以下 B1–B3 contract **不变**：

```text
A = 4
K = 6
H = 4
generation temperature = 0.2
FormalState D = 27
strict structured output
semantic parser
duplicate handling
neutral deterministic fallback
prior preference normalization
posterior candidate feature = state + plan + prior + value + uncertainty
candidate feature dim = 46
```

动作仍然只有：

```text
no_op / analyse / remove / restore
```

不得因为 provider/model 变化重新引入旧 action cost、delay、checkpoint coverage、early-warning evidence。

## 4. Structured output

OFOX OpenAI-compatible Chat Completions 官方接口支持：

```text
response_format.type = json_schema
```

因此 B2-live 继续使用服务端 JSON Schema + 本地 frozen semantic parser 双重验证：

```text
exact candidates = 6
exact actions/plan = 4
action enum = no_op/analyse/remove/restore
prior_score in [0,1]
reason = string
```

服务端 structured output 不能替代本地 parser、duplicate handling 和 fallback。

## 5. Network controls

保留可审计网络模式，但改为 OFOX 命名：

```text
default:
  sdk_environment_proxy

OFOX_DISABLE_ENV_PROXY=1:
  direct_no_env_proxy

OFOX_PROXY_URL=<proxy-url>:
  explicit_proxy
```

`OFOX_PROXY_URL` 与 `OFOX_DISABLE_ENV_PROXY` 互斥。

显式/直连模式均使用 `trust_env=False`，避免 Windows/Git Bash 隐式代理影响实验。

## 6. Superseded artifacts

以下只作为 B2-live 历史记录，不再是正式 active provider implementation：

```text
formal_experiments/ours/gemini_prior_client.py
Gemini-specific live test expectations
Gemini-specific quota remediation path
```

旧失败历史仍保留于 `docs/gate-b-B2-live.md`，用于审计为何发生 provider/model amendment。

## 7. Gate condition after amendment

正式 B2-live 必须重新执行：

```text
OFOX_API_KEY
-> OpenAI SDK
-> https://api.ofox.ai/v1/chat/completions
-> openai/gpt-5.6-sol
-> JSON Schema output
-> frozen parser
-> 6 x 4 PriorBatch
```

只有 OFOX exact-model live response PASS 后，才可关闭 B2-live 并进入 B4 PPO。

## 8. Paper reporting

论文实现细节必须写实际模型，例如：

> The LLM prior was instantiated with GPT-5.6 Sol accessed through the OFOX OpenAI-compatible gateway.

如果后续再次更换模型/provider，必须在训练 PPO 前重新更新 Gate B contract；禁止在 test 阶段替换模型。
