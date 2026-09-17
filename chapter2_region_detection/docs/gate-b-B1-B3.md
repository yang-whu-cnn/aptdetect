# Gate B — B1–B3 Formal Prior / Posterior Contract

日期：2026-09-17  
状态：**B1/B3 CLOSED；B2 PROVIDER/MODEL AMENDED TO OFOX GPT-5.6 SOL / REGRESSION PENDING**

---

## 1. Scope

本子阶段冻结：

- B1 response reward contract re-audit；
- B2 four-action LLM prior v2.1；
- B3 PPO posterior candidate observation。

B4 PPO network/training/checkpoint selection 尚未开始；B5 stability 尚未开始。

legacy `src/llm_prior.py / src/ppo_posterior.py / src/plan_eval.py / src/env_region.py` 不修改、不复用为正式实现。

2026-09-17 对 B2 做窄范围 provider/model amendment：

```text
Gemini-3.1 direct Google API
    -> superseded
OFOX OpenAI-compatible API
+ openai/gpt-5.6-sol
```

正式修订记录：

```text
docs/gate-b-B2-ofox-amendment.md
```

B1/B3 与 B2 的 K/H/A/parser/fallback/posterior contracts 均保持不变。

## 2. B1 response reward contract

继续使用 Gate A 已冻结 objective：

```text
r_resp = -1.0 * incident_active_ticks
         + 1.0 * raw incident-host LWF penalty
```

不包含：

- early-warning lead reward；
- fixed action cost；
- lambda_delay；
- checkpoint coverage。

`IncidentResponseBookkeeper` 使用 interval-begin presence 计算 active tick：

```text
False -> True at t+1: [t,t+1] 不计 active
True  -> False at t+1: [t,t+1] 仍计 active
```

因此闭合事件满足：

```text
sum(active tick penalties)
= t_normal - t_compromise
= attack eradication duration
```

可执行 test 使用：

```text
False->True @1
True->False @4
```

累计 active ticks=3，event `T_erad=4-1=3`。

## 3. B2 LLM prior v2.1

正式模块：

```text
formal_experiments/ours/llm_prior_v2.py
configs/lwm_rl_gate_b_v2_1.yaml
```

当前正式 provider/model：

```text
provider protocol = OFOX OpenAI-compatible
base_url          = https://api.ofox.ai/v1
model             = openai/gpt-5.6-sol
paper model label = GPT-5.6 Sol via OFOX gateway
API key env       = OFOX_API_KEY
```

冻结参数保持：

```text
K = 6 candidates
H = 4 high-level decisions
generation temperature = 0.2
```

正式 action vocabulary：

```text
0 no_op
1 analyse
2 remove
3 restore
```

prompt 只使用 planner-visible FormalState D=27；不选择 host target，host target 仍由后续 shared resolver 决定。

prompt 明确禁止：

- hidden compromise truth；
- future reward；
- attack label；
- test information。

## 4. Parser / duplicate / fallback freeze

LLM 输出格式冻结为 strict JSON：

```json
{
  "candidates": [
    {
      "actions": ["no_op", "analyse", "remove", "restore"],
      "prior_score": 0.8,
      "reason": "..."
    }
  ]
}
```

parser 规则：

- 每条 plan 必须 exact H=4；
- 只接受 4 个 canonical names；
- `prior_score` 必须 finite 且位于 [0,1]；
- unknown action / wrong length / invalid score 直接丢弃；
- exact duplicate plan 去重，保留最高 prior_score；
- valid plans 按 score descending、plan lexicographic tie-break；
- 最多取 K=6。

prior preference normalization：

```text
nonnegative linear normalize to sum=1
```

全部 raw score 为 0 时使用 uniform 1/6。

fallback：

```text
deterministic_uniform_plan_space_sample
plan space = 4^4 = 256
fixed seed = 20260916
state-dependent = false
hidden-truth-dependent = false
fallback raw prior score = 0
```

它不是 risk heuristic，不读取状态，不偏向某个响应动作；只用于 API 成功后 semantic parse 不完整时补齐 K。

## 5. B3 posterior observation

正式模块：

```text
formal_experiments/ours/posterior_features.py
```

PPO action 不是高层 response action，而是：

```text
candidate plan index ∈ {0,...,5}
```

每个 candidate 正式 feature：

```text
current state             27
candidate plan one-hot    4*4 = 16
LLM prior preference       1
predicted value             1
predictive uncertainty      1
--------------------------------
candidate feature dim      46
```

不增加旧 evidence：

- Cost；
- checkpoint coverage C；
- delay；
- early-warning。

## 6. Value / uncertainty definitions

直接复用 A4.5c 已验证 H=4 member-return machinery：

```text
value_i = mean_m member_return[i,m]
uncertainty_i = population_std_m member_return[i,m]
```

member return 来自：

- exact same v2 absolute WM；
- exact same shared response reward predictor；
- fixed-member H=4 mean rollout；
- duration-aware `gamma_tick=0.99`。

posterior builder 强制 `expected_return == member_return mean`。

## 7. B4 pretrain boundary already recorded

Gate-B config 只提前锁定：

```text
learning_rate = 0.0003
rollout_length = 128
update_epochs = 5
clip_epsilon = 0.2
GAE lambda = 0.95
entropy coefficient = 0.01
```

正式系统继续使用：

```text
gamma_tick = 0.99
duration-aware discount = true
```

train seeds=1000..1031；validation=2000..2007；calibration/test forbidden。

PPO hidden/minibatch/value coefficient/network structure 留到 B4 单独冻结。

## 8. Legacy audit

formal-use flags：

```text
src_llm_prior_allowed: false
src_ppo_posterior_allowed: false
src_plan_eval_allowed: false
src_env_region_allowed: false
old_a5_checkpoint_allowed: false
gemini_prior_client_formal_allowed: false
```

新的 formal prior/posterior 不得出现：

- control_traffic；
- monitor/light_evidence/heavy_evidence；
- local_mitigate/strong_mitigate；
- action_cost；
- lambda_delay；
- checkpoint_coverage。

## 9. Tests

`tests/test_gate_b_llm_prior_posterior_contract.py` 仍为 15 tests。

provider/model amendment 后必须重新执行，因为第 1 项 model/provider/config contract 已变化；其余 B1/B3/parser/posterior tests 应保持原语义不变。

目标：

```text
Ran 15 tests
OK
```

## 10. Historical closure and amendment semantics

2026-09-16 原 Gemini 版本曾 15/15 PASS，证明当时的 B1/B3/parser/posterior contract 正确。

2026-09-17 provider/model 改为 OFOX GPT-5.6 Sol 后：

```text
B1 reward contract           : remains CLOSED
B3 posterior contract        : remains CLOSED
B2 A/K/H/parser/fallback     : remains FROZEN
B2 provider/model            : REOPENED NARROWLY and AMENDED
B2 amended regression        : PENDING
B2 live OFOX smoke           : PENDING
```

因此不能引用旧 Gemini 15/15 作为当前 OFOX model-contract 的最终 closure；必须重跑 amended regression。

## 11. Current conclusion

```text
B1 reward re-audit          : PASS / CLOSED
B2 four-action prior logic  : FROZEN
B2 provider/model           : OFOX GPT-5.6 Sol AMENDED
B2 strict parser            : FROZEN
B2 duplicate/fallback       : FROZEN
B3 posterior feature        : PASS / CLOSED
B3 value/uncertainty        : PASS / CLOSED
Legacy isolation            : READY
Amended B1-B3 regression    : PENDING (15)
B2 OFOX live smoke          : PENDING

FINAL STATUS: PROVIDER AMENDMENT APPLIED / REGRESSION + LIVE SMOKE PENDING
```
