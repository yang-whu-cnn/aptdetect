# Gate B4 — One-Model Tiny CC4 Pipeline Smoke

日期：2026-09-17  
状态：**PASS / CLOSED**

---

## 1. Preconditions

```text
B0.1 WM final audit                 PASS / CLOSED
B2 multi-LLM / cache               PASS / CLOSED
B3 posterior 46D regression        PASS / CLOSED
B4 PPO core                        PASS / CLOSED
B4 rollout/trainer integration     PASS / CLOSED (12/12 local tests)
Design/paper alignment audit       PASS
```

本阶段不是 provisional/formal PPO experiment，不产生论文性能结果。

---

## 2. Purpose

第一次把完整 LWM-RL runtime 链在真实 CC4 上串起来：

```text
planner-visible D27
-> exact train prior cache
-> one exact LLM on cache miss
-> frozen M5 WM + reward predictor H4 rollout
-> 46D posterior / candidate
-> PPO chooses candidate index
-> requested A4 = selected plan[0]
-> shared target resolver / CybORG adapter
-> real CC4 asynchronous action interval
-> real response reward / dt / done
-> PPO rollout buffer
-> one finite PPO optimizer update
```

只验证工程/语义一致性，不比较算法性能，不选择 primary LLM。

---

## 3. Frozen smoke configuration

```text
model_alias = llm_l_gemini35_flash_lite
exact_model = google/gemini-3.5-flash-lite
train_seed = 1000
scenario_steps = 20
ppo_seed = 20260917
device = cpu
pad_spaces = false
cache_format = 2 / agent-aware
```

使用 Tier-L 的唯一原因是 B2 已验证其接口且成本/延迟最低，适合作为工程 smoke。

**这不是模型选择。** `primary_model_selected=false`。

---

## 4. Candidate-context semantics

每个 ready decision state 只构造一次 candidate context：

```text
state + public agent id
-> train cache lookup
-> miss: bounded live provider transaction
-> PriorBatch K6/H4
-> frozen WM/reward rollout
-> [6,46] posterior features
```

当动作完成且 episode 未结束时，在真实 `next_state` 上预构造下一 candidate context，并用其 critic value 作为 `V(next)`。

下一 decision epoch 复用该 context，并通过 exact float32 D27 SHA256 验证 next state 完全一致。

---

## 5. Runtime action contract

PPO action 是 candidate index：

```text
candidate_index i
-> requested_action_id = plans[i][0]
```

只有 `plan[0]` 进入真实 CC4。随后统一走：

```text
requested A4
-> shared observable target resolver
-> shared CybORG adapter
-> actual executed action
```

Target 不由 LLM/PPO 直接选择。无合法 observable target 时 targeted request fallback 到 Sleep；requested action 保留用于审计。

---

## 6. Reward / trajectory alignment

每个完成 transition 同时进入：

1. formal `DecisionEpochReplayCollector`；
2. `AsyncPPORolloutBuffer`。

逐 transition 硬比较：

```text
episode_seed
agent_name
decision_index
selected candidate
requested action == selected plan[0]
real response_reward
real decision_dt
done
```

PPO reward 直接取 `DecisionEpochTransition.response_reward`。Terminal `critic_next_value=0`；非 terminal critic bootstrap 来自真实 next-state posterior context。

Hidden Red truth 只允许进入既有 incident-response reward bookkeeping，不进入 state/prompt/posterior/policy。

---

## 7. PPO update

Smoke episode 完成后：

```text
formal replay transition count == PPO rollout step count
-> per-agent duration-aware GAE
-> merge batch
-> PPOTrainer.update()
```

Optimizer 前后的 cache-miss / live-transaction counter 必须完全不变，证明 repeated gradient epochs 不触发 API。

---

## 8. Real local acceptance result

用户本地真实运行：

```text
transition_count = 92
ppo_step_count = 92
per_agent = [19,19,19,19,16]
all_agents_done = true

requested actions:
  no_op   28
  analyse 48
  remove  10
  restore  6

executed families:
  Sleep   89
  Analyse 3
fallback_count = 61

plan0_match = 92/92
PPO-replay alignment = 92/92

response_reward_total = -10.0
official_reward_total = -30.0

cache_hits = 0
cache_misses_before_update = 92
cache_misses_after_update  = 92
live_transactions_before_update = 92
live_transactions_after_update  = 92
estimated_cost_usd = 0.112192

training_batch_size = 92
optimizer_steps = 10
policy_loss_mean = 0.0022907955
value_loss_mean = 2.2935701
entropy_mean = 1.7917554
approx_kl_mean = 7.195e-06
clip_fraction_mean = 0.0
grad_norm_mean = 5.74391
grad_norm_max = 10.58417
policy_parameters_changed = true
primary_model_selected = false
pass = true
```

完整 JSON 进一步审计：

```text
decision_dt:
  1 -> 89 transitions
  2 -> 3 transitions
```

三条 `dt=2` 正好对应真正执行的 3 次 Analyse；其余 targeted requests 在无合法 observable target 时按 shared resolver 正确 fallback 到 Sleep。

```text
response reward distribution:
  0  -> 85
 -1  -> 4
 -2  -> 3
sum = -10
```

5 个 Blue agent 各有一个 terminal transition。

初始 policy entropy 约等于 `ln(6)`，与 fresh near-uniform candidate policy 一致；该 smoke 不用于评估策略性能。`clip_grad_norm_` 报告的是 clipping 前 norm，因此 `grad_norm > 0.5` 不代表 max-grad-norm contract 失效。

---

## 9. PASS conditions result

```text
[x] offline protocol tests PASS
[x] train seed only
[x] all 5 Blue agents >=1 completed transition
[x] replay count == PPO count >0
[x] requested action == selected plan[0] 100%
[x] reward/dt/done PPO-replay alignment 100%
[x] cache split=train, exact model namespace, agent-aware v2
[x] next-context exact-state guard PASS
[x] primary_model_selected=false
[x] all agents terminal
[x] PPO update finite
[x] policy parameters changed
[x] cache misses unchanged during optimizer
[x] live transactions unchanged during optimizer
[x] no validation/calibration/test
[x] no formal checkpoint retained
```

结论：**PASS / CLOSED**。

---

## 10. After PASS

进入 amended provisional stage：

```text
3 formal LLM variants
x fresh provisional PPO per variant
x train environment only
x staged <=20k decision transitions / variant
-> B0.2 per-model OOD/model-exploitation audit
```

Frozen cumulative provisional stages：

```text
2k -> 5k -> 10k -> 20k
```

每个 stage 后先做 B0.2。若 PASS 则停止该 variant 的 provisional collection；若仅 action coverage incomplete 才进入下一 stage；真实 model-shift FAIL 直接停止并 reopen shared WM path。

任何 provisional PPO weight 都不得进入正式论文结果；B0.2 完成后 formal PPO 必须重新初始化。
