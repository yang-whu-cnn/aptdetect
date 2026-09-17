# Gate B4 — One-Model Tiny CC4 Pipeline Smoke

日期：2026-09-17  
状态：**SOURCE READY / LOCAL OFFLINE TEST + LIVE SMOKE PENDING**

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

本阶段仍不是 provisional/formal PPO experiment。

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

默认：

```text
model_alias = llm_l_gemini35_flash_lite
train_seed = 1000
scenario_steps = 20
ppo_seed = 20260917
device = cpu
pad_spaces = false
```

使用 Tier-L 的唯一原因是 B2 已验证其接口且成本/延迟最低，适合作为工程 smoke。

**这不是模型选择。** `primary_model_selected` 必须仍为 false。

允许通过 CLI 改用 registry 中另一个已验证 model alias 做工程复测，但不能据此改变正式 multi-LLM protocol。

---

## 4. Candidate-context semantics

每个 ready decision state 只构造一次 candidate context：

```text
state
-> train cache lookup
-> miss: bounded live provider transaction
-> PriorBatch K6/H4
-> frozen WM/reward rollout
-> [6,46] posterior features
```

当动作完成且 episode 未结束时，在真实 `next_state` 上预构造下一 candidate context，并用其 critic value 作为 `V(next)`。

下一 decision epoch 必须复用这份 context；通过 exact float32 D27 SHA256 验证 next state 完全一致。

目的：

- critic bootstrap 与真正下一策略输入一致；
- 不因 `V(next)` 额外重复调用 LLM；
- 不因下一 decision 再次重复 WM rollout。

---

## 5. Runtime action contract

PPO action 是 candidate index：

```text
candidate_index i
-> requested_action_id = plans[i][0]
```

只有 `plan[0]` 进入真实 CC4。

随后统一走：

```text
requested A4
-> shared observable target resolver
-> shared CybORG adapter
-> actual executed action
```

Target 不由 LLM/PPO直接选择。

无合法 observable target 时 targeted request 可以 fallback 到 Sleep；requested action 保留用于 audit。

---

## 6. Reward / trajectory alignment

每个完成 transition 必须同时进入：

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

PPO reward 必须直接取 `DecisionEpochTransition.response_reward`。

Terminal：

```text
critic_next_value = 0
```

非 terminal：critic next value 来自预构造的真实 next-state posterior context。

Hidden Red truth 只允许进入既有 incident-response reward bookkeeping，不得进入 state/prompt/posterior/policy。

---

## 7. PPO update

Episode 完成后：

```text
formal replay transition count == PPO rollout step count
-> build per-agent duration-aware GAE
-> merge batch
-> one PPOTrainer.update()
```

Smoke 使用现有 formal optimizer defaults；其权重不保存为正式 checkpoint。

记录 optimizer 前后的 cache-miss / live-transaction counter；必须完全不变，证明 5 个 update epochs 不触发 API。

---

## 8. Artifacts

Production runner：

```text
formal_experiments/evaluation/run_b4_tiny_pipeline_smoke.py
```

Offline protocol tests：

```text
tests/test_gate_b4_tiny_pipeline_smoke.py
```

Live report：

```text
outputs/lwm_rl_v2/b4/tiny_pipeline_smoke.json
```

不保存 formal PPO checkpoint。

---

## 9. PASS conditions

全部满足：

```text
all offline protocol tests PASS
train seed only
all 5 Blue agents have >=1 completed transition
formal replay count == PPO step count >0
100% requested action == selected plan[0]
100% reward/dt/done PPO-replay alignment
cache split = train and model alias exact
next-context exact-state reuse guard never fails
primary_model_selected = false
all agents reach episode terminal
PPO update metrics finite
policy parameters change after update
cache misses unchanged during PPO optimizer epochs
live transactions unchanged during PPO optimizer epochs
no validation/calibration/test
no formal checkpoint retained
```

API/cache hits may make live API calls zero on rerun; this is allowed. First fresh run normally generates train-cache misses.

---

## 10. Failure branch

- schema/provider/cache failure → fix provider/cache integration; do not weaken parser;
- plan[0]/resolver mismatch → fix runtime integration; do not add method-specific resolver；
- replay/PPO reward or dt mismatch → stop; B4 cannot proceed；
- next-context SHA mismatch → fix scheduler/prefetch semantics；
- NaN/Inf optimizer → reopen B4 PPO integration；
- CC4 terminal/accounting mismatch → compare against frozen Step 7 semantics before changing environment rules。

不得用 validation/test 来 debug smoke。

---

## 11. After PASS

进入 amended provisional stage：

```text
3 formal LLM variants
x one identical PPO development seed each
x train environment only
x <=20k decision transitions / variant
-> B0.2 per-model + union OOD/model-exploitation audit
```

任何 provisional PPO weight 都不得进入正式论文结果；B0.2 之后 formal PPO 必须重新初始化。
