# Gate B4 — PPO Core / Duration-Aware GAE

日期：2026-09-17  
状态：**SOURCE READY / LOCAL CORE TEST PENDING**

---

## 1. Preconditions

已关闭：

```text
B0.1 frozen WM audit        PASS / CLOSED
B1 response reward          PASS / CLOSED
B2 multi-LLM prior study    PASS / CLOSED
B3 posterior 46D regression PASS / CLOSED
```

Primary LLM 仍未选择；B4 后续正式实验必须让三个 LLM variant 分别训练自己的 shared PPO policy。

## 2. Scope of this gate

本阶段只冻结并验收：

1. candidate-wise actor/critic 网络；
2. candidate permutation contract；
3. duration-aware discount / GAE；
4. clipped PPO loss；
5. advantage normalization；
6. action index / finite guards。

本阶段**不**执行：

- CC4 rollout collection；
- OFOX live API generation；
- provisional PPO；
- B0.2 OOD audit；
- formal 100k PPO training；
- validation checkpoint selection；
- test evaluation。

## 3. Frozen posterior input

每 candidate 固定 46D：

```text
state                  27
plan one-hot           16
prior preference        1
predicted value         1
predictive uncertainty  1
--------------------------
feature dim            46
```

不得增加 provider/model ID、API latency/cost、rationale text、action cost、delay、early-warning 或 hidden evidence。

## 4. PPO network

Production source：

```text
formal_experiments/ours/ppo_core.py
```

Architecture：

```text
candidate features [...,K,46]
    -> shared Linear(46,128) + ReLU
    -> shared Linear(128,128) + ReLU

actor:
    shared Linear(128,1)
    -> K logits
    -> Categorical(candidate index)

critic:
    mean pool K embeddings
    -> Linear(128,128) + ReLU
    -> Linear(128,1)
```

Main experiment K=6；implementation 支持 variable K。

Required invariants：

- actor candidate-permutation equivariant；
- critic candidate-order invariant；
- candidate rows 不 flatten 成 K×46；
- PPO action 是 candidate index，不是 A4 action ID；
- actor output gain=0.01；
- critic output gain=1.0；
- hidden layers orthogonal gain=sqrt(2)；
- biases=0。

## 5. Duration-aware discount / GAE

冻结：

```text
gamma_tick = 0.99
lambda = 0.95
gamma_t = gamma_tick ** decision_dt

delta_t
= real_response_reward_t
  + gamma_t * (1-done_t) * V(next_t)
  - V(t)

GAE_t
= delta_t
  + gamma_t * lambda * (1-done_t) * GAE_{t+1}
```

`lambda` 是 per decision transition；禁止使用 `lambda ** decision_dt`。

`real_response_reward_t` 必须是该真实 CC4 decision interval 的 accumulated response reward。

明确禁止：

- WM predicted value 作为 PPO reward target；
- LLM prior 作为 reward；
- action cost / delay / early-warning reward 回流。

Terminal transition 必须同时：

- 禁止 next-value bootstrap；
- 截断 GAE continuation。

实现已修正 device safety：GAE recursion scalar / done / value tensors 与 reward tensor 保持同 device。

## 6. PPO objective

冻结：

```text
learning_rate = 3e-4
clip_epsilon = 0.2
value_coef = 0.5
entropy_coef = 0.01
max_grad_norm = 0.5
advantage_normalization = true
update_epochs = 5
minibatch_size = 64
rollout_length = 128 decision transitions
```

Clipped surrogate：

```text
ratio = exp(new_log_prob - old_log_prob)
policy_loss = -mean(min(ratio*A, clip(ratio,1-eps,1+eps)*A))
value_loss = 0.5 * MSE(V, return)
total = policy_loss + 0.5*value_loss - 0.01*entropy
```

同时记录 approx-KL 与 clip fraction。

## 7. Training budgets frozen for later stages

本 core gate 不执行训练，但配置先冻结：

```text
provisional PPO <= 20,000 real decision transitions
formal PPO      <= 100,000 real decision transitions/model/seed
checkpoint every 10,000
minimum 3 PPO training seeds/model
```

Provisional checkpoint 永不进入正式结果；B0.2 后 formal PPO 从新初始化。

## 8. Unit tests

```text
tests/test_gate_b4_ppo_core.py
```

当前 14 tests 覆盖：

1. unbatched/batched K=6 shapes；
2. variable K；
3. actor permutation equivariance / critic invariance；
4. deterministic selected candidate permutation mapping；
5. orthogonal output gains / zero biases；
6. invalid feature/action guards；
7. `gamma_tick ** dt`；
8. duration-aware GAE manual reference；
9. λ per decision, not `lambda**dt`；
10. terminal bootstrap/GAE cut；
11. GAE API excludes WM/LLM reward target；
12. advantage normalization；
13. clipped PPO loss/clip fraction reference；
14. length/nonfinite loss guards。

## 9. PASS

B4 core PASS requires：

```text
14/14 unit tests PASS
46D input only
actor permutation equivariance PASS
critic permutation invariance PASS
duration GAE reference PASS
terminal boundary PASS
no predicted reward leakage
PPO loss finite/reference PASS
config matches v2.3 taskbook
```

No live API or environment rollout is required for this gate.

## 10. After PASS

Only after B4 core closes：

```text
B4 collector/trainer integration
-> one-model tiny pipeline smoke
-> provisional PPO on train seeds only (<=20k)
-> B0.2 policy-induced OOD/model-exploitation audit
```

Only B0.2 PASS permits formal 3-model × >=3-seed PPO training.
