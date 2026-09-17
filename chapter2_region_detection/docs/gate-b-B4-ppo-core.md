# Gate B4 — PPO Core / Duration-Aware GAE

日期：2026-09-17  
状态：**PASS / CLOSED**

---

## 1. Preconditions

已关闭：

```text
B0.1 frozen WM audit         PASS / CLOSED
B1 response reward           PASS / CLOSED
B2 multi-LLM prior study     PASS / CLOSED
B3 posterior 46D regression  PASS / CLOSED
```

Primary LLM 仍未选择；正式实验必须让三个 LLM variant 分别训练自己的 shared PPO policy。

## 2. Scope

本 Gate 冻结并验收：

1. candidate-wise actor/critic；
2. candidate permutation contract；
3. duration-aware discount / GAE；
4. clipped PPO loss；
5. advantage normalization；
6. action index / finite guards。

不执行 CC4 rollout、OFOX live generation、provisional PPO、B0.2、formal 100k PPO 或 test evaluation。

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

不得增加 provider/model ID、API latency/cost、rationale、action cost、delay、early-warning 或 hidden evidence。

## 4. Network

Production：

```text
formal_experiments/ours/ppo_core.py
```

```text
candidate [...,K,46]
-> shared Linear(46,128)+ReLU
-> shared Linear(128,128)+ReLU

actor: shared Linear(128,1) -> K logits -> Categorical(candidate index)
critic: mean-pool K embeddings -> Linear(128,128)+ReLU -> Linear(128,1)
```

Main K=6；implementation 支持 variable K。

Frozen invariants：

- actor candidate-permutation equivariant；
- critic candidate-order invariant；
- 不 flatten K×46；
- PPO action=candidate index；
- actor output gain=0.01；
- critic output gain=1.0；
- hidden orthogonal gain=sqrt(2)；
- biases=0。

## 5. Duration-aware GAE

```text
gamma_tick = 0.99
lambda = 0.95
gamma_t = gamma_tick ** decision_dt

delta_t = real_response_reward_t
        + gamma_t * (1-done_t) * V(next_t)
        - V(t)

GAE_t = delta_t
      + gamma_t * lambda * (1-done_t) * GAE_{t+1}
```

`lambda` 是 per decision transition，不是 `lambda ** decision_dt`。

PPO target 只允许真实 CC4 accumulated response reward；WM predicted value 只是 observation/evidence，禁止作为 PPO reward target。

Terminal transition 同时禁止 next-value bootstrap 并截断 GAE continuation。

GAE recursion 已做 device-safe 修正。

## 6. PPO objective / defaults

```text
learning_rate = 3e-4
rollout_length = 128
update_epochs = 5
minibatch_size = 64
clip_epsilon = 0.2
gae_lambda = 0.95
entropy_coef = 0.01
value_coef = 0.5
max_grad_norm = 0.5
advantage_normalization = true
```

同时记录 approximate KL 与 clip fraction。

后续预算已冻结：

```text
provisional PPO <= 20,000 real decision transitions
formal PPO      <= 100,000 real decision transitions/model/seed
checkpoint every 10,000
minimum 3 PPO seeds/model
```

Provisional checkpoint 永不进入正式结果；B0.2 后 formal PPO 从新初始化。

## 7. Local acceptance

用户本地执行：

```text
python -m unittest tests.test_gate_b4_ppo_core -v
```

结果：

```text
Ran 14 tests
OK
```

覆盖：network shapes、variable K、actor equivariance、critic invariance、selected-candidate permutation mapping、orthogonal init、guards、gamma^dt、manual GAE、lambda semantics、terminal cut、reward leakage guard、advantage normalization、clipped PPO reference。

结论：

```text
Gate B4 PPO core = PASS / CLOSED
```

## 8. Next

下一阶段不是正式训练：

```text
B4 rollout/trainer integration
-> pure asynchronous rollout/update tests
-> one-model tiny CC4 pipeline smoke
-> provisional PPO <=20k train-only
-> B0.2 OOD/model-exploitation audit
```

只有 B0.2 PASS 后才允许 formal 3-model × >=3-seed PPO training。
