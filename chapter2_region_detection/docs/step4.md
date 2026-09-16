# Step 4 — Vectorized Shared Rollout Evaluator

日期：2026-09-16  
状态：**SOURCE READY / LOCAL TEST PENDING**

---

## 1. Purpose

Step 4 提供 LWM-RL、UG-CEM-APT、CEM-APT 共用的 model-based rollout evaluator。

输入：

```text
state : [D]
plans : [N,H]
```

正式 contract：

```text
D = 27
A = 4
H = 4
M = 5 for formal checkpoint
gamma_tick = 0.99
```

输出：

```text
next_states    : [H,N,M,D]
member_returns : [N,M]
expected_return: [N]
```

输出保持 `torch.Tensor`，可直接传入现有 `UGUncertainty.compute(next_states)`。

## 2. Source

新增：

```text
shared/rollout_evaluator.py
```

主要类型：

- `SharedRolloutConfig`；
- `SharedRolloutResult`；
- `SharedRolloutEvaluator`。

## 3. Frozen rollout semantics

每个 candidate plan：

1. 从同一个当前 FormalState 开始；
2. ensemble member 在整个 H=4 horizon 内固定，不跨步切换；
3. 每一步根据该 member 当前 predicted state，把 requested action 通过 `shared.model_space_action.canonicalize_requested_tensor` 转为 canonical action；
4. 调用该 fixed member 的 `predict_member_mean_tensor`；
5. 不使用 predicted variance 做随机采样；
6. 用 shared `ResponseRewardPredictor.predict_tensor(s,a,s_next)` 预测该 decision reward；
7. 按 canonical executed action 的真实 duration 累积 elapsed global ticks；
8. discount 使用 `gamma_tick ** cumulative_elapsed_ticks`；
9. 得到每个 plan × member return；
10. 对 M 个 member 求均值得到 expected return。

因此 evaluator 与 A4.6a 的 model-space semantics 一致，但现在是正式 reusable shared planner component。

## 4. Vectorization rule

实现只在：

```text
for step in H:
    for member in M:
        batch all N plans
```

因此 world-model forward 数量：

```text
M * H
```

而不是：

```text
N * M * H
```

正式 M=5、H=4 时，每次 population evaluation 为 20 次 batched member forward，不随 N 线性增加 forward-call 次数。

reward predictor 同样为 M*H 次 batched calls。

## 5. Explicit exclusions

`SharedRolloutEvaluator` 不包含：

- Categorical CEM；
- UG uncertainty penalty；
- beta；
- elite selection；
- MPC warm-start；
- PPO；
- LLM；
- candidate ranking；
- host resolver；
- CC4 environment execution。

这些逻辑分别属于后续 Step 5 / Gate B / fair comparison harness。

## 6. Tests

新增：

```text
tests/test_shared_rollout_evaluator.py
```

共 13 个 tests：

1. N=1 / D=27 output shape；
2. N=64 output shape；
3. fixed-member identity through H；
4. forward-call count exactly M*H，且每次 batch=N；
5. deterministic mean path / no distribution sampling；
6. vectorized result == slow reference；
7. duration-aware discount；
8. requested→canonical fallback 对 reward/duration 的影响；
9. expected_return == mean(member_returns, M)；
10. CPU device + finite output；
11. CUDA device（若本机 CUDA 不可用则 unittest skip）；
12. nonfinite / invalid input guards；
13. component contract guards，包括 selected absolute WM 与 same-device requirement。

## 7. Local Gate

在 `.venv_cc4`、`chapter2_region_detection`：

```bash
python -m unittest tests.test_shared_rollout_evaluator -v
```

预期：

```text
Ran 13 tests
OK
```

若 CUDA 不可用，`test_cuda_device` 显示 `skipped` 是允许的；最终 unittest 必须为 `OK`。

随后建议回归 Gate A final integration：

```bash
python -m unittest tests.test_gate_a_final_integration -v
```

预期：8 tests，OK。

## 8. Close condition

Step 4 只有在：

- 13 个 Step-4 tests 全部 pass/allowed skip；
- Gate A final integration 8/8 继续 PASS；

之后才能 CLOSED。

Step 4 CLOSED 后才进入 Step 5 UGCEM Planner。

## 9. Current conclusion

```text
Shared evaluator source : READY
Output contract         : READY
Fixed-member rollout    : READY
Vectorized M*H path     : READY
Duration discount       : READY
Canonical fallback      : READY
Unit-test source        : READY (13)
Local test execution    : PENDING

FINAL STATUS: SOURCE READY / LOCAL TEST PENDING
```
