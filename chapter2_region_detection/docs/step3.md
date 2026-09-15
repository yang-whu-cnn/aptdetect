# Step 3 — UG Uncertainty 审核记录

> 分支：`ug-cem-apt`  
> 日期：2026-09-15  
> 最终状态：**PASS**

## 1. 阶段目标

本阶段实现独立的 UG trajectory uncertainty 模块。

输入：

```text
next_states: [H,N,M,D]
```

输出：

```text
uncertainty: [N]
```

其中 H 为计划长度，N 为候选计划数，M 为世界模型 ensemble 成员数，D 为状态维度。

本阶段不负责 world-model rollout、reward、beta penalty、MPC、PPO、LLM、CC4 或 Chapter 1。

## 2. 实现文件

```text
chapter2_region_detection/baselines/ug_cem_apt/uncertainty.py
chapter2_region_detection/tests/test_ug_uncertainty.py
```

## 3. 官方机制对照

已重新对照 UG 官方：

```text
mbrl-lib-uncertainty_guided_planning/mbrl/planning/trajectory_opt.py
```

当前实现保留以下核心顺序：

```text
初始化 obs_mean / obs_std
→ EMA 更新状态统计
→ 状态归一化
→ 沿 M 维计算模型分歧
→ 对 D 维求平均
→ EMA 更新 horizon_std
→ horizon normalization
→ 对 H 求平均
→ uncertainty [N]
```

默认初始化与官方一致：

```text
obs_mean = 0
obs_std = 0.1
horizon_std = 0.01
alpha = 0.01
```

## 4. Canonical shape 审核

```text
[H,N,M,D]
  ↓ std over M
[H,N,D]
  ↓ mean over D
[H,N]
  ↓ horizon normalization
[H,N]
  ↓ mean over H
[N]
```

审核结果：维度语义正确。

## 5. 数值稳定性

实现使用很小的 `eps` 防止分母过小，并对输入和最终输出进行 finite 检查。

这是数值稳定性适配，不改变正常条件下的 UG 公式。

## 6. update_stats / reset

`update_stats=True`：按官方方式更新 running statistics。

`update_stats=False`：不执行 EMA 更新；如果尚未初始化，只创建官方默认初值。

`reset()`：清空 running statistics，下一次调用重新初始化。

初始化后 H 和 D 不允许静默改变；N 可以改变。

## 7. 测试结果

本地共通过 12 项测试，包括：

1. identical members -> uncertainty 接近 0；
2. model disagreement 增大 -> uncertainty 增大；
3. 输出 shape 正确；
4. 连续计算保持 finite；
5. running statistics 更新正确；
6. reset 正确；
7. update_stats=False 不修改已有统计量；
8. update_stats=False 未初始化时使用官方初值；
9. 与官方公式进行数值 reference 对照；
10. H / D 改变时抛异常；
11. 非法配置抛异常；
12. 非法输入抛异常。

其中 reference test 同时核对最终 uncertainty、obs_mean、obs_std 和 horizon_std。

## 8. 模块隔离

确认 `uncertainty.py` 不依赖：

```text
world model
Categorical CEM
CC4 / CybORG
Chapter 1
PPO
LLM
reward
beta
```

官方评分中的：

```text
predicted_return - beta * uncertainty / (iteration + 1)
```

留到 Step 5 Planner 集成和审核。

## 9. Git 范围

本阶段实现提交只新增：

```text
uncertainty.py
test_ug_uncertainty.py
```

未修改 `chapter2_region_detection/src/`，未修改 UG 官方源码。

## 10. 最终结论

```text
Canonical shape          PASS
Official formula         PASS
State normalization      PASS
Horizon normalization    PASS
Numerical stability      PASS
Running stats / reset    PASS
Reference test           PASS
Module isolation         PASS
Git scope                PASS

FINAL STATUS: PASS
```

Step 3 正式完成并冻结。

当前下一阶段：

```text
Gate A — 动作语义与环境映射冻结
```

Gate A 完成后再进入 Step 4 — Vectorized World Model Rollout Evaluator。
