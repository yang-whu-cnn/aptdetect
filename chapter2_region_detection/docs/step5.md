# Step 5 — UGCEM Planner

日期：2026-09-16  
状态：**SOURCE READY / LOCAL TEST PENDING**

---

## 1. Design verification before implementation

实现前已重新对照：

- `docs/UG_CEM_APT_REPRODUCTION_PLAN.md` Step 5；
- Step 2 `CategoricalCEMOptimizer`；
- Step 3 `UGUncertainty`；
- Step 4 `SharedRolloutEvaluator`；
- vendored UG official source：`mbrl-lib-uncertainty_guided_planning/mbrl/planning/trajectory_opt.py`。

审核结论：当前 Step 5 设计与论文/任务书主线一致，没有需要修改研究方案的阻塞项。

关键对应关系：

```text
UG official / taskbook
    CEM optimization
    + ensemble trajectory disagreement
    + uncertainty-guided score
    + previous-solution MPC shift

current implementation
    CategoricalCEMOptimizer
    + SharedRolloutEvaluator
    + UGUncertainty
    + UGCEMPlanner warm-start
```

官方源码中的 uncertainty-guided 更新明确为：

```text
values -= beta * pop_std / (i + 1)
```

当前 Step 5 保留完全相同的 iteration denominator：

```text
J_i = G_i - beta * omega_i / (k + 1)
```

其中 `k` 为 zero-based CEM iteration。

## 2. Formal planner role

新增：

```text
baselines/ug_cem_apt/planner.py
```

`UGCEMPlanner` 只负责组合：

```text
Categorical CEM
+ SharedRolloutEvaluator
+ UGUncertainty
+ MPC warm-start
```

不负责：

- CC4 environment execution；
- target resolver；
- host selection；
- LLM；
- PPO；
- old action cost；
- lambda_delay；
- control_traffic；
- test-time heuristic / bandit boost。

## 3. Formal score

每个 CEM iteration：

```text
rollout = SharedRolloutEvaluator(state, plans)
G_i     = rollout.expected_return
omega_i = UGUncertainty(rollout.next_states)

score_i = G_i - beta * omega_i / (iteration + 1)
```

score 越大越好，直接交给 Step 2 Categorical CEM 做 elite selection。

`beta=0` 时不使用另一套实现：

```text
CEM-APT = same UGCEMPlanner with beta=0
```

uncertainty 仍通过同一 machinery 计算用于统一诊断，但不影响 score。

## 4. MPC / decision-epoch semantics

每个真实 decision epoch 都重新调用 `plan()`：

1. 第一次 planning 使用 uniform categorical probabilities；
2. CEM 完整优化 H=4 plan；
3. 返回所有 iteration 中真实 sampled 的最高分 plan；
4. 外部只执行 `best_plan[0]`；
5. 将本次 `final_probs` 左移一行；
6. 最后一行补 A=4 uniform `[0.25,0.25,0.25,0.25]`；
7. 下一 decision epoch 将该 shifted matrix 作为 CEM `initial_probs`。

这与任务书冻结的 categorical MPC adaptation 一致。

注意：decision epoch 已由 Gate A 冻结，所以无论动作 duration 是 1/2/3/5 ticks，下一次 planner 调用都表示向前推进了一个 high-level decision，因此概率矩阵只左移一行。

## 5. Episode reset and uncertainty statistics

vendored UG official `TrajectoryOptimizer.reset()` 只重置 previous solution，不自动清空 CEM 内部 uncertainty running statistics。

当前实现对应为：

```text
planner.reset_episode()
    -> clear MPC warm-start only

planner.reset()
    -> alias of reset_episode()

planner.reset_uncertainty_stats()
    -> explicit UGUncertainty.reset()
```

因此正常 episode reset 不会偷偷重置 uncertainty normalization。

这也为 Step 6 的 warm-up 提供正确接口：每次 warm-up planner call 前执行 `reset_episode()`，即可保证不使用 previous solution，同时让 uncertainty running stats 跨 warm-up calls 持续更新。

## 6. Parameters and freeze boundary

Step 5 不提前做 Step 9 的超参数选择。

`UGCEMPlannerConfig` 当前只持有集成层参数：

```text
beta = 0.10                  # development/default only
update_uncertainty_stats=True
```

`beta=0.10` 不是论文最终最优值，也不是 test-time 冻结值；正式 beta 由 Step 9 使用统一 validation budget 从任务书候选集合选择。

CEM 的 population / iterations / elite_ratio / alpha / prob_floor 继续由 `CategoricalCEMConfig` 提供。

正式 Step 5 planner 强制检查：

```text
H = 4
A = 4
M = 5
same device for CEM / rollout / uncertainty
```

因此 Step 2 历史 generic default `n_actions=5` 无法误进入正式 planner。

## 7. Required debug output

`UGCEMPlanResult` 返回：

- `action_id` = best_plan[0]；
- `best_plan`；
- `best_score`；
- best plan 对应 `expected_return`；
- best plan 对应 `uncertainty`；
- `final_probs`；
- mean `final_probs_entropy`；
- per-step entropy；
- `iteration_history`；
- `planning_latency_sec`；
- `warm_start_used`。

`iteration_history` 每轮至少保存：

- iteration index；
- iteration best score；
- iteration best expected return；
- iteration best uncertainty；
- mean score；
- mean expected return；
- mean uncertainty。

## 8. Best sampled plan consistency

Step 2 已冻结 CEM 返回“所有 iteration 中实际评估过的最高分 sampled plan”，不对 categorical action ID 做均值。

Step 5 同时独立记录 objective 内的 global best debug 信息，并在 CEM 返回后强制检查：

```text
CEM best_plan == planner debug best_plan
CEM best_score == planner debug best_score
```

若不一致直接报错，避免 debug value/uncertainty 对不上最终 plan。

## 9. Tests

新增：

```text
tests/test_ug_cem_planner.py
```

共 14 tests：

1. `/(iteration+1)` score 公式精确检查；
2. beta=0 的 CEM-APT 同 machinery；
3. best sampled plan + execute first action；
4. `final_probs` 左移 + final uniform row；
5. episode reset 只清 warm-start；
6. uncertainty reset 必须显式调用；
7. 每次 `plan()` 都重新 CEM planning；
8. iteration history + 全部 required debug fields；
9. final-probability entropy reference；
10. `update_uncertainty_stats` flag 透传；
11. invalid beta guard；
12. A/H/M/device component contract guards；
13. 与真实 `CategoricalCEMOptimizer + UGUncertainty` integration smoke；
14. warm-start shape / finite guards。

## 10. Regression boundary

Step 5 不修改：

- Step 2 Categorical CEM；
- Step 3 UGUncertainty；
- Step 4 SharedRolloutEvaluator；
- Gate A frozen model-space contract。

因此本阶段本地关闭时建议同时回归四个 planner components。

当前测试数量：

```text
Step 2 Categorical CEM       16
Step 3 UG uncertainty        12
Step 4 shared rollout        13
Step 5 UGCEM planner         14
TOTAL                        55
```

## 11. Local Gate

在 `.venv_cc4`、`chapter2_region_detection`：

先跑 Step 5：

```bash
python -m unittest tests.test_ug_cem_planner -v
```

预期：

```text
Ran 14 tests
OK
```

再跑 Step 2–5 combined regression：

```bash
python -m unittest \
  tests.test_categorical_cem \
  tests.test_ug_uncertainty \
  tests.test_shared_rollout_evaluator \
  tests.test_ug_cem_planner -v
```

预期共 55 tests；若 CUDA 不可用，Step 4 CUDA test 允许 skip，最终 unittest 必须 `OK`。

## 12. Close condition

Step 5 只有在：

- 14 个 planner tests PASS；
- Step 2–5 combined regression 最终 OK；

之后才能 CLOSED。

Step 5 CLOSED 后进入 Step 6 — Uncertainty Normalizer Warm-up。

## 13. Current conclusion

```text
Paper/taskbook alignment       : PASS
Official-source alignment      : PASS
Score formula                  : READY
Categorical CEM integration    : READY
Shared rollout integration     : READY
UG uncertainty integration     : READY
MPC warm-start                 : READY
Episode reset semantics        : READY
CEM-APT beta=0 path            : READY
Debug output                   : READY
Unit-test source               : READY (14)
Local execution                : PENDING

FINAL STATUS: SOURCE READY / LOCAL TEST PENDING
```
