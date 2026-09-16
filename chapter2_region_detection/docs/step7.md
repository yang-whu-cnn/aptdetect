# Step 7 — Integration Smoke Tests

日期：2026-09-16  
状态：**SOURCE FIXED / LOCAL 22+100 REGRESSION PENDING / REAL SMOKE RERUN PENDING**

---

## 1. Purpose

Step 7 只做集成联调，不做论文主实验、不做 beta tuning、不训练 PPO/LLM。

它验证已经冻结的：

```text
FormalState D=27
-> UG-CEM planner
-> restored Step-6 per-agent normalizer
-> SharedRolloutEvaluator
-> frozen WM/reward predictor
-> shared requested->canonical action semantics
-> CybORG target resolver / action adapter
-> multi-tick async decision epochs
-> five-agent CC4 execution
```

## 2. Step-6 normalizer restore

新增：

```text
baselines/ug_cem_apt/normalizer_bundle.py
```

Step 7 不允许重新创建一个空白 `UGUncertainty` 后直接规划；必须从：

```text
outputs/ug_cem_v2/step6/ug_normalizers_online.pt
```

恢复 blue_agent_0..4 各自的：

- obs_mean[27]；
- obs_std[27]；
- horizon_std[4]；
- internal H=4 / D=27 shape contract。

loader 同时强制检查：

- format_version=1；
- source_split=calibration；
- calibration seeds=3000..3007；
- 100 calls/agent；
- freeze_after_warmup=False；
- online_updates_after_warmup=True；
- exactly five Blue agents；
- finite / positive std。

因此 Step 6 calibration 不会在 Step 7 被意外丢弃。

## 3. Local smoke — development only

新增 runner：

```text
formal_experiments/evaluation/run_ug_cem_step7_smoke.py
```

local smoke 复用 calibration state-only 文件作为 development-only state adapter。

严格按任务书执行两档：

```text
1 episode x 20 planner decisions
5 episodes x 100 planner decisions
```

具体 local seeds：

```text
short: 3000
long : 3000..3004
```

每个 episode 边界执行 `planner.reset_episode()`，只清 MPC warm-start；Step-6 normalizer 继续 source-faithful online EMA。

local smoke 检查：

- planner action ID ∈ [0,3]；
- predicted value / uncertainty / score / latency finite；
- exact planner-call count；
- action distribution；
- episode reset 可正常工作。

该结果明确标记：

```text
development_only = True
closed_loop_cc4 = False
eligible_for_paper_table = False
```

不能进入论文主表。

## 4. Official train-seed smoke

同一个 runner 随后执行真正 CC4 closed-loop smoke：

```text
seeds = [1000, 1001]
ticks = 50 per episode
pad_spaces = False
Blue agents = blue_agent_0..4
```

环境仍为：

- EnterpriseGreenAgent；
- FiniteStateRedAgent；
- BlueFixedActionWrapper。

每个 ready decision epoch：

```text
current visible observation
-> FormalStateEncoder
-> corresponding per-agent UGCEMPlanner
-> action_id
-> shared CybORGActionAdapter
-> real action object
-> async duration scheduler
```

busy agent 不提交 filler action。

## 5. Official smoke hard checks

每次 decision 都检查：

- requested action ID 合法；
- FormalState feature17 与 adapter fallback exact consistency；
- no_op 必须执行 Sleep；
- valid targeted action 保持 Analyse/Remove/Restore family；
- unavailable targeted action fallback Sleep；
- action object family 与 adapter result 一致；
- executed duration 必须为 Sleep/Analyse/Remove/Restore = 1/2/3/5；
- environment 每次只推进一个 global tick；
- scheduler 不错过 next decision epoch；
- 5 个 Blue agent 都产生 planner decisions；
- predicted value / uncertainty / latency finite；
- online normalizer 始终 finite。

official smoke 记录：

- per-agent decision counts；
- requested action distribution；
- executed family distribution；
- fallback count/rate；
- official reward by agent（仅诊断）；
- planning latency；
- predicted return；
- uncertainty；
- normalizer finite。

它不读取 hidden Red truth，不计算 incident bookkeeping，也不使用 validation/test seeds。

## 6. Episode reset semantics

official smoke 使用同一组 5 个 planner 顺序跑 train seeds 1000、1001。

episode boundary：

```text
planner.reset_episode()
```

只清 MPC previous solution。

Step-6 main policy 的 uncertainty EMA 继续在线更新，这与 `online_updates_after_warmup=True` 一致。

## 7. Fixed smoke planner profile

Step 7 沿用 Step 6 development/calibration planner profile：

```text
H=4
A=4
M=5
N=64
I=4
elite_ratio=0.30
alpha=0.10
prob_floor=0.01
beta=0.10
```

`beta=0.10` 在 Step 7 仍只是 integration profile，不是最终论文冻结值；Step 9 才做 equal-budget validation tuning。

## 8. Unit tests

新增：

```text
tests/test_ug_normalizer_bundle.py       8 tests
tests/test_ug_cem_step7_smoke.py        14 tests
-----------------------------------------------
Step 7 source tests                     22 tests
```

覆盖：

- bundle metadata/agent/shape/finite/positive-scale guards；
- exact per-agent restore；
- restored uncertainty 可直接 compute 而不重新初始化；
- taskbook smoke profile constants；
- local smoke exact counts / reset / development-only label；
- family duration contract；
- requested→adapter resolution contract。

真实 CC4 episodes 不塞进 unittest；它们作为 Step-7 real system smoke 单独运行。

## 9. Regression count

Step 2–6 combined 已为 78 tests。

加入 Step 7：

```text
Step 2–6  78
Step 7    22
-------------
TOTAL    100
```

## 10. Local test Gate

先执行：

```bash
python -m unittest \
  tests.test_ug_normalizer_bundle \
  tests.test_ug_cem_step7_smoke -v
```

目标：

```text
Ran 22 tests
OK
```

再执行 Step 2–7 regression：

```bash
python -m unittest \
  tests.test_categorical_cem \
  tests.test_ug_uncertainty \
  tests.test_shared_rollout_evaluator \
  tests.test_ug_cem_planner \
  tests.test_ug_normalizer_warmup \
  tests.test_ug_normalizer_calibration \
  tests.test_ug_normalizer_bundle \
  tests.test_ug_cem_step7_smoke -v
```

目标：

```text
Ran 100 tests
OK
```

CUDA unavailable skip 规则与 Step 4 相同。

## 11. Real smoke command

测试通过后执行：

```bash
python -m formal_experiments.evaluation.run_ug_cem_step7_smoke --device cpu
```

输出：

```text
outputs/ug_cem_v2/step7/step7_smoke_report.json
```

最终 summary 必须至少满足：

```text
local_short_calls = 20
local_long_calls = 500
official_seeds = [1000, 1001]
official_ticks = [50, 50]
pass = True
```

并且两个 official episodes 中五个 agent decision count 都 >0、normalizer 全 finite。

## 12. Close condition

Step 7 只有在：

- Step 7 22 tests PASS；
- Step 2–7 100-test regression 最终 OK；
- real local smoke 20 + 500 calls PASS；
- official train-seed 1000/1001 ×50 ticks PASS；

之后才能 CLOSED。

Step 7 CLOSED 后进入 Gate B；此时仍不能把 smoke 结果作为论文最终结果。

## 13. Current conclusion

```text
Step-6 normalizer restore    : READY
Local smoke harness          : READY
Official CC4 smoke harness   : READY
Five-agent async scheduler   : READY
Adapter/duration checks      : READY
Unit-test source             : READY (22)
Local test execution         : PENDING
Real smoke execution         : PENDING

FINAL STATUS: SOURCE READY / TEST + REAL SMOKE PENDING
```


## 14. Official-smoke controller tick correction

首次 real smoke 时 local smoke 与全部 unit regression 已通过，但 official episode 在结束检查处报：

```text
RuntimeError: official smoke ended at tick 49, expected 50
```

该失败发生在 episode 结束后的额外 assertion，而不是 planner / adapter / normalizer / scheduler 中间检查。

对照 Gate A collector 后确认：Gate A 从未把 `controller.step_count` 的最终编号当作 episode length；CC4 controller 可以在 reset 后以 `step_count=-1` 开始，因此真实调用 50 次 `env.step()` 后最终 controller tick 为 49 是合法的。

Step 7 初版错误地把：

```text
controller final tick == requested steps
```

作为硬条件。

现改为：

```text
environment_steps_executed == requested_steps
controller_tick_end - controller_tick_start == environment_steps_executed
all_agents_done == True
```

并继续保留每次 step 内：

```text
tick_end == tick_start + 1
```

因此不是简单放宽 `49`，而是直接按真实 `env.step()` 次数验证 50-tick smoke，同时要求 controller 连续推进且所有 Blue agent 正常 terminated/truncated。

report 新增：

- `environment_steps_executed`；
- `controller_tick_start`；
- `controller_tick_end`；
- `all_agents_done`。

并新增 2 个 regression tests：

1. `controller -1 -> 49` + 50 次 env.step 合法；
2. 49 次 env.step 或 50 次后 agents 未结束必须拒绝。

因此 Step 7 source tests 从 20 增至 22，Step 2–7 combined 从 98 增至 100。

该修复后必须重新执行 22/100 tests，再 rerun real smoke；旧失败不能作为 Step 7 PASS。

## 15. Native CC4 scenario-step correction (supersedes Section 14 interpretation)

第二次 real smoke 在修复后仍报告：

```text
official smoke environment-step count mismatch:
executed=49, requested=50
```

进一步核对 CC4 官方源码后确认，Section 14 中“reset tick=-1，因此 50 次 env.step 后到 49”的解释不准确，现正式废止。

CC4 当前源码语义：

```text
SimulationController.reset():
    step_count = 0

SimulationController.step():
    step_count += 1

EnterpriseScenarioGenerator.determine_done():
    return step_count >= (steps - 1)
```

因此 `EnterpriseScenarioGenerator(steps=50)` 的原生 episode 是：

```text
controller tick labels: 0..49
scenario ticks:         50
post-reset env.step():  49
terminal tick:          49
```

这也与 CC4 official evaluation 的行为一致：evaluation 把 `steps=EPISODE_LENGTH` 传给 scenario，并在 term/trunc 全部为真时提前退出最多 `EPISODE_LENGTH` 次的调用循环。

Step 7 现在按原生 CC4 语义验证：

```text
expected_terminal_tick = scenario_steps - 1
controller_tick_end == expected_terminal_tick
post_reset_env_steps == expected_terminal_tick - controller_tick_start
controller_tick_end - controller_tick_start == post_reset_env_steps
all_agents_done == True
```

对正式 `scenario_steps=50` 且 reset 后 start=0：

```text
controller_tick_end = 49
post_reset_env_steps = 49
scenario_ticks = 50
```

runner report/summary 现在明确区分：

- `scenario_steps` / `scenario_ticks`；
- `post_reset_env_steps` / `environment_steps_executed`；
- `controller_tick_start/end`。

因此任务书中的“2 episodes × 50 ticks”解释为 CC4 `EnterpriseScenarioGenerator(steps=50)` 的 **50 个 scenario ticks**，而不是 50 次 post-reset wrapper `env.step()` API 调用。

原 Section 14 保留作为第一次失败历史，但其 `-1 -> 49 + 50 env.step` 解释已被本节正式 supersede。

Step 7 tests 数量不变：bundle 8 + smoke 14 = 22；combined Step2–7=100。需要重新执行 22/100 regression 和 real smoke。