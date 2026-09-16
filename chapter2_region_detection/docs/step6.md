# Step 6 — Uncertainty Normalizer Warm-up

日期：2026-09-16  
状态：**6A/6B SOURCE READY / LOCAL 23+78 REGRESSION PENDING / REAL CALIBRATION PENDING**

---

## 1. Design verification

Step 6 对照任务书与 Step 3 / Step 5 后冻结为：

```text
UG uncertainty running normalizer
= obs_mean + obs_std + horizon_std
```

本阶段不训练新的 world model，也不改变 uncertainty 公式。

目标只是让 UG uncertainty 在正式 planner 使用前先经过 source-faithful EMA calibration，避免从：

```text
obs_mean = 0
obs_std = 0.1
horizon_std = 0.01
```

直接进入正式比较。

## 2. Source-faithful policy

任务书冻结：

- 默认 100 planner calls；
- warm-up 不使用 previous solution；
- 每 agent/planner 独立维护 normalizer；
- 记录 normalizer finite；
- 主版本 warm-up 后继续在线 EMA；
- `freeze_after_warmup=true` 仅作为敏感性版本。

当前实现保持上述语义。

Step 5 已确认 vendored UG official `TrajectoryOptimizer.reset()` 只清 previous solution，因此 warm-up 每次调用：

```text
planner.reset_episode()
planner.plan(state)
```

即可保证每次从 uniform CEM start 开始，同时保留同一个 uncertainty normalizer 的 EMA 累积。

## 3. Seed / leakage contract

新增：

```text
baselines/ug_cem_apt/normalizer_warmup.py
```

`WarmupState` 强制 provenance：

```text
train       : 1000..1031
calibration : 3000..3007
```

明确禁止：

```text
validation : 2000..2007
test       : 4000..4019
```

不仅检查 `split` 字符串，还检查 seed 必须属于该 split 的冻结范围，因此 test seed 不能伪装成 train/calibration 输入。

本阶段只消费 planner-visible 27D state，不读取 hidden compromise truth、incident label、future state 或 test result。

## 4. Per-agent / per-planner normalizer

任务书要求每 agent/planner 维护匹配自身 model/planner distribution 的 normalizer。

因此一个 `UGNormalizerWarmup` runner：

- 只绑定一个 `UGCEMPlanner`；
- 一次 run 只允许一个 `agent_name`；
- 不允许把多个 Blue agent 的 warm-up state 混进同一个 normalizer；
- 不会在不同 planner 实例之间隐式共享 `UGUncertainty`。

正式 full-system 时应为每个 agent/planner 分别执行 warm-up。

## 5. Warm-up procedure

默认：

```text
planner_calls = 100
freeze_after_warmup = False
```

流程：

```text
for each selected warm-up state:
    verify split + frozen seed
    verify same agent
    planner.reset_episode()
    planner.plan(state)
    assert warm_start_used == False
    UGUncertainty.compute(..., update_stats=True)

snapshot obs_mean / obs_std / horizon_std
planner.reset_episode()
```

最终 warm-up CEM solution 不会泄漏到正式第一个 decision；normalizer stats 保留。

## 6. Main mode vs sensitivity mode

### Main / source-faithful

```text
freeze_after_warmup = False
```

warm-up 后：

```text
update_uncertainty_stats = True
```

因此后续 planner 继续在线 EMA，与 UG source 风格一致。

### Sensitivity

```text
freeze_after_warmup = True
```

warm-up 后：

```text
update_uncertainty_stats = False
```

已校准的 `obs_mean / obs_std / horizon_std` 保留，但正式 planning 不再更新。

该模式只用于 Step 9 sensitivity，不替代主版本。

## 7. Warm-up report

`UGNormalizerWarmupReport` 记录：

- planner call count；
- train call count；
- calibration call count；
- unique episode seeds；
- agent name；
- 是否所有 call 都未使用 warm-start；
- normalizer 是否 finite；
- freeze policy；
- post-warm-up online-update policy；
- `obs_mean / obs_std / horizon_std` snapshot。

## 8. Tests

新增：

```text
tests/test_ug_normalizer_warmup.py
```

共 14 tests：

1. default=100 calls + online EMA；
2. validation/test split rejection；
3. seed/split mismatch 与 test-seed disguise rejection；
4. D=27 / finite / integer seed validation；
5. insufficient warm-up states rejection；
6. 每次 warm-up 强制 no previous solution；
7. train/calibration provenance count；
8. snapshot finite + D27/H4 shape；
9. main mode warm-up 后继续 EMA；
10. freeze sensitivity 关闭未来 EMA 但不删除 stats；
11. 只使用前 `planner_calls` 条记录；
12. 单 runner 拒绝 mixed-agent input；
13. 不同 planner normalizer 不隐式共享；
14. invalid config / item type guards。

## 9. Current boundary

当前 source 只实现 warm-up mechanism + leakage guards。

尚未宣称已经完成正式 100-call numerical calibration，因为正式 local run 仍需要：

- frozen v2 world model；
- shared reward predictor；
- real planner-visible train/calibration states；
- per-agent planner instance。

因此不能仅凭源码测试把 Step 6 标记 CLOSED。

## 10. Local unit-test Gate

在 `.venv_cc4`、`chapter2_region_detection`：

```bash
python -m unittest tests.test_ug_normalizer_warmup -v
```

预期：

```text
Ran 14 tests
OK
```

随后回归 Step 2–6：

```bash
python -m unittest \
  tests.test_categorical_cem \
  tests.test_ug_uncertainty \
  tests.test_shared_rollout_evaluator \
  tests.test_ug_cem_planner \
  tests.test_ug_normalizer_warmup -v
```

理论总数：

```text
Step 2  16
Step 3  12
Step 4  13
Step 5  14
Step 6  14
-------------
TOTAL   69
```

若 CUDA test 因本机无 CUDA 被允许 skip，最终 unittest 仍必须 `OK`。

## 11. Close sequence

Step 6 分两段关闭：

```text
6a source/unit tests
-> 14 + combined 69 tests PASS

6b real 100-call calibration
-> per-agent normalizer report
-> finite stats
-> no previous_solution
-> train/calibration only
-> no test leakage
```

只有 6a + 6b 都通过，Step 6 才正式 CLOSED，然后进入 Step 7 Integration Smoke Tests。

## 12. Current conclusion

```text
Taskbook alignment             : PASS
Source-faithful EMA policy     : READY
100-call default               : READY
No previous_solution warm-up   : READY
Frozen seed leakage guard      : READY
Per-agent/planner isolation    : READY
Main online-EMA policy         : READY
Freeze sensitivity policy      : READY
Unit-test source               : READY (14)
Initial local tests            : PASS (14/14; combined 69/69)
Runtime freeze source fix      : APPLIED / RERUN PENDING
Real 100-call calibration      : PENDING

FINAL STATUS: 6A FIX RERUN PENDING / 6B PENDING
```


## 13. Runtime-freeze integration correction

在进入真实 6b calibration 前进行 source audit 时发现：`UGCEMPlanner.plan()` 实际读取 runtime 字段 `_update_uncertainty_stats`，并通过 `set_uncertainty_update_mode()` 控制；Step 6 初版 warm-up 只替换 `planner.config.update_uncertainty_stats`，因此 `freeze_after_warmup=True` 的 config 表面值会变化，但 runtime compute 仍可能继续更新 EMA。

已修复：

```text
warm-up start -> planner.set_uncertainty_update_mode(True)
warm-up end   -> planner.set_uncertainty_update_mode(not freeze_after_warmup)
```

并强化现有 freeze sensitivity test：warm-up 后再次 `planner.plan()`，直接比较 `obs_mean / obs_std / horizon_std`，要求三者保持完全不变。

该问题不影响主版本 `freeze_after_warmup=False` 的 intended online-EMA 语义，但必须在 6b 前完成回归。由于测试代码已强化，不能沿用修复前的 14/69 PASS 作为最终 6a closure；需本地 rerun。

## 14. Formal 6b calibration pipeline

新增 state-only collector：

```text
formal_experiments/data_collection/collect_ug_calibration_states.py
```

正式固定：

```text
seeds       = 3000..3007
steps       = 100
pad_spaces  = False
state_dim   = 27
agents      = blue_agent_0..4
```

collector 复用 Gate A 已冻结的真实 CC4 trajectory/state encoder，但落盘时只投影以下字段：

```text
split
episode_seed
agent_name
decision_index
global_tick_start
state
```

输出文件明确不保存：

- incident host/event identity；
- response reward；
- official reward；
- hidden compromise truth；
- next_state / future labels。

并要求每个 Blue agent 至少有 100 个 calibration states，否则直接失败。

正式 calibration runner：

```text
formal_experiments/evaluation/calibrate_ug_normalizer.py
```

默认加载：

```text
outputs/ug_cem_v2/step6/calibration_states.jsonl
outputs/world_model_v2/a4_5b/world_model_absolute.pt
outputs/world_model_v2/a4_5c/response_reward_predictor.pt
```

每个 agent 独立建立：

```text
Categorical CEM
+ SharedRolloutEvaluator(shared frozen WM/reward)
+ independent UGUncertainty
+ UGCEMPlanner
```

calibration planner profile：

```text
H=4
A=4
M=5
N=64
I=4
elite_ratio=0.30
alpha=0.10
prob_floor=0.01
beta=0.10   # calibration/development profile, not Step-9 final beta
UG alpha=0.01
```

每 agent 从 8 个 calibration seeds 中 deterministic round-robin 选 100 个 states，确保主 run 覆盖全部 `3000..3007`，而不是让文件顺序导致 warm-up 偏向前几个 episode。

主版本输出：

```text
outputs/ug_cem_v2/step6/ug_normalizers_online.pt
outputs/ug_cem_v2/step6/ug_normalizers_online_report.json
```

bundle 保存 5 个 agent 各自的：

- obs_mean[27]；
- obs_std[27]；
- horizon_std[4]；
- calibration/planner profile metadata。

report 记录每 agent 100 calls、实际 seed coverage、finite、no-warm-start 与 online update policy。

## 15. Additional 6b tests

新增：

```text
tests/test_ug_normalizer_calibration.py
```

共 9 tests，覆盖：

1. state-only projection exact schema；
2. 非 calibration seed rejection；
3. all-agent >=100 + no hidden/reward label summary；
4. loader 拒绝额外 hidden 字段；
5. exact calibration schema loading；
6. deterministic balanced selection + all 8 seed coverage；
7. missing seed / insufficient state rejection；
8. formal N64/I4/A4/H4 calibration profile；
9. v2 default artifact binding + tensor finite summary。

因此 Step 6 当前测试总数：

```text
warm-up mechanism        14
formal calibration        9
---------------------------
Step 6 total             23
```

Step 2–6 combined total 变为：

```text
16 + 12 + 13 + 14 + 14 + 9 = 78
```

## 16. Updated local close sequence

由于 runtime-freeze fix 与 6b pipeline 都是在上一轮 14/69 PASS 之后新增，必须重新执行：

```bash
python -m unittest \
  tests.test_ug_normalizer_warmup \
  tests.test_ug_normalizer_calibration -v
```

目标：

```text
Ran 23 tests
OK
```

再执行：

```bash
python -m unittest \
  tests.test_categorical_cem \
  tests.test_ug_uncertainty \
  tests.test_shared_rollout_evaluator \
  tests.test_ug_cem_planner \
  tests.test_ug_normalizer_warmup \
  tests.test_ug_normalizer_calibration -v
```

目标：

```text
Ran 78 tests
OK
```

然后才进行真实 6b：

```bash
python -m formal_experiments.data_collection.collect_ug_calibration_states
python -m formal_experiments.evaluation.calibrate_ug_normalizer --device cpu
```

正式 main run 不加 `--freeze-after-warmup`；该 flag 只留给 Step 9 sensitivity。

6b 最终必须输出 `pass: True`，5 个 agent 均为 100 calls、seeds=3000..3007、finite=True、all_warm_starts_disabled=True、online_updates_after_warmup=True。