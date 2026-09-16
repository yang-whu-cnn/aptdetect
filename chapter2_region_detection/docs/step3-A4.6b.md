# Step A4.6b — Formal v2.1 Config Freeze + Legacy Isolation

日期：2026-09-16  
状态：**PASS / CLOSED**

---

## 1. 目标

A4.6b 只冻结 Gate A 已经确认的共享 formal contract，不提前修改 Step 4/5/9 尚未完成的 planner-specific tuning。

正式比较对象：

- LWM-RL；
- UG-CEM-APT；
- CEM-APT。

三者后续必须共享同一套 state/action/target/action-adapter/world-model/reward/seeds/decision-epoch contract。

## 2. 新正式配置

新增：

```text
configs/compare_ug_cem_formal_v2_1.yaml
```

冻结：

- FormalStateEncoder D=27；
- availability feature=`any_valid_observable_target`，index=17；
- categorical A=4：no_op / analyse / remove / restore；
- CC4 mapping：Sleep / Analyse / Remove / Restore；
- duration ticks：1 / 2 / 3 / 5；
- H=4；
- gamma_tick=0.99；
- decision-epoch semantics；
- execute plan[0] then replan；
- v2 train/validation replay；
- selected absolute bootstrap WM；
- shared response reward predictor；
- M=5，hidden=128；
- fixed member through horizon；
- deterministic mean rollout；
- no aleatoric resampling；
- train/validation/calibration/test exact seed protocol；
- shared fairness contract；
- calibration/test update forbidden。

formal artifacts 固定为：

```text
outputs/formal_replay_v2/train.jsonl
outputs/formal_replay_v2/validation.jsonl
outputs/world_model_v2/a4_5b/world_model_absolute.pt
outputs/world_model_v2/a4_5c/response_reward_predictor.pt
outputs/world_model_v2/a4_6a/action_consistency_report.json
```

## 3. Legacy isolation

历史开发配置继续保留：

```text
configs/compare_ug_cem_local_online.yaml
```

但文件头已明确标记：

```text
LEGACY / DEVELOPMENT ONLY — DO NOT USE FOR FORMAL v2.1 EXPERIMENTS
```

旧 A=5 / D=8 / cost-delay 配置只用于 provenance / historical development，不允许进入 formal v2.1 run。

正式 v2.1 YAML 不包含：

- control_traffic；
- A=5；
- D=8；
- old fixed action cost；
- old delay penalty；
- old cost/delay plan score。

## 4. Regression tests

新增：

```text
tests/test_gate_a_formal_comparison_config.py
```

当前包含 9 个 tests：

1. formal v2.1 identity/status；
2. D=27 state contract；
3. A=4 action contract 与 source code exact match；
4. formal artifacts 全部绑定 v2；
5. WM + reward + H4/gamma contract；
6. exact seed protocol；
7. fairness shared-interface contract；
8. formal YAML 无 legacy objective/action/dimension key；
9. legacy config 被明确标记且 formal run 禁止引用。

## 5. 本地 Gate

在 `.venv_cc4`、`chapter2_region_detection` 目录执行：

```bash
python -m unittest tests.test_gate_a_formal_comparison_config -v
```

关闭条件：

```text
Ran 9 tests
OK
```

本地 `.venv_cc4` 已执行该测试文件，9/9 tests 全部通过。

## 6. 当前结论

```text
Formal v2.1 config source : PASS
Legacy config isolation   : PASS
Regression test source    : PASS (9 tests)
Local test execution      : PASS (9/9)

FINAL STATUS: PASS / CLOSED
```

A4.6b 已正式关闭。下一步进入 A4.6c A4 final integration review。
