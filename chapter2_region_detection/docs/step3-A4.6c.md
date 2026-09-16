# Step A4.6c — A4 Final Integration Review

日期：2026-09-16  
状态：**SOURCE READY / LOCAL REGRESSION PENDING**

---

## 1. 目标

A4.6c 不重新设计 A4.1–A4.6b，而是确认正式模型空间链路已经形成一个可被 Step 4 直接复用的共享接口：

```text
FormalState(27D)
-> planner requested action (A=4)
-> shared model-space canonicalizer
-> canonical/executed action
-> bootstrap world model
-> response reward predictor
-> H=4 duration-aware planning value
```

并检查：

- Ours / UG-CEM / CEM 后续共享同一 domain/model interface；
- planner-visible 层不导入 hidden incident bookkeeping；
- formal artifact 只绑定 v2 train/validation 产物；
- calibration/test 不参与 Gate A fitting/update；
- legacy A=5 / D=8 / cost-delay contract 不进入 formal path。

## 2. Integration review 中发现并修复的两个问题

### 2.1 Canonicalizer layering

A4.6a 原先把 requested→canonical action 函数定义在：

```text
formal_experiments/evaluation/audit_model_space_action_consistency.py
```

这会导致未来 Step 4 planner 依赖 evaluation/audit 层。

现已抽取为正式 shared interface：

```text
shared/model_space_action.py
```

导出：

- `TARGET_THRESHOLD=0.5`；
- `ANY_TARGET_INDEX=17`；
- `canonicalize_requested_action`；
- `canonicalize_requested_tensor`。

A4.6a evaluator 现在直接复用该 shared implementation，不再维护第二份 canonicalization 逻辑。

### 2.2 Response objective self-contained freeze

A4.3 已冻结 response reward：

```text
-lambda_time * incident_active_ticks
+ lambda_failure * raw incident-host LWF penalty
```

源码默认：

```text
lambda_time = 1.0
lambda_failure = 1.0
```

A4.6b 初版 formal YAML 只冻结 reward predictor checkpoint，未显式记录这两个 objective 权重。

现已补入：

```yaml
response_objective:
  lambda_time: 1.0
  lambda_failure: 1.0
  attack_eradication_time: true
  incident_host_lwf_only: true
  official_team_reward_separate: true
```

因此 formal config 现在可以独立描述 shared response objective。

## 3. Formal config integration additions

`configs/compare_ug_cem_formal_v2_1.yaml` 进一步冻结：

```text
model_space_canonicalizer = shared.model_space_action.canonicalize_requested_action
a4_6a_closed = true
a4_6b_closed = true
```

这些字段不改变已通过的 A4.6a/A4.6b 研究语义，只消除后续 Step 4 的实现歧义。

## 4. A4.6c regression tests

新增：

```text
tests/test_gate_a_final_integration.py
```

共 8 个 tests：

1. D=27 / A=4 / duration 与 action audit 单一契约；
2. A4.6a evaluator 与 shared canonicalizer 使用同一个函数对象；
3. requested→canonical scalar/tensor 行为；
4. BootstrapWorldModelConfig 与 formal YAML 一致；
5. ResponseRewardPredictorConfig / ResponseRewardConfig 与 formal YAML 一致；
6. H=4 / gamma_tick / execute-first / replan / duration-aware semantics；
7. formal artifacts 仅使用 v2 且不绑定 calibration/test；
8. planner-visible shared modules 不导入 hidden incident bookkeeping。

同时由于 A4.6c 修改了 shared canonicalizer 与 formal config，必须回归：

- `tests.test_gate_a_model_space_action_consistency`：9 tests；
- `tests.test_gate_a_formal_comparison_config`：9 tests；
- `tests.test_gate_a_final_integration`：8 tests。

合计 targeted regression = **26 tests**。

## 5. 本地关闭 Gate

在 `.venv_cc4`、`chapter2_region_detection` 目录：

```bash
python -m unittest \
  tests.test_gate_a_model_space_action_consistency \
  tests.test_gate_a_formal_comparison_config \
  tests.test_gate_a_final_integration -v
```

预期：

```text
Ran 26 tests
OK
```

然后再次执行：

```bash
python -m formal_experiments.evaluation.audit_model_space_action_consistency --device cpu
```

numerical result 必须继续满足：

- train/validation mapping accuracy = 1.0；
- feature=1 fallback = 0；
- feature=0 nonfallback = 0；
- H4 state RMSE < persistence；
- H4 value RMSE < constant baseline；
- H4 value Spearman > 0.3；
- positive episode Spearman >= 5/8；
- `quality_gate.pass=True`。

因为 A4.6c 只是 shared-layer refactor，正常情况下数值应与 A4.6a v2 rerun 基本完全一致。

## 6. 当前结论

```text
Shared canonicalizer       : READY
Formal reward freeze       : READY
Formal v2.1 integration    : READY
A4.6a regression source    : READY (9)
A4.6b regression source    : READY (9)
A4.6c integration tests    : READY (8)
Local targeted regression  : PENDING
Local numerical regression : PENDING

FINAL STATUS: SOURCE READY / LOCAL REGRESSION PENDING
```

本地 regression + numerical audit 通过后，A4.6c 才能 CLOSED；随后生成 A4 aggregate closure 文档并进入 A5 Gate A Final Review。
