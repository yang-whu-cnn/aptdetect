# Gate B0 — Frozen World Model Final Audit

日期：2026-09-17  
状态：**B0.1 SOURCE READY / LOCAL TEST + REAL AUDIT PENDING；B0.2 WAITING FOR PROVISIONAL PPO**

---

## 1. Purpose

Gate A 已证明当前 bootstrap probabilistic WM 在 aggregate validation 上达到 planning-ready 条件，但 aggregate 指标可能掩盖 action-specific weakness。Gate B0 不默认重训 WM，而是在 PPO 前后增加两次审计：

```text
B0.1: frozen replay 上的 per-action / H4 final audit
B0.2: provisional PPO 后的 policy-induced OOD / model-exploitation audit
```

只有 audit FAIL 才重新打开 shared WM training。

## 2. Frozen inputs

B0.1 只能读取：

```text
outputs/formal_replay_v2/train.jsonl
outputs/formal_replay_v2/validation.jsonl
outputs/world_model_v2/a4_5b/world_model_absolute.pt
```

不训练参数、不重新 fit normalizer、不读取 calibration/test。

Frozen WM contract：

```text
D=27
A=4
M=5
target_mode=absolute
H=4 audit
```

## 3. Per-action audit

对 `no_op/analyse/remove/restore`：

- completed sample count；
- one-step RMSE / MAE；
- action-specific persistence RMSE / MAE；
- RMSE/persistence ratio；
- sample epistemic uncertainty vs error Spearman；
- uncertainty quartiles；
- first-action-conditioned H4 metrics；
- per-agent action diagnostics。

H4 window 必须复用 A4.5 的严格 chained decision-epoch definition：同 episode seed、同 agent、连续 decision index、tick continuity、state continuity、all actions completed。

## 4. Aggregate reproduction

重新计算并与 A4.5b frozen values 比较：

```text
one_step_rmse                       0.132231702
one_step_persistence_rmse           0.180830250
h4_rmse                             0.191052066
h4_persistence_rmse                 0.219133339
h4_uncertainty_error_spearman       0.687620634
```

absolute tolerance：`5e-4`。

## 5. Acceptance rules

硬条件：

```text
validation count/action >= 150
one-step RMSE/persistence <= 1.25 for every action
>=3/4 actions beat persistence
aggregate reproduction abs delta <=5e-4
all finite
no action-ID/family mismatch
D27/A4/M5/absolute checkpoint
```

H4 safety：

```text
first-action H4 bucket missing
OR nonfinite ratio
OR RMSE/persistence > 2.0
```

会设置：

```text
manual_review_required = True
pass = False
```

因此 catastrophic H4 bucket 不会被 aggregate one-step Gate 静默掩盖。

## 6. Failure branch

实现/路径/数值 bug：修 bug 后重跑，不改变阈值。

真实质量 FAIL：

```text
train-only targeted recollection
-> new replay version
-> retrain shared WM
-> reward distribution changed 时同步 retrain reward predictor
-> rerun A4.5/A4.6
-> rerun B0.1
```

任何新 WM 必须同时给 LWM-RL / UG-CEM / CEM-APT 使用。

## 7. Source

```text
formal_experiments/evaluation/audit_world_model_final.py
```

该实现复用：

```text
validate_bootstrap_world_model.build_rollout_windows
validate_frozen_split
shared action contract
BootstrapProbabilisticWorldModel checkpoint loader
```

没有复制一套不同的 rollout 语义。

## 8. Tests

```text
tests/test_gate_b0_world_model_audit.py
```

当前 9 个 unit tests 锁定：

1. frozen thresholds/action names；
2. happy-path Gate；
3. insufficient per-action count FAIL；
4. action ratio >1.25 FAIL；
5. fewer than 3 actions beat persistence FAIL；
6. frozen aggregate reproduction tolerance FAIL；
7. catastrophic H4 ratio requires review/FAIL；
8. missing H4 bucket requires review；
9. incomplete transitions excluded from action counts。

## 9. Local execution

在：

```bash
cd /d/paper/github-me/aptdetect/chapter2_region_detection
source .venv_cc4/Scripts/activate
```

先跑：

```bash
python -m unittest tests.test_gate_b0_world_model_audit -v
```

预期：

```text
Ran 9 tests
OK
```

然后运行真实 audit：

```bash
python -m formal_experiments.evaluation.audit_world_model_final --device cpu
```

关键 summary：

```text
[GATE B0.1 WM FINAL AUDIT]
train_counts: ...
validation_counts: ...
no_op/analyse/remove/restore per-action metrics
aggregate_reproduction: ...
actions_beating_persistence: ...
h4_manual_review_required: False
pass: True
```

正式报告：

```text
outputs/world_model_v2/b0_1/per_action_audit.json
```

## 10. Close condition

只有：

```text
9/9 tests PASS
+
real audit pass=True
+
h4_manual_review_required=False
+
aggregate reproduction PASS
```

之后 B0.1 才 CLOSED，并进入 B2 registry / validation state bank。

B0.2 在 provisional PPO 后再执行，此时 B0.1 不替代 B0.2。
