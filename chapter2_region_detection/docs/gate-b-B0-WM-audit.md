# Gate B0 — Frozen World Model Final Audit

日期：2026-09-17  
状态：**B0.1 PASS / CLOSED；B0.2 WAITING FOR PROVISIONAL PPO**

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

9 个 unit tests 锁定：

1. frozen thresholds/action names；
2. happy-path Gate；
3. insufficient per-action count FAIL；
4. action ratio >1.25 FAIL；
5. fewer than 3 actions beat persistence FAIL；
6. frozen aggregate reproduction tolerance FAIL；
7. catastrophic H4 ratio requires review/FAIL；
8. missing H4 bucket requires review；
9. incomplete transitions excluded from action counts。

本地执行结果：

```text
Ran 9 tests in 0.001s
OK
```

## 9. Real B0.1 result

正式运行：

```bash
python -m formal_experiments.evaluation.audit_world_model_final --device cpu
```

completed coverage：

```text
train_counts:
  no_op   6114
  analyse  916
  remove   957
  restore  965

validation_counts:
  no_op   1641
  analyse  212
  remove   230
  restore  232
```

one-step / H4 action-conditioned 结果：

```text
no_op:
  one-step RMSE       0.1230845151
  persistence         0.1475714589
  ratio               0.8340672102
  uncertainty/error ρ 0.4706396871
  H4 windows          1592
  H4 ratio            0.8595271162

analyse:
  one-step RMSE       0.1342953655
  persistence         0.3119809499
  ratio               0.4304601468
  uncertainty/error ρ 0.0002947129
  H4 windows          184
  H4 ratio            1.1682244783

remove:
  one-step RMSE       0.1136328298
  persistence         0.1479138938
  ratio               0.7682363492
  uncertainty/error ρ 0.1490678298
  H4 windows          201
  H4 ratio            0.8259541276

restore:
  one-step RMSE       0.1950259091
  persistence         0.2482390671
  ratio               0.7856374556
  uncertainty/error ρ 0.3841989740
  H4 windows          218
  H4 ratio            0.7930127734
```

aggregate reproduction：

```text
one_step_rmse                      0.13223170196479042
one_step_persistence_rmse          0.18083025022779947
h4_rmse                            0.1910520662354734
h4_persistence_rmse                0.21913333903939589
h4_uncertainty_error_spearman      0.6876206337584564
```

Gate：

```text
actions_beating_persistence = 4 / 4
h4_manual_review_required   = False
pass                        = True
```

正式报告：

```text
outputs/world_model_v2/b0_1/per_action_audit.json
```

## 10. Interpretation

B0.1 证明 aggregate 指标没有掩盖某个动作的一步预测崩溃：四动作均优于各自 persistence baseline，且四个 H4 first-action bucket 均低于 catastrophic threshold 2.0。

`analyse` 的 one-step uncertainty-error Spearman 接近 0，且 H4 ratio=1.168 >1，说明该动作的局部 uncertainty calibration / multi-step quality 仍是后续 B0.2 OOD audit 的重点观察项；但它不违反预先冻结的 B0.1 hard gate，也没有达到 catastrophic review 条件。

因此当前结论：

```text
CURRENT WM MODIFICATION / RETRAIN: NOT REQUIRED
CURRENT WM STATUS: KEEP FROZEN
B0.1: PASS / CLOSED
B0.2: REQUIRED AFTER PROVISIONAL PPO
```

B0.1 PASS 不能替代 B0.2。若 provisional PPO 将 policy distribution 推入 WM replay 未覆盖区域，仍必须按 B0.2 规则重新评估并在必要时重开 WM training。
