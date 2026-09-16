# Step A4 — Formal State / Replay / World Model / Reward / Integration Aggregate Closure

日期：2026-09-16  
状态：**PASS / CLOSED**

---

## 1. Scope

A4 将 CC4 当前可见 observation 统一映射为正式 model/planner space，并冻结后续 LWM-RL、UG-CEM-APT、CEM-APT 共用的动态模型与 response-value 接口。

最终链路：

```text
current visible CC4 observation
-> ObservableHostEvidenceTracker
-> FormalStateEncoder (D=27)
-> requested high-level action (A=4)
-> shared model-space canonicalizer
-> canonical/executed action
-> bootstrap probabilistic ensemble world model
-> shared response reward predictor
-> H=4 duration-aware planning value
```

## 2. A4.1 Formal state

- D=27；
- five Blue agents supported；
- feature 17=`any_valid_observable_target`；
- only planner-visible observation/evidence enters state；
- hidden Red sessions / compromise truth / future labels / reward truth excluded；
- A4.1c corrected observable-evidence availability into truly executable observable-target availability without changing D。

## 3. A4.2 Decision-epoch replay

- per-agent asynchronous decision epoch；
- requested + executed action both recorded；
- actual executed duration recorded；
- fallback reason recorded；
- busy agents omitted rather than filled with Sleep；
- terminal incomplete transitions preserved for audit but excluded from formal dynamics/reward fitting when required。

## 4. A4.3 Response objective

Frozen objective：

```text
response_reward
= -1.0 * incident_active_ticks
  + 1.0 * raw incident-host LWF penalty
```

- attack eradication time uses compromise→normal duration semantics；
- only current active incident host GreenLocalWork failure enters incident-specific objective；
- official CC4 team reward stays separate；
- hidden truth restricted to reward/evaluation bookkeeping。

## 5. A4.4 Bootstrap probabilistic world model

- state_dim=27；
- n_actions=4；
- ensemble M=5；
- two-layer MLP hidden=128；
- diagonal Gaussian；
- independent member initialization / bootstrap sampling / optimizer；
- train-only state normalizer；
- dynamics learns executed/canonical action；
- no decision_dt model input；
- exact selected checkpoint shared by all formal planners。

## 6. A4.5 Validation and planning-value readiness

Corrective v2 replay：

- train transitions=9041；
- validation transitions=2336；
- requested→executed mapping exact after A4.1c；
- all non-feature17 replay content unchanged from previous collection。

Selected WM：absolute。

Key v2 WM validation：

```text
one-step RMSE = 0.132231702 < persistence 0.180830250
H4 RMSE       = 0.191052066 < persistence 0.219133339
H4 uncertainty-error Spearman = 0.687620634
positive per-episode Spearman = 8/8
```

Shared response-reward predictor：

```text
one-step RMSE = 2.258064 < train-mean baseline 4.854820
oracle H4 RMSE = 6.944553
selected WM + reward H4 RMSE = 7.626011 < 15.522030
H4 Spearman = 0.670119
positive episodes = 8/8
```

## 7. A4.6 Integration closure

### A4.6a

- requested→canonical executed action exact on real replay states；
- formal default artifacts locked to `_v2` paths；
- 9/9 regression tests PASS；
- numerical quality gate PASS。

Final numerical rerun：

```text
train mapping accuracy = 1.0
validation mapping accuracy = 1.0
H4 state RMSE = 0.2000148377762988 < 0.21913333903939589
H4 value RMSE = 7.608134616100138 < 15.522029956815578
H4 value Spearman = 0.6814048261786793
positive episode Spearman = 8/8
quality_gate.pass = True
```

### A4.6b

- `configs/compare_ug_cem_formal_v2_1.yaml` created；
- D=27 / A=4 / H=4 / gamma_tick=0.99 frozen；
- v2 replay / WM / reward artifacts frozen；
- exact seed protocol frozen；
- old `compare_ug_cem_local_online.yaml` marked LEGACY / DEVELOPMENT ONLY；
- 9/9 config regression tests PASS。

### A4.6c

- canonicalizer promoted to `shared/model_space_action.py`；
- evaluator reuses same shared function object；
- formal response weights explicitly frozen in YAML；
- 8 final-integration tests added；
- targeted A4.6a+A4.6b+A4.6c regression 26/26 PASS；
- post-refactor numerical audit reproduced exact A4.6a values。

## 8. Formal seed protocol

```text
train       = 1000..1031 (32)
validation  = 2000..2007 (8)
calibration = 3000..3007 (8)
test        = 4000..4019 (20)
```

A4 fitting/design used train + validation only；calibration/test remained outside Gate A model fitting。

## 9. Frozen artifacts and shared interface

```text
configs/compare_ug_cem_formal_v2_1.yaml
outputs/formal_replay_v2/train.jsonl
outputs/formal_replay_v2/validation.jsonl
outputs/world_model_v2/a4_5b/world_model_absolute.pt
outputs/world_model_v2/a4_5c/response_reward_predictor.pt
outputs/world_model_v2/a4_6a/action_consistency_report.json
shared/model_space_action.py
```

Local output artifacts are intentionally not committed to Git。

## 10. Conclusion

```text
A4.1 formal state                  : PASS
A4.2 decision replay              : PASS
A4.3 response objective           : PASS
A4.4 bootstrap world model        : PASS
A4.5 validation/value readiness   : PASS
A4.6a action consistency          : PASS
A4.6b formal config isolation     : PASS
A4.6c final integration           : PASS

FINAL STATUS: A4 PASS / CLOSED
```

Next: A5 Gate A Final Review. A5 does not introduce a new model design; it performs complete Gate A regression/freeze verification before Step 4.
