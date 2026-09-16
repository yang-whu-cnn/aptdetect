# Step A4.6a v2 — Corrective Runtime Result

日期：2026-09-16  
状态：RUNTIME PASS / SOURCE AUDIT PENDING

## Contract

- state_dim = 27；
- n_actions = 4；
- any_valid_observable_target index = 17；
- target threshold = 0.5；
- planning horizon H = 4；
- gamma_tick = 0.99；
- canonical rule：no_op→Sleep；targeted + no valid observable target→Sleep；otherwise keep requested；
- hidden inputs = false；
- calibration/test seeds 未使用。

## True-state requested→executed mapping

Train：

- count = 9041；
- overall accuracy = 1.0；
- targeted count = 6804；
- targeted accuracy = 1.0；
- feature=1 but fallback = 0；
- feature=0 but non-fallback = 0；
- nonbinary feature count = 0。

Validation：

- count = 2336；
- overall accuracy = 1.0；
- targeted count = 1759；
- targeted accuracy = 1.0；
- feature=1 but fallback = 0；
- feature=0 but non-fallback = 0；
- nonbinary feature count = 0。

## Integrated requested-plan H=4

- final-state RMSE = 0.200015 < persistence 0.219133；
- response-value RMSE = 7.608135 < constant baseline 15.522030；
- Pearson = 0.871661；
- Spearman = 0.681405；
- positive episode value Spearman = 8 / 8；
- return-uncertainty / true-error Spearman = 0.621106；
- targeted member-step match rate = 0.917540。

The targeted member-step match rate is diagnostic only: future rollout states are WM predictions, so canonical availability can be mispredicted. The 100% hard requirement applies to mapping from real replay states to real executed actions, which now passes exactly.

## Gate

All frozen A4.6a hard-Gate conditions passed.

RUNTIME STATUS: PASS

Source audit remains pending because the A4.6a evaluator and its unit test have not yet been pushed to the repository.
