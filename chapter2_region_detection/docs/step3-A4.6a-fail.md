# Step A4.6a — Model-Space Requested→Executed Consistency Audit

日期：2026-09-16  
状态：FAIL / CORRECTIVE STATE REVISION REQUIRED

## Runtime result

Train：

- mapping accuracy = 0.992368；
- targeted accuracy = 0.989859；
- feature=1 but fallback = 69；
- feature=0 but non-fallback = 0。

Validation：

- mapping accuracy = 0.990582；
- targeted accuracy = 0.987493；
- feature=1 but fallback = 22；
- feature=0 but non-fallback = 0。

Integrated requested-plan H=4：

- state RMSE = 0.199569 < persistence 0.218748；
- value RMSE = 7.613871 < constant baseline 15.522030；
- value Spearman = 0.686128；
- 8/8 validation episodes value Spearman > 0；
- targeted member-step canonicalization match rate = 0.914351。

## Failure cause

Current FormalState feature 17 `any_observable_target` only indicates that `ObservableHostEvidenceTracker.observable_host_scores()` is non-empty.

The production CC4 resolver requires a stricter condition:

`observable scored host ∩ currently valid wrapper host-actions != empty`

Therefore evidence may exist for a host that is not currently a valid Blue host-action target. In that case feature 17 is 1 while the real adapter correctly falls back to Sleep.

Observed failures are one-sided:

- feature=0 -> non-fallback: 0 cases；
- feature=1 -> fallback: 69 train / 22 validation cases。

This is a representation insufficiency, not a planner-quality failure.

## Corrective decision

Do NOT relax the exact Gate and do NOT add a hidden-truth heuristic.

Revise FormalState feature 17 in-place, keeping D=27:

`any_observable_target`
→
`any_valid_observable_target`

New semantics:

- input source remains planner-visible only；
- compute from current Blue wrapper labels/mask plus observable host scores；
- true iff at least one observable scored host is also a currently valid targeted CC4 action host；
- no controller true state / Red session / incident truth。

Because Analyse / Remove / Restore use the same host-validity/session rule in the frozen CC4 BlueFixedActionWrapper and A3 has already verified all three families are present for all five Blue agents, one shared availability bit is sufficient for the frozen environment.

This changes the semantic meaning of state dimension 17, so existing A4.5 replay/checkpoints are superseded for formal use. A4.5a replay must be recollected, then A4.5b/A4.5c rerun. Model/reward architectures do not need redesign.

FINAL STATUS: FAIL AS DESIGNED
