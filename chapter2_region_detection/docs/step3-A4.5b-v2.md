# Step A4.5b v2 — World Model Rerun After A4.1c

日期：2026-09-16  
状态：PASS

## Data / contract

- train replay: A4.5a v2, 9041 transitions / 8952 completed samples；
- validation replay: A4.5a v2, 2336 transitions；
- D=27，M=5，hidden=128，epochs=50；
- target modes: absolute / delta；
- selector: held-out H=4 final-state RMSE；
- calibration/test seeds 未使用。

## Absolute

- one-step RMSE = 0.132231702 < persistence 0.180830250；
- H2 RMSE = 0.162538685；
- H4 RMSE = 0.191052066 < persistence 0.219133339；
- H4 epistemic-error Spearman = 0.687620634；
- H4 positive per-episode Spearman = 8 / 8；
- H4 high-error AUROC = 0.792691119。

## Delta

- one-step RMSE = 0.127634449 < persistence 0.180830250；
- H2 RMSE = 0.162598385；
- H4 RMSE = 0.192142777 < persistence 0.219133339；
- H4 epistemic-error Spearman = 0.551209414；
- H4 positive per-episode Spearman = 8 / 8。

## Selection

Frozen rule:

delta only if BOTH:
1. aggregate H4 RMSE improves by at least 2%；
2. delta wins H4 RMSE on at least 6 / 8 validation episodes。

Observed:

- delta relative H4 improvement = -0.005708973；
- delta episode H4 wins = 2 / 8；
- selected target mode = absolute。

## Gate

Selected absolute model:

- one-step beats persistence：PASS；
- H4 beats persistence：PASS；
- aggregate H4 uncertainty/error Spearman > 0：PASS；
- positive H4 per-episode Spearman = 8 / 8：PASS。

FINAL STATUS: PASS

Important diagnostic only: absolute one-step MAE (0.041888) is slightly worse than persistence MAE (0.040673), and H4 MAE (0.089110) is worse than persistence H4 MAE (0.063165). These were not frozen selection/Gate criteria, so no post-hoc retuning is performed.

Next: rerun A4.5c on v2 replay using the selected `world_model_absolute.pt` checkpoint from `outputs/world_model_v2/a4_5b/`.
