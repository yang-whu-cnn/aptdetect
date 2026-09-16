# Step A4.5b — Held-out World Model Validation 审核记录

日期：2026-09-16  
状态：PASS

## Git 范围

实现提交：

`61ad846fa39c015576a6ce5009c31d59b066a4ac`

相对 A4.5b contract 冻结点仅新增：

- `chapter2_region_detection/formal_experiments/evaluation/validate_bootstrap_world_model.py`
- `chapter2_region_detection/tests/test_gate_a_world_model_validation.py`

未修改 replay collector、FormalState、incident reward、bootstrap WM implementation、CEM、UG uncertainty 或 PPO。

## 源码审核

A4.5b evaluator 满足冻结要求：

- train / validation seed split 做严格检查；
- calibration / test seeds 不参与；
- absolute / delta 使用相同 architecture / seeds / epochs；
- one-step RMSE / MAE / mixture Gaussian NLL；
- persistence baseline；
- 27D per-feature diagnostics；
- per-executed-action diagnostics；
- per-agent diagnostics；
- H=2 / H=4 contiguous decision-epoch rollout；
- fixed member through horizon；
- terminal incomplete transition 不进入 rollout window；
- rollout 不跨 agent / episode / discontinuous state chain；
- H=4 ensemble disagreement 与实际 final-state error 做 Spearman；
- high-error AUROC + uncertainty quantiles；
- target-mode selection rule 在结果前冻结并在代码中实现。

测试文件包含 13 个 unit tests。

## Held-out validation result

正式 train replay：9041 transitions，其中 8952 completed。  
正式 validation replay：2336 transitions，其中 2315 completed。

### Absolute model

- one-step RMSE = 0.132581；
- persistence one-step RMSE = 0.180653；
- H=2 RMSE = 0.162506；
- H=2 persistence RMSE = 0.214534；
- H=4 RMSE = 0.191241；
- H=4 persistence RMSE = 0.218748；
- H=4 epistemic-error Spearman = 0.695511；
- 8 / 8 validation episodes 的 H=4 Spearman > 0；
- H=4 high-error AUROC = 0.774108。

### Delta model

- one-step RMSE = 0.128586；
- H=4 RMSE = 0.193774；
- H=4 epistemic-error Spearman = 0.567560；
- 8 / 8 validation episodes 的 H=4 Spearman > 0。

### Frozen selection rule

Delta 只有同时满足：

1. aggregate H=4 RMSE 比 absolute 至少低 2%；
2. 8 个 validation episodes 至少 6 个 episode 的 H=4 RMSE 更低；

才允许切换。

实际：

- delta relative H=4 improvement = -0.013247；
- delta H=4 episode wins = 1 / 8。

因此正式选择：

`target_mode = absolute`

## Quality Gate

Selected absolute model：

- one-step RMSE beats persistence：PASS；
- H=4 RMSE beats persistence：PASS；
- aggregate H=4 uncertainty-error Spearman > 0：PASS；
- positive per-episode H=4 Spearman count = 8 / 8：PASS。

FINAL STATUS: PASS

注意：WM 的 MAE 并非所有 horizon 都优于 persistence，因此论文不能表述为“所有误差指标全面优于 persistence”。正式 Gate 以事前冻结的 RMSE + H=4 uncertainty calibration 为准。
