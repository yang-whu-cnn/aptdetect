# Step A4.5c v2 — Response Reward Predictor Rerun After A4.1c

日期：2026-09-16  
状态：PASS

## Contract

- state_dim = 27；
- n_actions = 4；
- reward-model input = state + executed/canonical action + next_state；
- hidden reward inputs = false；
- gamma_tick = 0.99；
- horizon = 4；
- selected world-model target mode = absolute；
- calibration/test seeds 未使用。

## Training / one-step

- train samples = 8952；
- train reward mean = -2.526139；
- train reward std = 4.894484；
- one-step RMSE = 2.258064；
- train-mean baseline RMSE = 4.854820；
- action-mean baseline RMSE = 3.709165；
- Pearson = 0.885308；
- Spearman = 0.716369。

## H=4 planning value

Oracle-state reward-model H4：

- RMSE = 6.944553；
- constant baseline RMSE = 15.522030；
- Pearson = 0.893095；
- Spearman = 0.746864。

Selected absolute-WM + reward predictor H4：

- RMSE = 7.626011；
- constant baseline RMSE = 15.522030；
- Pearson = 0.870926；
- Spearman = 0.670119；
- return-uncertainty / true-error Spearman = 0.630400；
- positive episode Spearman = 8 / 8。

## Gate

- one-step beats train-mean baseline：PASS；
- WM H4 beats constant-return baseline：PASS；
- aggregate WM H4 Spearman > 0.3：PASS；
- at least 5 / 8 positive per-episode Spearman：PASS（8 / 8）。

FINAL STATUS: PASS

Response reward remains the frozen two-penalty scalar; no positive reward bonus is introduced.

Next: rerun A4.6a on v2 replay + v2 absolute WM + v2 reward predictor, with the original 100% requested→executed mapping hard Gate unchanged.
