# Step A4.5c — Response Reward Predictor / Planning-Value Readiness 审核记录

日期：2026-09-16  
状态：PASS

## Git 范围

实现提交：

`9f70457c7e1ea61e6b07b3aa99bb9e8eecb1e831`

相对 A4.5c contract 冻结点仅新增：

- `chapter2_region_detection/formal_experiments/training/response_reward_predictor.py`
- `chapter2_region_detection/formal_experiments/evaluation/validate_response_reward_predictor.py`
- `chapter2_region_detection/tests/test_gate_a_response_reward_predictor.py`

未修改 FormalState、CC4 adapter、decision replay、incident reward、bootstrap WM、CEM、UG uncertainty 或 PPO。

## 源码审核

A4.5c 满足冻结要求：

- reward predictor 输入仅为 planner/model-space variables：state + executed/canonical action + next_state；
- hidden incident_active_ticks / incident_event_id / incident_host_id / LWF breakdown 不作为 predictor input；
- target 仍是 A4.3 frozen response_reward；
- incomplete terminal transition 默认排除；
- completed transition 校验 decision_dt == executed_duration；
- dataset 学 executed action，fallback targeted -> Sleep 作为 action 0；
- 2-layer MLP hidden=128，ReLU，Adam lr=3e-4，batch=256，epochs=50；
- train-only state normalizer + train-only scalar reward normalizer；
- checkpoint 保存 config、state/reward normalizer 与 model weights；
- H=4 validation 使用 selected absolute WM；
- fixed WM member through whole horizon；
- reward accumulation 使用 gamma_tick=0.99 和 cumulative elapsed global ticks；
- train / validation split 严格检查，calibration/test 未使用；
- 14 个 unit tests。

## Held-out validation result

Training samples = 8952。

One-step response reward：

- RMSE = 2.249156；
- train-mean baseline RMSE = 4.854820；
- action-mean baseline RMSE = 3.709165；
- Pearson = 0.886240；
- Spearman = 0.696551。

Oracle-state H=4 value：

- RMSE = 6.911375；
- constant baseline RMSE = 15.522030；
- Pearson = 0.894129；
- Spearman = 0.729760。

Selected Absolute-WM + reward predictor H=4 value：

- RMSE = 7.614861；
- constant baseline RMSE = 15.522030；
- Pearson = 0.871133；
- Spearman = 0.675280；
- return-uncertainty / true-error Spearman = 0.608789；
- positive per-episode value Spearman = 8 / 8。

## Gate

- one-step beats train-mean baseline：PASS；
- WM H=4 beats constant-return baseline：PASS；
- aggregate WM H=4 Spearman > 0.3：PASS；
- at least 5 / 8 positive per-episode Spearman：PASS（8 / 8）。

FINAL STATUS: PASS

注意：response_reward 仍只有 attack-eradication-time penalty + current-incident-host normal-operation-failure penalty 两项；reward predictor 不引入任何额外正奖励项。
