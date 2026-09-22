# Table 3 reward ablations

`Delay-Only`, `Fail-Only`, and `Full-Reward` share the same D27/A4 policy and
absolute five-member world model. The ablation is applied in both reward paths:

- PPO receives the selected reward's real per-tick bookkeeping value, accumulated
  once over the executed asynchronous action interval.
- imagined H=4 rollouts use the separately trained and frozen predictor for that
  same reward mode.

Delay-Only and Fail-Only predictors are derived only from the frozen train and
validation replays. They use the same single-MLP architecture, inputs,
normalizers, optimizer, learning rate, batch size, fixed 50-epoch policy, model
seed, and split as the existing Full-Reward artifact. Delay-Only retains MSE.
The approved minimal Fail-Only change uses unweighted SmoothL1/Huber loss with
`beta=1.0` on standardized reward labels. Class weighting and positive-example
resampling are forbidden. The original Fail-Only MSE result is diagnostic-only.
The existing Full-Reward pipeline does not perform early checkpoint selection;
the ablations match it rather than introducing validation-driven checkpoint
selection. Test episodes are not used for training, stopping, or gate tuning.

Training command (repeat for `Fail-Only`):

```powershell
.venv_cc4\Scripts\python.exe -m formal_experiments.ours.reward_ablation `
  --mode Delay-Only `
  --out-dir outputs/formal_v3/table3_reward_models/delay_only
```

The frozen manifest includes H=1 and oracle-state H=4 RMSE/rank metrics,
constant and persistence baselines, the checkpoint SHA, and a fail-closed gate.
The real CC4 smoke refuses to load an ablation unless this gate passes and the
SHA matches. A development smoke is not a performance result.

Fail-Only is strongly zero-inflated. A two-part hurdle predictor is retained in
code for train-only diagnostics, but it is forbidden from the formal artifact.
Weighted loss, resampling, alternate seeds selected by validation, and a relaxed
gate are likewise forbidden. If the approved locked single predictor fails the
unchanged gate, Fail-Only remains `BLOCKED`.
