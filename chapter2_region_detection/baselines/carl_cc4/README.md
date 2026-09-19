# CARL-CC4 (adapted)

This is a disclosed CAICS/Dyna/SimPLe-style adaptation. A D27-to-A4 PPO policy
collects real trajectories. During training only, a transition model and causal
reward SCM generate labelled synthetic trajectories which are mixed with real
experience at a fixed ratio. There is no decision-time tree search, MCTS, CEM,
or model rollout in `policy.act`.

The fastest transition route reuses the frozen five-member final-paper world
model; this is disclosed reuse, not a claim of a CARL-native transition model.
The causal DAG contains observable evidence, action, target role, pre/post
incident proxies, service failure/availability, and reward. Hidden incident
truth may create offline SCM/reward labels but is never a policy input.

The standard CAICS mapping freezes `alpha_comp=.025`, `beta_comp=2`, isolate
state/change terms to zero, and A4 costs to no-op/analyse=0 and remove/restore=1.
The +1000/-1000 terminal terms apply only to explicitly labelled attack-cleared
or severe-failure termination, never an ordinary 500-tick end.

The paper's eight synthetic rollouts per real rollout are mandatory. Requested
H=256 imagination is `BLOCKED` while the reused world model is validated only
to H=4. An explicitly selected H=4 truncation is reported as
`ADAPTED_TRUNCATED` and is not silently formal-eligible.

`formal_training.py` is the minimum formal runner. It is CPU-only and requires
the real CC4 collector, frozen shared-WM generator, and per-seed SCM fitter to be
injected; it never replaces a missing dependency with synthetic smoke data.
Training is fixed to seeds `1000..1031`, 500 ticks, policy seeds
`51001..51005`, and uses `2000..2007` only for validation selection. Test seeds
`4000..4099` are rejected from selection/training provenance. Each completed
episode is atomically checkpointed, so `resume=True` continues only an exact
seed prefix with matching WM and validation-selection hashes. Final files use
the evaluator's `policy_<seed>.pt` schema (`method=carl_cc4`, `policy_seed`,
`formal_training_complete`, `training_split=train`, and
`training_episode_seeds`).

Formal entry also requires a clean Git worktree. Both progress and final
checkpoints bind `code_commit`, `git_dirty=false`, and `git_diff_sha256` along
with the exact validation-selection file hash and frozen-WM hash. Resume fails
if any of those identities differ.

The runner defaults to fail-closed H=256 validation. An explicit disclosed
H=4 adaptation is accepted only when its gate status is `ADAPTED_TRUNCATED`;
the status, effective horizon, and disclosure are persisted in the checkpoint.
