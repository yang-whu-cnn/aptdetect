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
