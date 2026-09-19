# TERLA-A4

This is a structural CC4/A4 adaptation of TERLA. The paper does not provide an
official implementation. It uses the paper's network-agnostic heterogeneous
graph and single-shared-policy design, while deleting the fifth Deploy Decoy
action to comply with the frozen A4 protocol.

## Preserved

- Mission, subnet and host node types with undirected mission–subnet and
  subnet–host relations.
- Two HGT-style typed-attention layers, ReLU after each layer, and global sum
  pooling of host embeddings.
- Two observable host features: malicious process and malicious network event.
- Hidden size `(2 host features + 4 A4 actions)*10 = 60`; PPO MLP `[120,120]`.
- One policy parameter set shared across all five agents, with separately owned
  per-agent context/history.
- Paper targeting order: both visible events score 3, process-only 2,
  network-only 1, neither 0. Analyse uses the lowest positive suspicious score;
  Remove/Restore use the highest. Ties preserve canonical visible host order.
- Action durations Sleep/Analyse/Remove/Restore = 1/2/3/5 and PPO starting
  values gamma `.97`, learning rate `1e-4`, entropy `.01`, rollout `128`.

## Disclosed implementation choice

`torch_geometric` is not required. `TypedAttentionLayer` is a pure-PyTorch,
auditable HGT-style implementation with node-type-specific query/key/value
projections, relation-specific transforms and destination-wise softmax. This
preserves typed attention semantics but is not claimed to be bit-identical to
PyG's `HGTConv`.

Raw CC4 rows are converted from Blue-visible process/connection evidence; a
normalised upstream adapter may instead provide the two explicit malicious
event flags. Hidden compromise, Red sessions and evaluation truth are rejected
recursively at the graph boundary.

## Reward and training gate

The original training signal is retained only as a separate environment-side
hidden reward channel:

`health = -(red sessions + service unreliability * OT multiplier)`

with OT multiplier 2 (otherwise 1), and per-timestep reward equal to change in
segment health. These inputs are legal for training reward computation but are
never graph/policy inputs. The policy observation does not provide these hidden
fields. The environment-side `CC4HiddenRewardAdapter` reads active Red/Green
sessions, exact active-service reliability, and actual `OTSERVICE` host role
without exposing them to the graph. `probe_terla_reward_channel.py` verifies
those paths in the installed CC4 build. The seed-3199 1x20 probe verified exact
steps and all reward fields; its report SHA256 is
`88e952553ea3f01680685c5ab84d2daf5318b3fadfcdf461b58688df22ab0dd1`.
Operational-zone routers correctly need not contain an OT service; OT role is
therefore defined by actual `ProcessName.OTSERVICE` presence, not a hostname
proxy. The development runner passed a 20-tick train smoke and an independent-
seed 20-tick evaluation smoke. It uses one shared policy, isolated per-agent
trajectories, shared A4 resolution, true durations and episode-boundary
truncation. Hidden-reward perturbation does not change policy actions, and eval
leaves the parameter hash unchanged. The rollout target is 128; a final short
batch is flushed only at an episode boundary. These gates are development-
verified, not formal-result eligibility. No official reward or proxy is used.

Development smoke summary SHA256:
`cad41bd9f4594baaf941ad9182287fe756f57145197a1123ac5753f8db7903cd`.

## Formal training

The CPU-only formal runner trains the five frozen policy seeds independently on
episodes `1000..1031`, always for 500 ticks. It snapshots every eight training
episodes and selects the best snapshot solely by mean original-TERLA reward on
validation episodes `2000..2007` (ties choose the earlier snapshot). Test
episodes `4000..4099` are prohibited from training and selection.

From `chapter2_region_detection`, run one repeat with
`python -m baselines.terla_a4.formal_training --policy-seed 51001 --resume`.
The resumable working state is separate from the final evaluator checkpoint at
`outputs/formal_v3/training/terla_a4/policy_<seed>.pt`. A final checkpoint is
published only after all training episodes and validation selection complete.
Formal training fails closed if Git reports tracked or untracked changes. The
working state, every validation candidate, and final checkpoint bind the exact
`code_commit`, `git_dirty: false`, and `git_diff_sha256`; resume under another
commit is rejected.
The episode runtime keeps one shared policy for all five agents while owning a
separate observable tracker/history per agent. Hidden truth remains confined to
the original TERLA environment-side reward wrapper; the policy graph cannot
read it. The runner does not use OFOX, an LLM, a world model, or a GPU.

Source paper SHA256:
`e2c53cc19c3647029870d9bd9438f9c20d50ab79e728374b8461852e729800ea`.
