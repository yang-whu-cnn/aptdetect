from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
import torch

from formal_experiments.ours.ppo_core import (
    compute_duration_aware_gae, compute_ppo_losses, normalize_advantages,
)


@dataclass(frozen=True)
class VariantTransition:
    agent_name: str
    policy_input: torch.Tensor
    policy_mask: torch.Tensor
    policy_action: int
    old_log_prob: float
    value: float
    next_value: float
    reward: float
    duration: int
    done: bool
    split: str


class VariantPPOBuffer:
    def __init__(self, *, split: str):
        if split not in ("train", "validation", "test"): raise ValueError("invalid split")
        self.split = split; self.transitions: list[VariantTransition] = []
    def append(self, item: VariantTransition):
        if item.split != self.split: raise ValueError("buffer split contamination")
        self.transitions.append(item)


def ppo_update(runtime, buffer: VariantPPOBuffer, *, learning_rate=3e-4) -> dict:
    if buffer.split != "train": raise RuntimeError("optimizer updates are train-split only")
    if not buffer.transitions: raise ValueError("empty PPO buffer")
    items = buffer.transitions
    # GAE recurrences must never cross agent trajectories. Transitions are kept
    # in global completion order for audit, so scatter per-agent results back.
    advantages = torch.empty(len(items)); returns = torch.empty(len(items)); discounts = torch.empty(len(items))
    by_agent: dict[str, list[int]] = {}
    for index, item in enumerate(items): by_agent.setdefault(item.agent_name, []).append(index)
    for indices in by_agent.values():
        seq = [items[index] for index in indices]
        # The next decision's critic value is the duration-aware TD bootstrap.
        # The final completed decision uses its explicitly recorded cutoff value
        # (zero in the bounded development smoke).
        next_values = [seq[index + 1].value for index in range(len(seq) - 1)] + [seq[-1].next_value]
        gae = compute_duration_aware_gae(
            [x.reward for x in seq], [x.value for x in seq], next_values,
            [x.done for x in seq], [x.duration for x in seq],
        )
        advantages[indices] = gae.advantages; returns[indices] = gae.returns; discounts[indices] = gae.discounts
    optimizer = torch.optim.Adam(runtime.policy.parameters(), lr=learning_rate)
    device = next(runtime.policy.parameters()).device
    inputs = torch.stack([x.policy_input for x in items]).to(device)
    actions = torch.tensor([x.policy_action for x in items], device=device)
    masks = torch.stack([x.policy_mask for x in items]).to(device=device, dtype=torch.bool)
    logits, values = runtime.policy(inputs)
    if logits.shape != masks.shape or torch.any(~masks.gather(1, actions.view(-1, 1))):
        raise ValueError("stored PPO action mask is invalid")
    distribution = torch.distributions.Categorical(logits=logits.masked_fill(~masks, float("-inf")))
    losses = compute_ppo_losses(
        distribution.log_prob(actions), torch.tensor([x.old_log_prob for x in items], device=device),
        normalize_advantages(advantages).to(device), values, returns.to(device), distribution.entropy(),
    )
    optimizer.zero_grad(); losses.total_loss.backward(); optimizer.step()
    duration_counts = dict(sorted(Counter(x.duration for x in items).items()))
    discount_sha = hashlib.sha256(discounts.numpy().tobytes()).hexdigest()
    return_abs_max_by_agent = {
        agent: float(returns[indices].abs().max()) for agent, indices in sorted(by_agent.items())
    }
    return {"samples": len(items), "loss": float(losses.total_loss.detach()),
            "policy_loss": float(losses.policy_loss.detach()),
            "value_loss": float(losses.value_loss.detach()),
            "reward_min": min(x.reward for x in items), "reward_max": max(x.reward for x in items),
            "return_abs_max": float(returns.abs().max()), "duration_counts": duration_counts,
            "return_abs_max_by_agent": return_abs_max_by_agent,
            "duration_discounts_sha256": discount_sha}
