"""A4 PriorRL PPO core. Collection freezes priors; optimization is cache-free."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import hashlib
from typing import Iterable

import torch
from torch import nn
from torch.distributions import Categorical

from formal_experiments.ours.ppo_core import (
    compute_duration_aware_gae, compute_ppo_losses, normalize_advantages,
)
from shared.action_contract import N_ACTIONS
from shared.formal_state import FORMAL_STATE_DIM
from .policy import PriorRLActorCritic, forward_kl


@dataclass(frozen=True)
class A4PPOConfig:
    gamma_tick: float = .99
    gae_lambda: float = .95
    clip_epsilon: float = .2
    value_coef: float = .5
    entropy_coef: float = .01
    alpha_kl: float = .05
    learning_rate: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 1.0

    def __post_init__(self):
        values = (self.gamma_tick, self.gae_lambda, self.clip_epsilon, self.value_coef,
                  self.entropy_coef, self.alpha_kl, self.learning_rate, self.max_grad_norm)
        if not all(isfinite(float(x)) for x in values):
            raise ValueError("PPO config must be finite")
        if not 0 < self.gamma_tick <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("invalid duration-aware discount configuration")
        if not 0 < self.clip_epsilon < 1 or min(self.value_coef, self.entropy_coef, self.alpha_kl) < 0:
            raise ValueError("invalid PPO loss configuration")
        if self.learning_rate <= 0 or self.epochs <= 0 or self.minibatch_size <= 0 or self.max_grad_norm <= 0:
            raise ValueError("invalid PPO optimizer configuration")


@dataclass(frozen=True)
class A4Transition:
    state: torch.Tensor
    action: int
    old_log_prob: float
    value: float
    reward: float
    next_value: float
    done: bool
    decision_dt: int
    frozen_prior: torch.Tensor
    action_mask: torch.Tensor
    trajectory_id: str


class A4RolloutBuffer:
    def __init__(self): self._items: list[A4Transition] = []
    def __len__(self): return len(self._items)
    def clear(self): self._items.clear()
    def add(self, *, state, action, old_log_prob, value, reward, next_value, done,
            decision_dt, frozen_prior, action_mask=(True, True, True, True), trajectory_id=""):
        state_t = torch.as_tensor(state, dtype=torch.float32).detach().cpu().clone()
        prior_t = torch.as_tensor(frozen_prior, dtype=torch.float32).detach().cpu().clone()
        if state_t.shape != (FORMAL_STATE_DIM,) or not torch.isfinite(state_t).all():
            raise ValueError("transition state must be finite D27")
        if prior_t.shape != (N_ACTIONS,) or not torch.isfinite(prior_t).all() or torch.any(prior_t < 0):
            raise ValueError("transition prior must be nonnegative finite A4")
        mask_t = torch.as_tensor(action_mask, dtype=torch.bool).detach().cpu().clone()
        if mask_t.shape != (N_ACTIONS,) or not bool(mask_t[0]):
            raise ValueError("transition mask must be A4 with Sleep legal")
        prior_t = torch.where(mask_t, prior_t, torch.zeros_like(prior_t))
        prior_t = prior_t + mask_t.to(prior_t.dtype) * torch.finfo(prior_t.dtype).eps
        prior_t /= prior_t.sum()
        if int(action) not in range(N_ACTIONS) or int(decision_dt) < 1:
            raise ValueError("invalid A4 action or decision duration")
        scalars = tuple(float(x) for x in (old_log_prob, value, reward, next_value))
        if not all(isfinite(x) for x in scalars):
            raise ValueError("transition log-prob/value/reward/next-value must be finite")
        if bool(done) and abs(scalars[3]) > 1e-7:
            raise ValueError("terminal transition next_value must be zero")
        trajectory = str(trajectory_id).strip()
        if not trajectory:
            raise ValueError("trajectory_id must be non-empty")
        self._items.append(A4Transition(state_t, int(action), scalars[0], scalars[1],
            scalars[2], scalars[3], bool(done), int(decision_dt), prior_t, mask_t, trajectory))

    def training_tensors(self, config: A4PPOConfig):
        if not self._items: raise ValueError("empty A4 rollout buffer")
        adv, ret = [], []
        for trajectory in sorted({x.trajectory_id for x in self._items}):
            rows = [x for x in self._items if x.trajectory_id == trajectory]
            terminal_indices = [index for index, row in enumerate(rows) if row.done]
            if terminal_indices and terminal_indices != [len(rows) - 1]:
                raise ValueError("trajectory contains transitions after terminal state")
            result = compute_duration_aware_gae(
                [x.reward for x in rows], [x.value for x in rows], [x.next_value for x in rows],
                [x.done for x in rows], [x.decision_dt for x in rows],
                gamma_tick=config.gamma_tick, gae_lambda=config.gae_lambda)
            adv.append(result.advantages); ret.append(result.returns)
        advantages = normalize_advantages(torch.cat(adv))
        returns = torch.cat(ret)
        # Match the same trajectory grouping order used above.
        ordered = [x for tid in sorted({x.trajectory_id for x in self._items})
                   for x in self._items if x.trajectory_id == tid]
        return {
            "states": torch.stack([x.state for x in ordered]),
            "actions": torch.tensor([x.action for x in ordered], dtype=torch.long),
            "old_log_probs": torch.tensor([x.old_log_prob for x in ordered]),
            "advantages": advantages, "returns": returns,
            "priors": torch.stack([x.frozen_prior for x in ordered]),
            "action_masks": torch.stack([x.action_mask for x in ordered]),
            "decision_dt": torch.tensor([x.decision_dt for x in ordered]),
        }


class A4PPOTrainer:
    """Optimizer deliberately owns no retriever/cache/API reference."""
    def __init__(self, policy: PriorRLActorCritic, config: A4PPOConfig | None = None):
        self.policy = policy; self.config = config or A4PPOConfig()
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.config.learning_rate)

    def optimize(self, buffer: A4RolloutBuffer, *, generator: torch.Generator | None = None):
        batch = buffer.training_tensors(self.config); device = next(self.policy.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}; n = len(batch["actions"])
        metrics = []
        for _ in range(self.config.epochs):
            order = torch.randperm(n, generator=generator)
            for start in range(0, n, self.config.minibatch_size):
                ix = order[start:start+self.config.minibatch_size].to(device)
                logits, values = self.policy(batch["states"][ix])
                masked_logits = logits.masked_fill(~batch["action_masks"][ix], torch.finfo(logits.dtype).min)
                dist = Categorical(logits=masked_logits)
                losses = compute_ppo_losses(
                    dist.log_prob(batch["actions"][ix]), batch["old_log_probs"][ix],
                    batch["advantages"][ix], values, batch["returns"][ix], dist.entropy(),
                    clip_epsilon=self.config.clip_epsilon, value_coef=self.config.value_coef,
                    entropy_coef=self.config.entropy_coef)
                prior_kl = forward_kl(logits, batch["priors"][ix], batch["action_masks"][ix])
                total = losses.total_loss + self.config.alpha_kl * prior_kl
                self.optimizer.zero_grad(set_to_none=True); total.backward()
                grad = nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                metrics.append({"total_loss": float(total.detach().cpu()),
                    "policy_loss": float(losses.policy_loss.detach().cpu()),
                    "value_loss": float(losses.value_loss.detach().cpu()),
                    "entropy": float(losses.entropy.detach().cpu()),
                    "prior_forward_kl": float(prior_kl.detach().cpu()),
                    "grad_norm": float(torch.as_tensor(grad).detach().cpu())})
        return metrics


def checkpoint_sha256(policy: PriorRLActorCritic) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()
