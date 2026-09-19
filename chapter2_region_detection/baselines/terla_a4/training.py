from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn
from torch.distributions import Categorical

from .graph import HeteroGraph
from .model import TERLAPolicy


@dataclass
class Transition:
    graph: HeteroGraph; action: int; old_log_prob: float; value: float
    reward: float; next_value: float; done: bool; duration: int; trajectory: str
    availability_mask: tuple[bool, bool, bool, bool] = (True, True, True, True)


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = .97; gae_lambda: float = .95; learning_rate: float = 1e-4
    entropy: float = .01; clip: float = .2; value_coef: float = .5
    rollout: int = 128; epochs: int = 2


def policy_hash(policy: TERLAPolicy) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class TERLAPPOTrainer:
    def __init__(self, policy: TERLAPolicy, config: PPOConfig | None = None):
        self.policy = policy; self.config = config or PPOConfig()
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.config.learning_rate)
        self.buffer: list[Transition] = []

    def add(self, rows: list[Transition]) -> None:
        self.buffer.extend(rows)

    def flush_episode(self) -> list[dict]:
        """Flush full rollout fragments and the final episode-boundary fragment."""
        reports = []
        while len(self.buffer) >= self.config.rollout:
            reports.append(self.optimize(self.buffer[:self.config.rollout]))
            del self.buffer[:self.config.rollout]
        if self.buffer:
            reports.append(self.optimize(self.buffer))
            self.buffer.clear()
        return reports

    def optimize(self, rows: list[Transition]) -> dict:
        if not rows: raise ValueError("empty TERLA rollout")
        advantages = torch.zeros(len(rows)); returns = torch.zeros(len(rows))
        by_trajectory = sorted({row.trajectory for row in rows})
        for trajectory in by_trajectory:
            indices = [i for i, row in enumerate(rows) if row.trajectory == trajectory]
            gae = 0.0
            for i in reversed(indices):
                row = rows[i]; discount = self.config.gamma ** row.duration
                delta = row.reward + discount * row.next_value * (not row.done) - row.value
                gae = delta + discount * self.config.gae_lambda * (not row.done) * gae
                advantages[i] = gae; returns[i] = gae + row.value
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        losses = []
        device = next(self.policy.parameters()).device
        for _ in range(self.config.epochs):
            for i, row in enumerate(rows):
                logits, value = self.policy(row.graph.to(device))
                mask = torch.tensor(row.availability_mask, dtype=torch.bool, device=device)
                if not bool(mask[row.action]):
                    raise RuntimeError("TERLA rollout action is unavailable under stored mask")
                masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
                dist = Categorical(logits=masked_logits)
                ratio = torch.exp(dist.log_prob(torch.tensor(row.action, device=device)) - row.old_log_prob)
                advantage = advantages[i].to(device); target_return = returns[i].to(device)
                surrogate = torch.minimum(ratio * advantage,
                                          ratio.clamp(1-self.config.clip, 1+self.config.clip)*advantage)
                loss = -surrogate + self.config.value_coef*(value-target_return).square() - self.config.entropy*dist.entropy()
                self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step(); losses.append(float(loss.detach()))
        return {"transitions": len(rows), "updates": len(losses), "mean_loss": sum(losses)/len(losses)}
