from __future__ import annotations

from dataclasses import dataclass
import hashlib
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class CARLPolicyConfig:
    state_dim: int = 27
    actions: int = 4
    hidden_dim: int = 128
    gamma_tick: float = 0.99


class CARLActorCritic(nn.Module):
    """D27-to-A4 PPO policy; hidden truth is absent from this API."""
    def __init__(self, config: CARLPolicyConfig | None = None):
        super().__init__(); self.config = config or CARLPolicyConfig()
        self.body = nn.Sequential(nn.Linear(27, self.config.hidden_dim), nn.ReLU(),
                                  nn.Linear(self.config.hidden_dim, self.config.hidden_dim), nn.ReLU())
        self.actor = nn.Linear(self.config.hidden_dim, 4); self.critic = nn.Linear(self.config.hidden_dim, 1)

    def forward(self, observable_d27):
        device = next(self.parameters()).device
        state = torch.as_tensor(observable_d27, dtype=torch.float32, device=device)
        if state.shape[-1] != 27 or not torch.isfinite(state).all():
            raise ValueError("CARL policy accepts finite Blue-observable D27 only")
        hidden = self.body(state)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def act(self, observable_d27, *, deterministic=False):
        logits, value = self(observable_d27); distribution = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action), value


def policy_fingerprint(policy: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        digest.update(name.encode("utf-8")); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FrozenEvaluationPolicy:
    def __init__(self, policy: CARLActorCritic):
        self.policy = policy.eval(); self._fingerprint = policy_fingerprint(policy)

    def act(self, observable_d27, *, deterministic=True):
        if policy_fingerprint(self.policy) != self._fingerprint:
            raise RuntimeError("CARL policy mutated during evaluation")
        with torch.inference_mode():
            return self.policy.act(observable_d27, deterministic=deterministic)
