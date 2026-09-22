from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import nn
from torch.distributions import Categorical

from shared.action_contract import N_ACTIONS
from shared.formal_state import FORMAL_STATE_DIM


class OfflineActionPrior:
    """Read-only bridge to an already-built cache; it has no generation fallback."""

    def __init__(self, cache_lookup, *, epsilon: float = 1e-6):
        if not callable(cache_lookup):
            raise TypeError("cache_lookup must be callable")
        self.cache_lookup = cache_lookup
        self.epsilon = float(epsilon)

    def load(self, state, *, agent_name: str) -> torch.Tensor:
        cached = self.cache_lookup(state, agent_name=agent_name)
        if cached is None:
            raise RuntimeError("PriorRL offline cache miss (fail closed)")
        prior = getattr(cached, "prior", cached)
        if not hasattr(prior, "plans") or not hasattr(prior, "prior_preferences"):
            raise TypeError("cache lookup did not return a PriorBatch-compatible object")
        return action_prior_from_plans(prior.plans, prior.prior_preferences, epsilon=self.epsilon)


def action_prior_from_plans(plans, preferences, *, epsilon: float = 1e-6) -> torch.Tensor:
    plans_t = torch.as_tensor(plans, dtype=torch.long)
    weights = torch.as_tensor(preferences, dtype=torch.float32)
    if plans_t.ndim != 2 or plans_t.shape[1] != 4 or plans_t.shape[0] != 6:
        raise ValueError("offline prior must contain K=6, H=4 plans")
    if weights.shape != (6,) or not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise ValueError("prior preferences must be six finite nonnegative values")
    if torch.any(plans_t < 0) or torch.any(plans_t >= N_ACTIONS):
        raise ValueError("prior plan contains an action outside A4")
    if not isfinite(float(epsilon)) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    mass = torch.full((N_ACTIONS,), float(epsilon), dtype=torch.float32)
    mass.scatter_add_(0, plans_t[:, 0], weights)
    return mass / mass.sum()


def masked_logits_and_prior(policy_logits: torch.Tensor, llm_prior: torch.Tensor, action_mask=None):
    prior = torch.as_tensor(llm_prior, dtype=policy_logits.dtype, device=policy_logits.device)
    if prior.shape != policy_logits.shape or torch.any(prior < 0) or not torch.isfinite(prior).all():
        raise ValueError("LLM action prior must be nonnegative, finite, and match policy shape")
    mask = torch.ones_like(policy_logits, dtype=torch.bool) if action_mask is None else torch.as_tensor(
        action_mask, dtype=torch.bool, device=policy_logits.device)
    if mask.shape != policy_logits.shape or torch.any(~mask[..., 0]) or torch.any(mask.sum(-1) < 1):
        raise ValueError("A4 mask must match logits and keep Sleep legal")
    masked_logits = policy_logits.masked_fill(~mask, torch.finfo(policy_logits.dtype).min)
    legal_prior = torch.where(mask, prior, torch.zeros_like(prior))
    legal_prior = legal_prior + mask.to(prior.dtype) * torch.finfo(prior.dtype).eps
    legal_prior = legal_prior / legal_prior.sum(dim=-1, keepdim=True)
    return masked_logits, legal_prior


def forward_kl(policy_logits: torch.Tensor, llm_prior: torch.Tensor, action_mask=None) -> torch.Tensor:
    """KL(pi || p_LLM), never the reverse direction."""
    policy_logits, prior = masked_logits_and_prior(policy_logits, llm_prior, action_mask)
    mask = torch.ones_like(policy_logits, dtype=torch.bool) if action_mask is None else torch.as_tensor(
        action_mask, dtype=torch.bool, device=policy_logits.device)
    log_pi = torch.log_softmax(policy_logits, dim=-1)
    pi = log_pi.exp()
    # Never evaluate log(0) on masked entries; torch.where alone would still
    # construct a NaN-producing branch and can poison backward gradients.
    safe_prior = torch.where(mask, prior, torch.ones_like(prior))
    terms = torch.where(mask, pi * (log_pi - safe_prior.log()), torch.zeros_like(pi))
    return terms.sum(dim=-1).mean()


def prior_regularized_ppo_loss(ppo_loss: torch.Tensor, policy_logits: torch.Tensor,
                               llm_prior: torch.Tensor, *, alpha_kl: float) -> torch.Tensor:
    alpha = float(alpha_kl)
    if not isfinite(alpha) or alpha < 0:
        raise ValueError("alpha_kl must be finite and >= 0")
    return ppo_loss + alpha * forward_kl(policy_logits, llm_prior)


@dataclass(frozen=True)
class PriorRLPPOConfig:
    method_name: str = "PriorRL-PPO-CC4"
    hidden_dim: int = 128
    alpha_kl: float = 0.05
    prior_epsilon: float = 1e-6
    gamma_tick: float = 0.99
    cache_miss_policy: str = "fail_closed"
    online_llm_allowed: bool = False
    world_model_allowed: bool = False

    def __post_init__(self) -> None:
        if self.method_name != "PriorRL-PPO-CC4" or self.hidden_dim <= 0:
            raise ValueError("invalid PriorRL identity or network size")
        if self.cache_miss_policy != "fail_closed" or self.online_llm_allowed or self.world_model_allowed:
            raise ValueError("PriorRL requires offline-only priors and forbids world-model use")
        if self.alpha_kl < 0 or self.prior_epsilon <= 0 or not 0 < self.gamma_tick <= 1:
            raise ValueError("invalid PriorRL optimization parameters")


class PriorRLActorCritic(nn.Module):
    def __init__(self, config: PriorRLPPOConfig | None = None):
        super().__init__()
        self.config = config or PriorRLPPOConfig()
        self.body = nn.Sequential(nn.Linear(FORMAL_STATE_DIM, self.config.hidden_dim), nn.ReLU(),
                                  nn.Linear(self.config.hidden_dim, self.config.hidden_dim), nn.ReLU())
        self.actor = nn.Linear(self.config.hidden_dim, N_ACTIONS)
        self.critic = nn.Linear(self.config.hidden_dim, 1)

    def forward(self, state):
        device = next(self.parameters()).device
        x = torch.as_tensor(state, dtype=torch.float32, device=device)
        if x.shape[-1] != FORMAL_STATE_DIM or not torch.isfinite(x).all():
            raise ValueError("PriorRL state must be finite D27")
        h = self.body(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, state, *, deterministic: bool = False, action_mask=None):
        logits, value = self(state)
        if action_mask is not None:
            mask = torch.as_tensor(action_mask, dtype=torch.bool, device=logits.device)
            if mask.shape != logits.shape or not bool(mask[..., 0].all()):
                raise ValueError("action mask must match A4 logits and keep Sleep legal")
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value
