from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import nn
from torch.distributions import Categorical

from formal_experiments.ours.posterior_features import POSTERIOR_CANDIDATE_DIM


DEFAULT_HIDDEN_DIM = 128
DEFAULT_GAMMA_TICK = 0.99
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_CLIP_EPSILON = 0.2
DEFAULT_VALUE_COEF = 0.5
DEFAULT_ENTROPY_COEF = 0.01


@dataclass(frozen=True)
class PPOCoreConfig:
    candidate_feature_dim: int = POSTERIOR_CANDIDATE_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    gamma_tick: float = DEFAULT_GAMMA_TICK
    gae_lambda: float = DEFAULT_GAE_LAMBDA
    clip_epsilon: float = DEFAULT_CLIP_EPSILON
    value_coef: float = DEFAULT_VALUE_COEF
    entropy_coef: float = DEFAULT_ENTROPY_COEF

    def __post_init__(self) -> None:
        if int(self.candidate_feature_dim) != POSTERIOR_CANDIDATE_DIM:
            raise ValueError("formal PPO candidate feature dim must be 46")
        if int(self.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be >0")
        if not (0.0 < float(self.gamma_tick) <= 1.0):
            raise ValueError("gamma_tick must lie in (0,1]")
        if not (0.0 <= float(self.gae_lambda) <= 1.0):
            raise ValueError("gae_lambda must lie in [0,1]")
        if not (0.0 < float(self.clip_epsilon) < 1.0):
            raise ValueError("clip_epsilon must lie in (0,1)")
        if float(self.value_coef) < 0.0 or float(self.entropy_coef) < 0.0:
            raise ValueError("loss coefficients must be non-negative")


@dataclass(frozen=True)
class PolicyEvaluation:
    logits: torch.Tensor
    value: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor


@dataclass(frozen=True)
class GAEResult:
    advantages: torch.Tensor
    returns: torch.Tensor
    discounts: torch.Tensor
    deltas: torch.Tensor


@dataclass(frozen=True)
class PPOLosses:
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    ratio: torch.Tensor


def _orthogonal_linear(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=float(gain))
    nn.init.constant_(layer.bias, 0.0)


class CandidateActorCritic(nn.Module):
    """Permutation-safe actor/critic over candidate-wise 46D posterior features.

    Candidate rows are encoded independently with shared parameters. Actor logits are
    therefore candidate-permutation equivariant, while the critic mean-pools candidate
    embeddings and is candidate-order invariant.
    """

    def __init__(self, config: PPOCoreConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else PPOCoreConfig()
        d = int(self.config.candidate_feature_dim)
        h = int(self.config.hidden_dim)

        self.encoder_fc1 = nn.Linear(d, h)
        self.encoder_fc2 = nn.Linear(h, h)
        self.actor_head = nn.Linear(h, 1)
        self.critic_fc = nn.Linear(h, h)
        self.critic_head = nn.Linear(h, 1)
        self.activation = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        hidden_gain = sqrt(2.0)
        _orthogonal_linear(self.encoder_fc1, hidden_gain)
        _orthogonal_linear(self.encoder_fc2, hidden_gain)
        _orthogonal_linear(self.actor_head, 0.01)
        _orthogonal_linear(self.critic_fc, hidden_gain)
        _orthogonal_linear(self.critic_head, 1.0)

    def _validate_features(self, candidate_features) -> torch.Tensor:
        features = candidate_features
        if not torch.is_tensor(features):
            features = torch.as_tensor(features, dtype=torch.float32)
        else:
            features = features.to(dtype=torch.float32)
        if features.ndim < 2:
            raise ValueError("candidate_features must have shape [...,K,46]")
        if int(features.shape[-1]) != int(self.config.candidate_feature_dim):
            raise ValueError("candidate feature dimension mismatch")
        if int(features.shape[-2]) < 1:
            raise ValueError("candidate dimension K must be >=1")
        if not torch.isfinite(features).all():
            raise ValueError("candidate features must be finite")
        return features

    def encode_candidates(self, candidate_features) -> torch.Tensor:
        x = self._validate_features(candidate_features)
        x = self.activation(self.encoder_fc1(x))
        x = self.activation(self.encoder_fc2(x))
        return x

    def forward(self, candidate_features) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encode_candidates(candidate_features)
        logits = self.actor_head(embeddings).squeeze(-1)
        pooled = embeddings.mean(dim=-2)
        value = self.critic_head(self.activation(self.critic_fc(pooled))).squeeze(-1)
        if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
            raise RuntimeError("actor/critic emitted non-finite output")
        return logits, value

    def distribution(self, candidate_features) -> tuple[Categorical, torch.Tensor]:
        logits, value = self(candidate_features)
        return Categorical(logits=logits), value

    def act(
        self,
        candidate_features,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.distribution(candidate_features)
        action = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def evaluate_actions(self, candidate_features, actions) -> PolicyEvaluation:
        dist, value = self.distribution(candidate_features)
        action_t = torch.as_tensor(actions, dtype=torch.long, device=dist.logits.device)
        expected_shape = tuple(dist.logits.shape[:-1])
        if tuple(action_t.shape) != expected_shape:
            raise ValueError(
                f"actions must have shape {expected_shape}; got {tuple(action_t.shape)}"
            )
        k = int(dist.logits.shape[-1])
        if action_t.numel() and (int(action_t.min()) < 0 or int(action_t.max()) >= k):
            raise ValueError("candidate action index out of range")
        log_prob = dist.log_prob(action_t)
        entropy = dist.entropy()
        return PolicyEvaluation(
            logits=dist.logits,
            value=value,
            log_prob=log_prob,
            entropy=entropy,
        )


def _as_float_vector(value, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")
    return tensor


def duration_discounts(decision_dt, *, gamma_tick: float = DEFAULT_GAMMA_TICK) -> torch.Tensor:
    gamma = float(gamma_tick)
    if not (0.0 < gamma <= 1.0):
        raise ValueError("gamma_tick must lie in (0,1]")
    dt = _as_float_vector(decision_dt, name="decision_dt")
    if torch.any(dt < 1.0):
        raise ValueError("decision_dt must be >=1 tick")
    if not torch.allclose(dt, torch.round(dt), rtol=0.0, atol=1e-6):
        raise ValueError("decision_dt must contain integer tick durations")
    return torch.pow(torch.full_like(dt, gamma), dt)


def compute_duration_aware_gae(
    real_response_rewards,
    values,
    next_values,
    dones,
    decision_dt,
    *,
    gamma_tick: float = DEFAULT_GAMMA_TICK,
    gae_lambda: float = DEFAULT_GAE_LAMBDA,
) -> GAEResult:
    """Compute GAE for one ordered agent trajectory in decision-epoch time.

    `real_response_rewards[t]` is the accumulated *real CC4 response reward* over the
    executed action's decision interval. World-model predicted value is not a reward
    target and is intentionally absent from this API.
    """

    lam = float(gae_lambda)
    if not (0.0 <= lam <= 1.0):
        raise ValueError("gae_lambda must lie in [0,1]")

    rewards = _as_float_vector(real_response_rewards, name="real_response_rewards")
    values_t = _as_float_vector(values, name="values").to(device=rewards.device)
    next_values_t = _as_float_vector(next_values, name="next_values").to(device=rewards.device)
    discounts = duration_discounts(decision_dt, gamma_tick=gamma_tick).to(device=rewards.device)
    done_t = torch.as_tensor(dones, dtype=torch.bool, device=rewards.device)
    if done_t.ndim != 1:
        raise ValueError("dones must be one-dimensional")

    n = int(rewards.numel())
    for name, tensor in (
        ("values", values_t),
        ("next_values", next_values_t),
        ("discounts", discounts),
        ("dones", done_t),
    ):
        if int(tensor.numel()) != n:
            raise ValueError(f"{name} length must match rewards")
    if n == 0:
        raise ValueError("trajectory must contain at least one transition")

    not_done = (~done_t).to(dtype=torch.float32)
    deltas = rewards + discounts * not_done * next_values_t - values_t
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=torch.float32, device=rewards.device)
    for index in range(n - 1, -1, -1):
        gae = deltas[index] + discounts[index] * lam * not_done[index] * gae
        advantages[index] = gae
    returns = advantages + values_t

    if not torch.isfinite(advantages).all() or not torch.isfinite(returns).all():
        raise RuntimeError("GAE emitted non-finite values")
    return GAEResult(
        advantages=advantages,
        returns=returns,
        discounts=discounts,
        deltas=deltas,
    )


def normalize_advantages(advantages, *, eps: float = 1e-8) -> torch.Tensor:
    adv = _as_float_vector(advantages, name="advantages")
    if adv.numel() == 0:
        raise ValueError("advantages must be non-empty")
    if float(eps) <= 0.0:
        raise ValueError("eps must be >0")
    mean = adv.mean()
    std = adv.std(unbiased=False)
    return (adv - mean) / (std + float(eps))


def compute_ppo_losses(
    new_log_prob,
    old_log_prob,
    advantages,
    new_values,
    returns,
    entropy,
    *,
    clip_epsilon: float = DEFAULT_CLIP_EPSILON,
    value_coef: float = DEFAULT_VALUE_COEF,
    entropy_coef: float = DEFAULT_ENTROPY_COEF,
) -> PPOLosses:
    """Pure clipped-PPO objective for an already-collected minibatch."""

    clip = float(clip_epsilon)
    if not (0.0 < clip < 1.0):
        raise ValueError("clip_epsilon must lie in (0,1)")
    if float(value_coef) < 0.0 or float(entropy_coef) < 0.0:
        raise ValueError("loss coefficients must be non-negative")

    new_lp = _as_float_vector(new_log_prob, name="new_log_prob")
    old_lp = _as_float_vector(old_log_prob, name="old_log_prob")
    adv = _as_float_vector(advantages, name="advantages")
    value = _as_float_vector(new_values, name="new_values")
    target = _as_float_vector(returns, name="returns")
    ent = _as_float_vector(entropy, name="entropy")
    n = int(new_lp.numel())
    if n == 0:
        raise ValueError("PPO minibatch must be non-empty")
    for name, tensor in (
        ("old_log_prob", old_lp),
        ("advantages", adv),
        ("new_values", value),
        ("returns", target),
        ("entropy", ent),
    ):
        if int(tensor.numel()) != n:
            raise ValueError(f"{name} length must match new_log_prob")

    log_ratio = new_lp - old_lp
    ratio = torch.exp(log_ratio)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = 0.5 * torch.mean((value - target) ** 2)
    entropy_mean = ent.mean()
    total = policy_loss + float(value_coef) * value_loss - float(entropy_coef) * entropy_mean
    approx_kl = torch.mean((ratio - 1.0) - log_ratio)
    clip_fraction = torch.mean((torch.abs(ratio - 1.0) > clip).to(torch.float32))

    for tensor in (total, policy_loss, value_loss, entropy_mean, approx_kl, clip_fraction, ratio):
        if not torch.isfinite(tensor).all():
            raise RuntimeError("PPO objective emitted non-finite values")

    return PPOLosses(
        total_loss=total,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy_mean,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        ratio=ratio,
    )
