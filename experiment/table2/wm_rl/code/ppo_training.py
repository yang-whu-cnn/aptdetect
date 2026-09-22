from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

import numpy as np
import torch
from torch import nn

from formal_experiments.ours.posterior_features import POSTERIOR_CANDIDATE_DIM
from formal_experiments.ours.ppo_core import (
    CandidateActorCritic,
    PPOCoreConfig,
    compute_duration_aware_gae,
    compute_ppo_losses,
    normalize_advantages,
)
from shared.formal_state import BLUE_AGENTS


FORMAL_K = 6
FORMAL_TRAIN_SEEDS = tuple(range(1000, 1032))


@dataclass(frozen=True)
class PPOTrainingConfig:
    learning_rate: float = 3e-4
    rollout_length: int = 128
    update_epochs: int = 5
    minibatch_size: int = 64
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    seed: int = 20260917

    def __post_init__(self) -> None:
        if not isfinite(float(self.learning_rate)) or float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be finite and >0")
        if int(self.rollout_length) <= 0:
            raise ValueError("rollout_length must be >0")
        if int(self.update_epochs) <= 0:
            raise ValueError("update_epochs must be >0")
        if int(self.minibatch_size) <= 0:
            raise ValueError("minibatch_size must be >0")
        if not isfinite(float(self.max_grad_norm)) or float(self.max_grad_norm) <= 0.0:
            raise ValueError("max_grad_norm must be finite and >0")


@dataclass(frozen=True)
class RolloutStep:
    episode_seed: int
    agent_name: str
    decision_index: int
    candidate_features: torch.Tensor
    candidate_index: int
    behavior_log_prob: float
    critic_value: float
    critic_next_value: float
    real_response_reward: float
    decision_dt: int
    done: bool


@dataclass(frozen=True)
class PPOTrainingBatch:
    candidate_features: torch.Tensor
    candidate_indices: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    critic_values: torch.Tensor
    real_response_rewards: torch.Tensor
    decision_dt: torch.Tensor
    dones: torch.Tensor
    provenance: tuple[tuple[int, str, int], ...]

    @property
    def size(self) -> int:
        return int(self.candidate_indices.numel())


@dataclass(frozen=True)
class PPOUpdateMetrics:
    samples: int
    optimizer_steps: int
    policy_loss_mean: float
    value_loss_mean: float
    entropy_mean: float
    approx_kl_mean: float
    clip_fraction_mean: float
    grad_norm_mean: float
    grad_norm_max: float


@dataclass
class _PendingDecision:
    episode_seed: int
    agent_name: str
    decision_index: int
    candidate_features: torch.Tensor
    candidate_index: int
    behavior_log_prob: float
    critic_value: float


def _snapshot_features(candidate_features) -> torch.Tensor:
    features = torch.as_tensor(candidate_features, dtype=torch.float32).detach().cpu().clone()
    if tuple(features.shape) != (FORMAL_K, POSTERIOR_CANDIDATE_DIM):
        raise ValueError("formal rollout candidate_features must have shape [6,46]")
    if not torch.isfinite(features).all():
        raise ValueError("candidate_features must be finite")
    return features


def _finite_scalar(value, name: str) -> float:
    out = float(value)
    if not isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def assert_formal_train_seed(seed: int) -> int:
    value = int(seed)
    if value not in FORMAL_TRAIN_SEEDS:
        raise ValueError(f"PPO collection seed must be frozen train split; got {value}")
    return value


class AsyncPPORolloutBuffer:
    """Asynchronous per-agent PPO decision buffer.

    A decision is opened when an agent is ready and the policy selects one of K=6
    candidate plans. It is completed only after the executed CC4 decision interval has
    finished and its *real* accumulated response reward / duration are known.

    The buffer deliberately stores no provider client, LLM response, hidden truth, or
    world-model reward target. The world-model candidate value may only exist inside the
    frozen 46D candidate feature snapshot.
    """

    def __init__(self, *, train_only: bool = True) -> None:
        self.train_only = bool(train_only)
        self._open: dict[str, _PendingDecision] = {}
        self._steps: list[RolloutStep] = []
        self._next_index: dict[tuple[int, str], int] = {}

    @property
    def open_agents(self) -> tuple[str, ...]:
        return tuple(sorted(self._open))

    @property
    def steps(self) -> tuple[RolloutStep, ...]:
        return tuple(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def clear_completed(self) -> None:
        if self._open:
            raise RuntimeError("cannot clear completed steps while decisions are open")
        self._steps.clear()

    def begin(
        self,
        *,
        episode_seed: int,
        agent_name: str,
        decision_index: int,
        candidate_features,
        candidate_index: int,
        behavior_log_prob: float,
        critic_value: float,
    ) -> None:
        seed = int(episode_seed)
        if self.train_only:
            seed = assert_formal_train_seed(seed)
        agent = str(agent_name)
        if agent not in BLUE_AGENTS:
            raise ValueError(f"unsupported blue agent: {agent}")
        if agent in self._open:
            raise RuntimeError(f"{agent}: PPO decision already open")

        index = int(decision_index)
        if index < 0:
            raise ValueError("decision_index must be >=0")
        key = (seed, agent)
        expected = self._next_index.get(key, 0)
        if index != expected:
            raise ValueError(f"{agent}: decision_index={index}, expected={expected}")

        action = int(candidate_index)
        if not 0 <= action < FORMAL_K:
            raise ValueError("candidate_index must lie in [0,5]")

        self._open[agent] = _PendingDecision(
            episode_seed=seed,
            agent_name=agent,
            decision_index=index,
            candidate_features=_snapshot_features(candidate_features),
            candidate_index=action,
            behavior_log_prob=_finite_scalar(behavior_log_prob, "behavior_log_prob"),
            critic_value=_finite_scalar(critic_value, "critic_value"),
        )

    def complete(
        self,
        *,
        agent_name: str,
        real_response_reward: float,
        decision_dt: int,
        done: bool,
        critic_next_value: float,
    ) -> RolloutStep:
        agent = str(agent_name)
        if agent not in self._open:
            raise RuntimeError(f"{agent}: no open PPO decision")
        dt = int(decision_dt)
        if dt < 1:
            raise ValueError("decision_dt must be >=1")
        reward = _finite_scalar(real_response_reward, "real_response_reward")
        next_value = _finite_scalar(critic_next_value, "critic_next_value")
        terminal = bool(done)
        if terminal and abs(next_value) > 1e-6:
            raise ValueError("terminal PPO transition must use critic_next_value=0")

        pending = self._open.pop(agent)
        step = RolloutStep(
            episode_seed=pending.episode_seed,
            agent_name=pending.agent_name,
            decision_index=pending.decision_index,
            candidate_features=pending.candidate_features,
            candidate_index=pending.candidate_index,
            behavior_log_prob=pending.behavior_log_prob,
            critic_value=pending.critic_value,
            critic_next_value=next_value,
            real_response_reward=reward,
            decision_dt=dt,
            done=terminal,
        )
        self._steps.append(step)
        key = (pending.episode_seed, pending.agent_name)
        self._next_index[key] = pending.decision_index + 1
        return step


def _trajectory_groups(steps: Iterable[RolloutStep]) -> list[list[RolloutStep]]:
    grouped: dict[tuple[int, str], list[RolloutStep]] = {}
    for step in steps:
        if not isinstance(step, RolloutStep):
            raise TypeError("steps must contain RolloutStep")
        grouped.setdefault((step.episode_seed, step.agent_name), []).append(step)

    result: list[list[RolloutStep]] = []
    for key in sorted(grouped):
        trajectory = sorted(grouped[key], key=lambda item: item.decision_index)
        for position, step in enumerate(trajectory):
            if position and step.decision_index != trajectory[position - 1].decision_index + 1:
                raise ValueError(f"non-contiguous decision indices in trajectory {key}")
            if position < len(trajectory) - 1 and trajectory[position].done:
                raise ValueError(f"trajectory {key} contains transitions after terminal step")
        result.append(trajectory)
    return result


def build_training_batch(
    steps: Iterable[RolloutStep],
    *,
    core_config: PPOCoreConfig | None = None,
    normalize_advantage: bool = True,
) -> PPOTrainingBatch:
    cfg = core_config if core_config is not None else PPOCoreConfig()
    trajectories = _trajectory_groups(tuple(steps))
    if not trajectories:
        raise ValueError("at least one completed PPO transition is required")

    flat: list[RolloutStep] = []
    advantage_parts: list[torch.Tensor] = []
    return_parts: list[torch.Tensor] = []

    for trajectory in trajectories:
        rewards = torch.tensor([item.real_response_reward for item in trajectory], dtype=torch.float32)
        values = torch.tensor([item.critic_value for item in trajectory], dtype=torch.float32)
        next_values = torch.tensor([item.critic_next_value for item in trajectory], dtype=torch.float32)
        dones = torch.tensor([item.done for item in trajectory], dtype=torch.bool)
        decision_dt = torch.tensor([item.decision_dt for item in trajectory], dtype=torch.float32)
        gae = compute_duration_aware_gae(
            rewards,
            values,
            next_values,
            dones,
            decision_dt,
            gamma_tick=cfg.gamma_tick,
            gae_lambda=cfg.gae_lambda,
        )
        flat.extend(trajectory)
        advantage_parts.append(gae.advantages)
        return_parts.append(gae.returns)

    advantages = torch.cat(advantage_parts, dim=0)
    if normalize_advantage:
        advantages = normalize_advantages(advantages)

    features = torch.stack([item.candidate_features for item in flat], dim=0)
    candidate_indices = torch.tensor([item.candidate_index for item in flat], dtype=torch.long)
    old_log_probs = torch.tensor([item.behavior_log_prob for item in flat], dtype=torch.float32)
    critic_values = torch.tensor([item.critic_value for item in flat], dtype=torch.float32)
    real_rewards = torch.tensor([item.real_response_reward for item in flat], dtype=torch.float32)
    decision_dt = torch.tensor([item.decision_dt for item in flat], dtype=torch.float32)
    dones = torch.tensor([item.done for item in flat], dtype=torch.bool)
    returns = torch.cat(return_parts, dim=0)

    tensors = (features, old_log_probs, advantages, returns, critic_values, real_rewards, decision_dt)
    if not all(torch.isfinite(item).all() for item in tensors):
        raise RuntimeError("PPO training batch contains non-finite values")

    return PPOTrainingBatch(
        candidate_features=features,
        candidate_indices=candidate_indices,
        old_log_probs=old_log_probs,
        advantages=advantages,
        returns=returns,
        critic_values=critic_values,
        real_response_rewards=real_rewards,
        decision_dt=decision_dt,
        dones=dones,
        provenance=tuple((item.episode_seed, item.agent_name, item.decision_index) for item in flat),
    )


class PPOTrainer:
    """Pure optimizer over an already-collected PPO batch.

    There is intentionally no LLM/provider/cache/environment dependency here. Repeated
    PPO epochs operate only on the frozen rollout batch, so optimizer epochs cannot
    trigger additional API calls.
    """

    def __init__(
        self,
        policy: CandidateActorCritic,
        *,
        training_config: PPOTrainingConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if not isinstance(policy, CandidateActorCritic):
            raise TypeError("policy must be CandidateActorCritic")
        self.training_config = training_config if training_config is not None else PPOTrainingConfig()
        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.training_config.learning_rate)
        self.update_index = 0

    def update(self, batch: PPOTrainingBatch) -> PPOUpdateMetrics:
        if not isinstance(batch, PPOTrainingBatch):
            raise TypeError("batch must be PPOTrainingBatch")
        n = batch.size
        if n <= 0:
            raise ValueError("PPO batch must be non-empty")

        features = batch.candidate_features.to(self.device)
        actions = batch.candidate_indices.to(self.device)
        old_log_probs = batch.old_log_probs.to(self.device)
        advantages = batch.advantages.to(self.device)
        returns = batch.returns.to(self.device)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.training_config.seed) + int(self.update_index))

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        kls: list[float] = []
        clip_fractions: list[float] = []
        grad_norms: list[float] = []
        optimizer_steps = 0

        self.policy.train()
        for _epoch in range(int(self.training_config.update_epochs)):
            order = torch.randperm(n, generator=generator)
            for start in range(0, n, int(self.training_config.minibatch_size)):
                indices_cpu = order[start : start + int(self.training_config.minibatch_size)]
                indices = indices_cpu.to(self.device)
                evaluation = self.policy.evaluate_actions(features[indices], actions[indices])
                losses = compute_ppo_losses(
                    evaluation.log_prob,
                    old_log_probs[indices],
                    advantages[indices],
                    evaluation.value,
                    returns[indices],
                    evaluation.entropy,
                    clip_epsilon=self.policy.config.clip_epsilon,
                    value_coef=self.policy.config.value_coef,
                    entropy_coef=self.policy.config.entropy_coef,
                )

                self.optimizer.zero_grad(set_to_none=True)
                losses.total_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    max_norm=float(self.training_config.max_grad_norm),
                )
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise RuntimeError("PPO gradient norm is non-finite")
                self.optimizer.step()

                policy_losses.append(float(losses.policy_loss.detach().cpu()))
                value_losses.append(float(losses.value_loss.detach().cpu()))
                entropies.append(float(losses.entropy.detach().cpu()))
                kls.append(float(losses.approx_kl.detach().cpu()))
                clip_fractions.append(float(losses.clip_fraction.detach().cpu()))
                grad_norms.append(float(torch.as_tensor(grad_norm).detach().cpu()))
                optimizer_steps += 1

        self.update_index += 1
        values = np.asarray(grad_norms, dtype=np.float64)
        return PPOUpdateMetrics(
            samples=n,
            optimizer_steps=int(optimizer_steps),
            policy_loss_mean=float(np.mean(policy_losses)),
            value_loss_mean=float(np.mean(value_losses)),
            entropy_mean=float(np.mean(entropies)),
            approx_kl_mean=float(np.mean(kls)),
            clip_fraction_mean=float(np.mean(clip_fractions)),
            grad_norm_mean=float(values.mean()),
            grad_norm_max=float(values.max()),
        )
