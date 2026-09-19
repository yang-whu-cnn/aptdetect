"""Development-only PriorRL pilot wiring; never formal-result eligible."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from formal_experiments.data_collection.incident_response import IncidentResponseBookkeeper
from shared.formal_state import BLUE_AGENTS, FORMAL_STATE_DIM
from .policy import PriorRLActorCritic
from .training import A4RolloutBuffer, checkpoint_sha256

TRAIN_SEEDS = tuple(range(1000, 1032))
EVAL_SEEDS = tuple(range(3200, 3210))
POLICY_SEEDS = (61001, 61002)


@dataclass(frozen=True)
class DevelopmentPilotConfig:
    training_episode_budget: int
    policy_seeds: tuple[int, ...] = POLICY_SEEDS
    eval_episodes_per_policy: int = 5
    episode_ticks: int = 100
    formal_result_eligible: bool = False
    use_world_model: bool = False
    online_llm_allowed: bool = False

    def __post_init__(self):
        if not 1 <= self.training_episode_budget <= len(TRAIN_SEEDS):
            raise ValueError("development training budget must be in [1,32]")
        if self.policy_seeds != POLICY_SEEDS or self.eval_episodes_per_policy != 5:
            raise ValueError("development pilot requires 2 policy seeds x 5 eval episodes")
        if self.episode_ticks != 100 or self.formal_result_eligible:
            raise ValueError("development pilot is fixed at 100 ticks and ineligible")
        if self.use_world_model or self.online_llm_allowed:
            raise ValueError("PriorRL pilot forbids world models and online LLM")

    def train_seeds(self): return TRAIN_SEEDS[:self.training_episode_budget]
    def eval_seed_map(self):
        return {61001: EVAL_SEEDS[:5], 61002: EVAL_SEEDS[5:]}


class FormalRewardChannel:
    """The only hidden-truth ingress; policy/retriever never see these arguments."""
    def __init__(self, bookkeeper: IncidentResponseBookkeeper):
        if not isinstance(bookkeeper, IncidentResponseBookkeeper):
            raise TypeError("formal reward requires IncidentResponseBookkeeper")
        self.bookkeeper = bookkeeper; self._decision_reward = 0.0

    def record_hidden_tick(self, *, global_tick_end, red_presence_after,
                           local_work_failures=()):
        accounting = self.bookkeeper.record_tick(
            global_tick_end=global_tick_end, red_presence_after=red_presence_after,
            local_work_failures=local_work_failures)
        self._decision_reward += float(accounting.response_reward)

    def finish_decision(self) -> float:
        reward = self._decision_reward; self._decision_reward = 0.0
        return reward


def make_formal_reward_channel(*, episode_seed: int, agent_name: str,
                               initial_red_presence_by_host, global_tick: int = 0):
    bookkeeper = IncidentResponseBookkeeper(episode_seed=episode_seed, agent_name=agent_name)
    bookkeeper.reset(initial_red_presence_by_host=initial_red_presence_by_host,
                     global_tick=global_tick)
    return FormalRewardChannel(bookkeeper)


class DevelopmentCollector:
    """Observable D27/A4 collection boundary with frozen per-transition priors."""
    def __init__(self, policy: PriorRLActorCritic, retriever, buffer: A4RolloutBuffer):
        self.policy = policy; self.retriever = retriever; self.buffer = buffer

    def decide(self, observable_state, *, agent_name: str, action_mask=(True, True, True, True),
               deterministic=False):
        if agent_name not in BLUE_AGENTS:
            raise ValueError("invalid Blue agent")
        state = torch.as_tensor(observable_state, dtype=torch.float32)
        if state.shape != (FORMAL_STATE_DIM,) or not torch.isfinite(state).all():
            raise ValueError("policy input must be observable finite D27")
        # Retrieval occurs only at collection time. Its result is detached and frozen.
        prior = torch.as_tensor(self.retriever.lookup(state.numpy(), agent_name=agent_name),
                                dtype=torch.float32).detach().cpu().clone()
        mask = torch.as_tensor(action_mask, dtype=torch.bool)
        if mask.shape != (4,) or not bool(mask[0]):
            raise ValueError("availability mask must be A4 and keep Sleep legal")
        prior = torch.where(mask, prior, torch.zeros_like(prior))
        prior = prior + mask.to(prior.dtype) * torch.finfo(prior.dtype).eps
        prior /= prior.sum()
        with torch.no_grad():
            action, log_prob, value = self.policy.act(state, deterministic=deterministic,
                                                       action_mask=mask.to(next(self.policy.parameters()).device))
        return {"state": state.clone(), "agent_name": agent_name, "action": int(action.cpu()),
                "old_log_prob": float(log_prob.cpu()), "value": float(value.cpu()),
                "frozen_prior": prior, "action_mask": mask}

    def complete(self, decision: dict, *, next_observable_state, reward_channel: FormalRewardChannel,
                 done: bool, decision_dt: int, trajectory_id: str):
        next_state = torch.as_tensor(next_observable_state, dtype=torch.float32)
        if next_state.shape != (FORMAL_STATE_DIM,):
            raise ValueError("next policy state must be D27")
        with torch.no_grad():
            _, next_value = self.policy(next_state)
        self.buffer.add(state=decision["state"], action=decision["action"],
            old_log_prob=decision["old_log_prob"], value=decision["value"],
            reward=reward_channel.finish_decision(), next_value=0.0 if done else float(next_value.cpu()),
            done=done, decision_dt=decision_dt, frozen_prior=decision["frozen_prior"],
            action_mask=decision["action_mask"],
            trajectory_id=trajectory_id)


def development_manifest(config: DevelopmentPilotConfig, policies: dict[int, PriorRLActorCritic]) -> dict:
    if set(policies) != set(POLICY_SEEDS):
        raise ValueError("one independently initialized policy is required per policy seed")
    return {
        "schema": "priorrl_cc4_development_pilot_v1",
        "formal_result_eligible": False, "training_episode_budget": config.training_episode_budget,
        "train_seeds": list(config.train_seeds()),
        "eval_seed_map": {str(k): list(v) for k, v in config.eval_seed_map().items()},
        "eval_episodes_per_policy": 5, "episode_ticks": 100,
        "world_model_used": False, "online_llm_calls": 0,
        "checkpoint_sha256": {str(k): checkpoint_sha256(v) for k, v in sorted(policies.items())},
        "reward_channel": "IncidentResponseBookkeeper",
        "policy_input": "observable_D27_only", "prior_mode": "frozen_prototype_read_only",
    }


def run_development_pilot(config: DevelopmentPilotConfig, *, policy_factory,
                          retriever_factory, train_episode, evaluate_episode,
                          device: str | torch.device | None = None) -> dict:
    """Orchestrate the fixed development split through dependency-injected CC4 episodes.

    Episode callbacks own the concrete CybORG environment/action-adapter loop and must
    use ``DevelopmentCollector`` plus ``FormalRewardChannel``. This function fixes
    policy initialization, split membership, episode counts, and result eligibility.
    """
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    policies = {}; train_records = []; eval_records = []
    for policy_seed in config.policy_seeds:
        torch.manual_seed(policy_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(policy_seed)
        policy = policy_factory(policy_seed).to(run_device)
        if not isinstance(policy, PriorRLActorCritic):
            raise TypeError("policy_factory must return PriorRLActorCritic")
        retriever = retriever_factory(policy_seed)
        for episode_seed in config.train_seeds():
            train_records.append(train_episode(
                episode_seed=episode_seed, policy_seed=policy_seed, policy=policy,
                retriever=retriever, episode_ticks=config.episode_ticks,
                reward_channel_factory=make_formal_reward_channel))
        policies[policy_seed] = policy
        policy.eval()
        with torch.no_grad():
            for episode_seed in config.eval_seed_map()[policy_seed]:
                eval_records.append(evaluate_episode(
                    episode_seed=episode_seed, policy_seed=policy_seed, policy=policy,
                    retriever=retriever, episode_ticks=config.episode_ticks,
                    reward_channel_factory=make_formal_reward_channel))
    manifest = development_manifest(config, policies)
    manifest.update({"device": str(run_device), "training_records": train_records,
                     "evaluation_records": eval_records})
    return manifest
