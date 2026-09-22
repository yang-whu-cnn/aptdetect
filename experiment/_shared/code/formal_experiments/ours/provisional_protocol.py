from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import json

import numpy as np
import torch

from formal_experiments.ours.ppo_training import PPOUpdateMetrics, RolloutStep
from shared.formal_state import BLUE_AGENTS


PROVISIONAL_STAGE_TARGETS = (2000, 5000, 10000, 20000)
DEFAULT_PROVISIONAL_STAGE_TARGET = PROVISIONAL_STAGE_TARGETS[0]
DEFAULT_PROVISIONAL_SCENARIO_STEPS = 100
MIN_PROVISIONAL_SCENARIO_STEPS = 5
FORMAL_ROLLOUT_TARGET = 128
MAX_PROVISIONAL_TRANSITIONS = PROVISIONAL_STAGE_TARGETS[-1]


@dataclass(frozen=True)
class UpdateRecord:
    policy_version_before: int
    policy_version_after: int
    batch_size: int
    rollout_target: int
    overshoot: int
    final_flush: bool
    cache_misses_before: int
    cache_misses_after: int
    live_api_calls_before: int
    live_api_calls_after: int
    metrics: PPOUpdateMetrics

    def to_json(self) -> dict:
        return {
            "policy_version_before": int(self.policy_version_before),
            "policy_version_after": int(self.policy_version_after),
            "batch_size": int(self.batch_size),
            "rollout_target": int(self.rollout_target),
            "overshoot": int(self.overshoot),
            "final_flush": bool(self.final_flush),
            "cache_misses_before": int(self.cache_misses_before),
            "cache_misses_after": int(self.cache_misses_after),
            "live_api_calls_before": int(self.live_api_calls_before),
            "live_api_calls_after": int(self.live_api_calls_after),
            "metrics": asdict(self.metrics),
        }


def validate_stage_target(value: int) -> int:
    target = int(value)
    if target not in PROVISIONAL_STAGE_TARGETS:
        raise ValueError(
            f"provisional stage target must be one of {PROVISIONAL_STAGE_TARGETS}; got {target}"
        )
    return target


def safe_episode_steps_for_remaining(
    remaining_transitions: int,
    *,
    default_steps: int = DEFAULT_PROVISIONAL_SCENARIO_STEPS,
    agent_count: int = len(BLUE_AGENTS),
    min_steps: int = MIN_PROVISIONAL_SCENARIO_STEPS,
) -> int | None:
    """Choose an episode length whose conservative transition upper bound fits budget.

    At most one decision transition per Blue agent can complete per global tick. Using
    agent_count * scenario_steps as the upper bound is deliberately conservative; it
    prevents the current stage from exceeding its frozen transition budget.
    """
    remaining = int(remaining_transitions)
    if remaining <= 0:
        return None
    if int(agent_count) <= 0 or int(default_steps) <= 0 or int(min_steps) <= 0:
        raise ValueError("episode scheduling parameters must be positive")
    max_steps_by_budget = remaining // int(agent_count)
    steps = min(int(default_steps), int(max_steps_by_budget))
    if steps < int(min_steps):
        return None
    return int(steps)


def safe_update_barrier(
    completed_batch_size: int,
    *,
    ppo_open_agents: Iterable[str],
    replay_open_agents: Iterable[str],
    rollout_target: int = FORMAL_ROLLOUT_TARGET,
    final_flush: bool = False,
) -> bool:
    batch_size = int(completed_batch_size)
    if batch_size < 0:
        raise ValueError("completed_batch_size must be >=0")
    ppo_open = tuple(ppo_open_agents)
    replay_open = tuple(replay_open_agents)
    if ppo_open or replay_open:
        return False
    if final_flush:
        return batch_size > 0
    return batch_size >= int(rollout_target)


def serialize_rollout_step(step: RolloutStep) -> dict:
    if not isinstance(step, RolloutStep):
        raise TypeError("step must be RolloutStep")
    return {
        "episode_seed": int(step.episode_seed),
        "agent_name": str(step.agent_name),
        "decision_index": int(step.decision_index),
        "candidate_features": step.candidate_features.detach().cpu(),
        "candidate_index": int(step.candidate_index),
        "behavior_log_prob": float(step.behavior_log_prob),
        "critic_value": float(step.critic_value),
        "critic_next_value": float(step.critic_next_value),
        "real_response_reward": float(step.real_response_reward),
        "decision_dt": int(step.decision_dt),
        "done": bool(step.done),
    }


def deserialize_rollout_step(payload: dict) -> RolloutStep:
    if not isinstance(payload, dict):
        raise TypeError("rollout-step payload must be dict")
    features = torch.as_tensor(payload["candidate_features"], dtype=torch.float32).detach().cpu().clone()
    if tuple(features.shape) != (6, 46) or not torch.isfinite(features).all():
        raise ValueError("serialized rollout candidate features violate [6,46]")
    agent = str(payload["agent_name"])
    if agent not in BLUE_AGENTS:
        raise ValueError("serialized rollout contains invalid Blue agent")
    return RolloutStep(
        episode_seed=int(payload["episode_seed"]),
        agent_name=agent,
        decision_index=int(payload["decision_index"]),
        candidate_features=features,
        candidate_index=int(payload["candidate_index"]),
        behavior_log_prob=float(payload["behavior_log_prob"]),
        critic_value=float(payload["critic_value"]),
        critic_next_value=float(payload["critic_next_value"]),
        real_response_reward=float(payload["real_response_reward"]),
        decision_dt=int(payload["decision_dt"]),
        done=bool(payload["done"]),
    )


def append_probe_records(path: str | Path, records: list[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    with destination.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()


def count_probe_records(path: str | Path) -> int:
    source = Path(path)
    if not source.exists():
        return 0
    return sum(1 for line in source.read_text(encoding="utf-8").splitlines() if line.strip())


def validate_probe_record(record: dict) -> None:
    required = {
        "model_alias",
        "policy_version",
        "episode_ordinal",
        "episode_seed",
        "agent_name",
        "decision_index",
        "global_tick_start",
        "global_tick_end",
        "state",
        "next_state",
        "requested_action_id",
        "canonical_action_id",
        "executed_action_family",
        "fallback",
        "action_completed",
        "selected_candidate_index",
        "selected_plan",
        "response_reward",
        "decision_dt",
        "done",
        "cache_key",
        "cache_hit",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"probe record missing fields: {sorted(missing)}")
    if str(record["agent_name"]) not in BLUE_AGENTS:
        raise ValueError("probe record contains invalid agent")
    for name in ("state", "next_state"):
        value = np.asarray(record[name], dtype=np.float32)
        if value.shape != (27,) or not np.isfinite(value).all():
            raise ValueError(f"probe {name} must be finite D27")
    requested = int(record["requested_action_id"])
    canonical = int(record["canonical_action_id"])
    if not 0 <= requested < 4 or not 0 <= canonical < 4:
        raise ValueError("probe action ID outside A4")
    plan = tuple(int(x) for x in record["selected_plan"])
    if len(plan) != 4 or any(x < 0 or x >= 4 for x in plan):
        raise ValueError("probe selected plan violates A4/H4")
    if int(record["selected_candidate_index"]) not in range(6):
        raise ValueError("probe candidate index outside K6")
    if requested != plan[0]:
        raise ValueError("probe requested action must equal selected plan[0]")
    if int(record["decision_dt"]) < 1:
        raise ValueError("probe decision_dt must be >=1")
    if not np.isfinite(float(record["response_reward"])):
        raise ValueError("probe response reward must be finite")
