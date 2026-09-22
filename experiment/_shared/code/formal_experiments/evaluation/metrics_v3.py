"""Frozen CC4 v3 evaluation metrics.

The functions in this module consume auditable episode events.  They deliberately
do not consume policy training rewards or pre-computed metric fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EPISODE_TICKS = 500
RECOVERY_ACTIONS = frozenset({"remove", "restore"})


@dataclass(frozen=True)
class EpisodeMetrics:
    cc4_official_reward: float
    operation_failure_penalty: float
    operation_failure_signed_sum: float
    operation_failure_count: int
    recovery_true_positives: int
    recovery_false_positives: int
    recovery_precision: float | None
    recovery_time_censored_mean: float | None
    recovery_time_completed_only_mean: float | None
    recovery_incident_count: int
    recovery_completed_count: int
    recovery_unrecovered_count: int
    recovery_unrecovered_rate: float | None
    recovery_time_censored_sum: float
    recovery_time_completed_sum: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RepeatMetrics:
    episode_count: int
    cc4_official_reward: float
    operation_failure_penalty: float
    operation_failure_signed_sum: float
    operation_failure_count: int
    recovery_true_positives: int
    recovery_false_positives: int
    recovery_precision: float | None
    recovery_time_censored_mean: float | None
    recovery_time_completed_only_mean: float | None
    recovery_incident_count: int
    recovery_completed_count: int
    recovery_unrecovered_count: int
    recovery_unrecovered_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def team_reward_once(value: Any) -> float:
    """Return one team reward, rejecting disagreeing per-agent duplicates."""
    if isinstance(value, Mapping):
        rewards = list(value.values())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rewards = list(value)
    else:
        return _finite_number(value, "team reward")
    if not rewards:
        raise ValueError("team reward duplicate collection is empty")
    numbers = [_finite_number(item, "team reward") for item in rewards]
    if not all(number == numbers[0] for number in numbers[1:]):
        raise ValueError("per-agent rewards disagree; cannot identify one team reward")
    return numbers[0]


def compute_episode_metrics(
    episode: Mapping[str, Any], *, expected_ticks: int = EPISODE_TICKS
) -> EpisodeMetrics:
    if isinstance(expected_ticks, bool) or not isinstance(expected_ticks, int) or expected_ticks <= 0:
        raise ValueError("expected_ticks must be a positive integer")
    tick_count = int(episode.get("tick_count", -1))
    end_tick = int(episode.get("episode_end_tick", tick_count))
    if tick_count != expected_ticks or end_tick != expected_ticks:
        raise ValueError(f"episode must complete exactly {expected_ticks} ticks")

    rewards = episode.get("tick_team_rewards")
    if not isinstance(rewards, list) or len(rewards) != expected_ticks:
        raise ValueError(f"tick_team_rewards must contain exactly {expected_ticks} tick records")
    official = float(sum(team_reward_once(value) for value in rewards))

    failures = episode.get("operation_failure_events", [])
    if not isinstance(failures, list):
        raise ValueError("operation_failure_events must be a list")
    signed_failure = 0.0
    failure_count = 0
    for index, event in enumerate(failures):
        if not isinstance(event, Mapping):
            raise ValueError("operation failure event must be an object")
        located = bool(str(event.get("agent_name", "")).strip()) and bool(
            str(event.get("host", "")).strip()
        )
        if not located and event.get("scope") != "all_blue":
            raise ValueError(
                f"failure[{index}] needs agent_name+host or scope=all_blue for coverage audit"
            )
        raw_penalty = _finite_number(event.get("raw_penalty"), f"failure[{index}].raw_penalty")
        signed_failure += min(raw_penalty, 0.0)
        failure_count += int(raw_penalty < 0.0)

    tp = fp = 0
    actions = episode.get("recovery_actions", [])
    if not isinstance(actions, list):
        raise ValueError("recovery_actions must be a list")
    for action in actions:
        if not isinstance(action, Mapping):
            raise ValueError("recovery action must be an object")
        family = str(action.get("executed_action", "")).lower()
        # Only actions that really started and completed are interventions.
        if family not in RECOVERY_ACTIONS or not action.get("started") or not action.get("completed"):
            continue
        if action.get("fallback", False):
            raise ValueError("a fallback action cannot be a completed recovery intervention")
        if not isinstance(action.get("active_incident_before"), bool):
            raise ValueError("completed recovery action needs boolean active_incident_before")
        if action["active_incident_before"]:
            tp += 1
        else:
            fp += 1

    censored_times: list[float] = []
    completed_times: list[float] = []
    incidents = episode.get("incidents", [])
    if not isinstance(incidents, list):
        raise ValueError("incidents must be a list")
    for index, incident in enumerate(incidents):
        if not isinstance(incident, Mapping):
            raise ValueError("incident must be an object")
        compromised = int(incident.get("t_compromise", -1))
        # Tick timestamps denote transition boundaries.  An incident first seen
        # after the final transition is stamped exactly episode_end_tick and has
        # a valid censor duration of zero.
        if not 0 <= compromised <= end_tick:
            raise ValueError(f"incident[{index}].t_compromise outside episode")
        recovered_raw = incident.get("t_recovered")
        if recovered_raw is None:
            censored_times.append(float(end_tick - compromised))
        else:
            recovered = int(recovered_raw)
            if not compromised <= recovered <= end_tick:
                raise ValueError(f"incident[{index}].t_recovered outside incident interval")
            elapsed = float(recovered - compromised)
            censored_times.append(elapsed)
            completed_times.append(elapsed)

    denominator = tp + fp
    incident_count = len(censored_times)
    completed_count = len(completed_times)
    unrecovered = incident_count - completed_count
    return EpisodeMetrics(
        cc4_official_reward=official,
        operation_failure_penalty=-signed_failure,
        operation_failure_signed_sum=signed_failure,
        operation_failure_count=failure_count,
        recovery_true_positives=tp,
        recovery_false_positives=fp,
        recovery_precision=(tp / denominator if denominator else None),
        recovery_time_censored_mean=(float(np.mean(censored_times)) if censored_times else None),
        recovery_time_completed_only_mean=(
            float(np.mean(completed_times)) if completed_times else None
        ),
        recovery_incident_count=incident_count,
        recovery_completed_count=completed_count,
        recovery_unrecovered_count=unrecovered,
        recovery_unrecovered_rate=(unrecovered / incident_count if incident_count else None),
        recovery_time_censored_sum=float(sum(censored_times)),
        recovery_time_completed_sum=float(sum(completed_times)),
    )


def aggregate_repeat(
    episodes: Iterable[Mapping[str, Any]], *, expected_ticks: int = EPISODE_TICKS
) -> RepeatMetrics:
    metrics = [compute_episode_metrics(episode, expected_ticks=expected_ticks) for episode in episodes]
    if not metrics:
        raise ValueError("repeat contains no episodes")
    tp = sum(item.recovery_true_positives for item in metrics)
    fp = sum(item.recovery_false_positives for item in metrics)
    incidents = sum(item.recovery_incident_count for item in metrics)
    completed = sum(item.recovery_completed_count for item in metrics)
    unrecovered = incidents - completed
    return RepeatMetrics(
        episode_count=len(metrics),
        cc4_official_reward=float(np.mean([item.cc4_official_reward for item in metrics])),
        operation_failure_penalty=float(
            np.mean([item.operation_failure_penalty for item in metrics])
        ),
        operation_failure_signed_sum=float(
            np.mean([item.operation_failure_signed_sum for item in metrics])
        ),
        operation_failure_count=sum(item.operation_failure_count for item in metrics),
        recovery_true_positives=tp,
        recovery_false_positives=fp,
        recovery_precision=(tp / (tp + fp) if tp + fp else None),
        recovery_time_censored_mean=(
            sum(item.recovery_time_censored_sum for item in metrics) / incidents
            if incidents
            else None
        ),
        recovery_time_completed_only_mean=(
            sum(item.recovery_time_completed_sum for item in metrics) / completed
            if completed
            else None
        ),
        recovery_incident_count=incidents,
        recovery_completed_count=completed,
        recovery_unrecovered_count=unrecovered,
        recovery_unrecovered_rate=(unrecovered / incidents if incidents else None),
    )
