"""Common policy boundary for every formal-v3 method."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from random import Random
from typing import Any, Mapping, Protocol, runtime_checkable

from shared.action_contract import N_ACTIONS


@dataclass(frozen=True)
class ActionDecision:
    method: str
    requested_action: int
    executed_action: int
    target: str | None
    fallback: bool
    fallback_reason: str | None
    decision_latency_sec: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.method.strip():
            raise ValueError("method must be non-empty")
        for name, value in (("requested_action", self.requested_action), ("executed_action", self.executed_action)):
            if isinstance(value, bool) or not isinstance(value, int) or value not in range(N_ACTIONS):
                raise ValueError(f"{name} must be an A4 integer")
        if self.decision_latency_sec < 0:
            raise ValueError("decision_latency_sec must be >= 0")
        if self.fallback and not self.fallback_reason:
            raise ValueError("fallback requires fallback_reason")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class FormalAgent(Protocol):
    def reset(self, *, episode_seed: int, agent_id: str) -> None: ...
    def act(self, *, blue_observation: Mapping[str, Any], tick: int, action_mask: tuple[bool, ...]) -> ActionDecision: ...
    def observe(self, *, transition: Mapping[str, Any]) -> None: ...
    def close(self) -> None: ...


class FixedActionAgent:
    def __init__(self, action_id: int = 0) -> None:
        if action_id not in range(N_ACTIONS):
            raise ValueError("action_id outside A4")
        self.action_id = int(action_id)

    def reset(self, *, episode_seed: int, agent_id: str) -> None:
        self.agent_id = str(agent_id)

    def act(self, *, blue_observation: Mapping[str, Any], tick: int, action_mask: tuple[bool, ...]) -> ActionDecision:
        requested = self.action_id
        available = len(action_mask) == N_ACTIONS and bool(action_mask[requested])
        executed = requested if available else 0
        return ActionDecision("fixed", requested, executed, None, not available, None if available else "masked", 0.0)

    def observe(self, *, transition: Mapping[str, Any]) -> None: pass
    def close(self) -> None: pass


class RandomActionAgent(FixedActionAgent):
    def reset(self, *, episode_seed: int, agent_id: str) -> None:
        super().reset(episode_seed=episode_seed, agent_id=agent_id)
        self._random = Random(f"{int(episode_seed)}:{agent_id}")

    def act(self, *, blue_observation: Mapping[str, Any], tick: int, action_mask: tuple[bool, ...]) -> ActionDecision:
        if len(action_mask) != N_ACTIONS:
            raise ValueError("action_mask must have four entries")
        choices = [index for index, available in enumerate(action_mask) if available]
        if not choices:
            choices = [0]
        action = self._random.choice(choices)
        return ActionDecision("random", action, action, None, False, None, 0.0)
