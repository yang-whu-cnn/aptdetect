from __future__ import annotations

import json

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from shared.action_contract import get_action
from shared.cyborg_action_adapter import CybORGActionResolution
from shared.formal_state import (
    BLUE_AGENTS,
    FORMAL_STATE_DIM,
)


EXECUTED_DURATION = {
    "Sleep": 1,
    "Analyse": 2,
    "Remove": 3,
    "Restore": 5,
}


def _state_array(
    value: Sequence[float] | np.ndarray,
) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=np.float32,
    )

    if array.shape != (
        FORMAL_STATE_DIM,
    ):
        raise ValueError(
            "state must have shape "
            f"({FORMAL_STATE_DIM},); "
            f"got {array.shape}"
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            "state contains "
            "non-finite values"
        )

    return array.copy()


def _nonnegative_int(
    name: str,
    value: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be "
            "a non-negative int"
        )

    return int(value)


def _finite_float(
    name: str,
    value: float,
) -> float:
    value = float(value)

    if not isfinite(value):
        raise ValueError(
            f"{name} must be finite"
        )

    return value


@dataclass(frozen=True)
class DecisionEpochTransition:
    """
    一条正式 decision-epoch transition。

    world model 的基本学习对象：

        state
        requested_action_id
        -> next_state

    同时保存 executed action，
    用于 fallback / duration / audit。

    A4.3 会正式冻结 response reward
    与 incident bookkeeping 的生成逻辑。
    """

    episode_seed: int
    agent_name: str
    decision_index: int

    global_tick_start: int
    global_tick_end: int
    decision_dt: int

    state: np.ndarray
    next_state: np.ndarray

    requested_action_id: int
    requested_action_name: str
    requested_cyborg_family: str
    requested_duration_ticks: int

    executed_index: int
    executed_label: str
    executed_action_family: str
    executed_duration_ticks: int
    target_host: Optional[str]

    fallback: bool
    fallback_reason: Optional[str]

    action_completed: bool
    completed_action_success: Optional[bool]

    # Interval-level bookkeeping.
    #
    # A4.3 会负责真实计算这些量；
    # A4.2 先冻结 replay contract。
    incident_active_ticks: int
    incident_host_lwf_count: int
    incident_host_lwf_raw_penalty: float
    response_reward: float

    official_reward: float

    done: bool

    def to_jsonable(
        self,
    ) -> dict:
        return {
            "episode_seed":
                int(self.episode_seed),

            "agent_name":
                self.agent_name,

            "decision_index":
                int(self.decision_index),

            "global_tick_start":
                int(self.global_tick_start),

            "global_tick_end":
                int(self.global_tick_end),

            "decision_dt":
                int(self.decision_dt),

            "state":
                self.state.tolist(),

            "next_state":
                self.next_state.tolist(),

            "requested_action_id":
                int(
                    self.requested_action_id
                ),

            "requested_action_name":
                self.requested_action_name,

            "requested_cyborg_family":
                self.requested_cyborg_family,

            "requested_duration_ticks":
                int(
                    self.requested_duration_ticks
                ),

            "executed_index":
                int(self.executed_index),

            "executed_label":
                self.executed_label,

            "executed_action_family":
                self.executed_action_family,

            "executed_duration_ticks":
                int(
                    self.executed_duration_ticks
                ),

            "target_host":
                self.target_host,

            "fallback":
                bool(self.fallback),

            "fallback_reason":
                self.fallback_reason,

            "action_completed":
                bool(
                    self.action_completed
                ),

            "completed_action_success":
                self.completed_action_success,

            "incident_active_ticks":
                int(
                    self.incident_active_ticks
                ),

            "incident_host_lwf_count":
                int(
                    self
                    .incident_host_lwf_count
                ),

            "incident_host_lwf_raw_penalty":
                float(
                    self
                    .incident_host_lwf_raw_penalty
                ),

            "response_reward":
                float(
                    self.response_reward
                ),

            "official_reward":
                float(
                    self.official_reward
                ),

            "done":
                bool(self.done),
        }


@dataclass
class _OpenDecision:
    episode_seed: int
    agent_name: str
    decision_index: int

    global_tick_start: int
    last_accounted_tick: int

    state: np.ndarray

    resolution: CybORGActionResolution

    requested_duration_ticks: int
    executed_duration_ticks: int

    incident_active_ticks: int = 0
    incident_host_lwf_count: int = 0
    incident_host_lwf_raw_penalty: float = 0.0

    response_reward: float = 0.0
    official_reward: float = 0.0


class DecisionReplayBuffer:
    def __init__(
        self,
    ):
        self._items: list[
            DecisionEpochTransition
        ] = []

    def append(
        self,
        transition: DecisionEpochTransition,
    ) -> None:
        if not isinstance(
            transition,
            DecisionEpochTransition,
        ):
            raise TypeError(
                "transition must be "
                "DecisionEpochTransition"
            )

        self._items.append(
            transition
        )

    def __len__(
        self,
    ) -> int:
        return len(
            self._items
        )

    @property
    def transitions(
        self,
    ) -> tuple[
        DecisionEpochTransition,
        ...,
    ]:
        return tuple(
            self._items
        )

    def to_jsonable(
        self,
    ) -> list[dict]:
        return [
            item.to_jsonable()
            for item
            in self._items
        ]

    def save_jsonl(
        self,
        path: str | Path,
    ) -> None:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            for transition in (
                self._items
            ):
                handle.write(
                    json.dumps(
                        transition
                        .to_jsonable(),
                        ensure_ascii=False,
                    )
                )
                handle.write("\n")


class DecisionEpochReplayCollector:
    """
    Multi-agent asynchronous
    decision-epoch replay collector。

    对每个 Blue agent 独立维护
    一个正在执行的 decision。

    典型调用：

        begin_decision(...)
        env.step(...)
        record_tick(...)
        ...
        end_decision(...)

    busy agent 不重新 begin_decision。
    """

    def __init__(
        self,
        episode_seed: int,
    ):
        if (
            isinstance(
                episode_seed,
                bool,
            )
            or not isinstance(
                episode_seed,
                int,
            )
        ):
            raise ValueError(
                "episode_seed must "
                "be an int"
            )

        self.episode_seed = int(
            episode_seed
        )

        self.buffer = (
            DecisionReplayBuffer()
        )

        self._open: dict[
            str,
            _OpenDecision,
        ] = {}

        self._next_decision_index = {
            agent: 0
            for agent in BLUE_AGENTS
        }

    @property
    def open_agents(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                self._open.keys()
            )
        )

    def begin_decision(
        self,
        *,
        agent_name: str,
        decision_index: int,
        global_tick_start: int,
        state: (
            Sequence[float]
            | np.ndarray
        ),
        resolution: (
            CybORGActionResolution
        ),
    ) -> None:
        agent_name = str(
            agent_name
        )

        if agent_name not in BLUE_AGENTS:
            raise ValueError(
                "unsupported agent: "
                f"{agent_name}"
            )

        if agent_name in self._open:
            raise RuntimeError(
                f"{agent_name}: "
                "decision already open"
            )

        decision_index = (
            _nonnegative_int(
                "decision_index",
                decision_index,
            )
        )

        expected_index = (
            self._next_decision_index[
                agent_name
            ]
        )

        if (
            decision_index
            != expected_index
        ):
            raise ValueError(
                f"{agent_name}: "
                f"decision_index="
                f"{decision_index}, "
                f"expected="
                f"{expected_index}"
            )

        global_tick_start = (
            _nonnegative_int(
                "global_tick_start",
                global_tick_start,
            )
        )

        if not isinstance(
            resolution,
            CybORGActionResolution,
        ):
            raise TypeError(
                "resolution must be "
                "CybORGActionResolution"
            )

        if (
            resolution.agent_name
            != agent_name
        ):
            raise ValueError(
                "resolution.agent_name "
                "does not match"
            )

        requested = get_action(
            resolution
            .requested_action_id
        )

        if (
            requested.name
            != resolution
            .requested_action_name
        ):
            raise ValueError(
                "requested action name "
                "does not match contract"
            )

        if (
            requested.cyborg_action
            != resolution
            .requested_cyborg_family
        ):
            raise ValueError(
                "requested action family "
                "does not match contract"
            )

        executed_family = (
            resolution
            .executed_action_family
        )

        if (
            executed_family
            not in EXECUTED_DURATION
        ):
            raise ValueError(
                "unsupported executed "
                f"family: {executed_family}"
            )

        self._open[
            agent_name
        ] = _OpenDecision(
            episode_seed=(
                self.episode_seed
            ),

            agent_name=agent_name,

            decision_index=(
                decision_index
            ),

            global_tick_start=(
                global_tick_start
            ),

            last_accounted_tick=(
                global_tick_start
            ),

            state=_state_array(
                state
            ),

            resolution=resolution,

            requested_duration_ticks=(
                requested
                .duration_ticks
            ),

            executed_duration_ticks=(
                EXECUTED_DURATION[
                    executed_family
                ]
            ),
        )

    def record_tick(
        self,
        *,
        agent_name: str,
        global_tick_end: int,
        official_reward: float = 0.0,
        incident_active_ticks: int = 0,
        incident_host_lwf_count: int = 0,
        incident_host_lwf_raw_penalty: float = 0.0,
        response_reward: float = 0.0,
    ) -> None:
        """
        记录从上一个 accounted tick
        到 global_tick_end 的 interval。

        正式 CC4 runner 中通常每个
        global tick 调用一次。

        incident_active_ticks 可在未来
        表示多个并发 incident 的
        aggregate incident-ticks，
        因此不强制 <= elapsed ticks。
        """

        agent_name = str(
            agent_name
        )

        if agent_name not in self._open:
            raise RuntimeError(
                f"{agent_name}: "
                "no open decision"
            )

        item = self._open[
            agent_name
        ]

        global_tick_end = (
            _nonnegative_int(
                "global_tick_end",
                global_tick_end,
            )
        )

        if (
            global_tick_end
            <= item.last_accounted_tick
        ):
            raise ValueError(
                "global_tick_end must "
                "advance time"
            )

        incident_active_ticks = (
            _nonnegative_int(
                "incident_active_ticks",
                incident_active_ticks,
            )
        )

        incident_host_lwf_count = (
            _nonnegative_int(
                "incident_host_lwf_count",
                incident_host_lwf_count,
            )
        )

        lwf_penalty = (
            _finite_float(
                "incident_host_lwf_raw_penalty",
                incident_host_lwf_raw_penalty,
            )
        )

        # CC4 LWF raw reward contribution
        # is zero or negative.
        if lwf_penalty > 0.0:
            raise ValueError(
                "incident_host_lwf_raw_penalty "
                "must be <= 0"
            )

        item.incident_active_ticks += (
            incident_active_ticks
        )

        item.incident_host_lwf_count += (
            incident_host_lwf_count
        )

        item.incident_host_lwf_raw_penalty += (
            lwf_penalty
        )

        item.response_reward += (
            _finite_float(
                "response_reward",
                response_reward,
            )
        )

        item.official_reward += (
            _finite_float(
                "official_reward",
                official_reward,
            )
        )

        item.last_accounted_tick = (
            global_tick_end
        )

    def end_decision(
        self,
        *,
        agent_name: str,
        global_tick_end: int,
        next_state: (
            Sequence[float]
            | np.ndarray
        ),
        done: bool,
        completed_action_success: (
            Optional[bool]
        ) = None,
    ) -> DecisionEpochTransition:

        agent_name = str(
            agent_name
        )

        if agent_name not in self._open:
            raise RuntimeError(
                f"{agent_name}: "
                "no open decision"
            )

        if not isinstance(
            done,
            bool,
        ):
            raise ValueError(
                "done must be bool"
            )

        if (
            completed_action_success
            is not None
            and not isinstance(
                completed_action_success,
                bool,
            )
        ):
            raise ValueError(
                "completed_action_success "
                "must be bool or None"
            )

        global_tick_end = (
            _nonnegative_int(
                "global_tick_end",
                global_tick_end,
            )
        )

        item = self._open[
            agent_name
        ]

        if (
            item.last_accounted_tick
            != global_tick_end
        ):
            raise RuntimeError(
                f"{agent_name}: final tick "
                "has not been accounted"
            )

        decision_dt = (
            global_tick_end
            - item.global_tick_start
        )

        if decision_dt <= 0:
            raise RuntimeError(
                "decision_dt must be > 0"
            )

        duration = (
            item.executed_duration_ticks
        )

        if decision_dt > duration:
            raise RuntimeError(
                f"{agent_name}: "
                f"decision_dt={decision_dt} "
                f"> executed duration="
                f"{duration}"
            )

        action_completed = (
            decision_dt == duration
        )

        # 未终止 episode 时，
        # 只能在真正下一个 decision epoch
        # 关闭 transition。
        if (
            not done
            and not action_completed
        ):
            raise RuntimeError(
                f"{agent_name}: "
                "cannot close busy action "
                "before next decision epoch"
            )

        # 只有 episode 提前结束时，
        # 才允许 dt < duration。
        if (
            not action_completed
            and not done
        ):
            raise RuntimeError(
                "incomplete action requires "
                "done=True"
            )

        if (
            not action_completed
            and completed_action_success
            is not None
        ):
            raise ValueError(
                "unfinished action cannot "
                "have completion success"
            )

        r = item.resolution

        transition = (
            DecisionEpochTransition(
                episode_seed=(
                    item.episode_seed
                ),

                agent_name=(
                    item.agent_name
                ),

                decision_index=(
                    item.decision_index
                ),

                global_tick_start=(
                    item
                    .global_tick_start
                ),

                global_tick_end=(
                    global_tick_end
                ),

                decision_dt=int(
                    decision_dt
                ),

                state=(
                    item.state.copy()
                ),

                next_state=(
                    _state_array(
                        next_state
                    )
                ),

                requested_action_id=(
                    int(
                        r
                        .requested_action_id
                    )
                ),

                requested_action_name=(
                    r
                    .requested_action_name
                ),

                requested_cyborg_family=(
                    r
                    .requested_cyborg_family
                ),

                requested_duration_ticks=(
                    int(
                        item
                        .requested_duration_ticks
                    )
                ),

                executed_index=(
                    int(
                        r.executed_index
                    )
                ),

                executed_label=(
                    r.executed_label
                ),

                executed_action_family=(
                    r
                    .executed_action_family
                ),

                executed_duration_ticks=(
                    int(
                        item
                        .executed_duration_ticks
                    )
                ),

                target_host=(
                    r.target_host
                ),

                fallback=(
                    bool(r.fallback)
                ),

                fallback_reason=(
                    r.fallback_reason
                ),

                action_completed=(
                    bool(
                        action_completed
                    )
                ),

                completed_action_success=(
                    completed_action_success
                ),

                incident_active_ticks=(
                    int(
                        item
                        .incident_active_ticks
                    )
                ),

                incident_host_lwf_count=(
                    int(
                        item
                        .incident_host_lwf_count
                    )
                ),

                incident_host_lwf_raw_penalty=(
                    float(
                        item
                        .incident_host_lwf_raw_penalty
                    )
                ),

                response_reward=(
                    float(
                        item
                        .response_reward
                    )
                ),

                official_reward=(
                    float(
                        item
                        .official_reward
                    )
                ),

                done=bool(done),
            )
        )

        self.buffer.append(
            transition
        )

        del self._open[
            agent_name
        ]

        self._next_decision_index[
            agent_name
        ] += 1

        return transition