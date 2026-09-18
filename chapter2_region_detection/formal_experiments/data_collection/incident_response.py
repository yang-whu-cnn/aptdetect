from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Mapping, Sequence

from shared.formal_state import BLUE_AGENTS


RESPONSE_REWARD_PROTOCOL = "final_paper_20260917_v1"


# ============================================================
# Validation
# ============================================================

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
            f"{name} must be a non-negative int"
        )

    return int(value)


def _finite_nonnegative(
    name: str,
    value: float,
) -> float:
    value = float(value)

    if (
        not isfinite(value)
        or value < 0.0
    ):
        raise ValueError(
            f"{name} must be finite and >= 0"
        )

    return value


def _presence_mapping(
    value: Mapping[str, bool],
) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise TypeError(
            "red_presence must be a mapping"
        )

    out: dict[str, bool] = {}

    for raw_host, raw_present in value.items():
        hostname = str(raw_host).strip()

        if not hostname:
            raise ValueError(
                "hostname must not be empty"
            )

        if not isinstance(
            raw_present,
            bool,
        ):
            raise ValueError(
                "red presence must be bool: "
                f"{hostname!r} -> "
                f"{raw_present!r}"
            )

        out[hostname] = raw_present

    if not out:
        raise ValueError(
            "red_presence must contain "
            "at least one tracked host"
        )

    return out


# ============================================================
# Reward config
# ============================================================

@dataclass(frozen=True)
class ResponseRewardConfig:
    """
    论文正式 response objective。

    lambda_time:
        compromise-age delay 权重。

    lambda_failure:
        agent 管辖范围内 normal-operation failure 权重。

    CC4 LWF raw penalty 本身 <= 0，
    因此 reward 中直接：

        lambda_failure * raw_lwf_penalty

    即得到负奖励。
    """

    lambda_time: float = 1.0
    lambda_failure: float = 1.0

    def __post_init__(
        self,
    ) -> None:
        _finite_nonnegative(
            "lambda_time",
            self.lambda_time,
        )

        _finite_nonnegative(
            "lambda_failure",
            self.lambda_failure,
        )

    def compute(
        self,
        *,
        incident_delay_penalty: float,
        incident_host_lwf_raw_penalty: float,
    ) -> float:
        delay_penalty = _finite_nonnegative(
            "incident_delay_penalty",
            incident_delay_penalty,
        )

        raw_penalty = float(
            incident_host_lwf_raw_penalty
        )

        if not isfinite(
            raw_penalty
        ):
            raise ValueError(
                "incident_host_lwf_raw_penalty "
                "must be finite"
            )

        if raw_penalty > 0.0:
            raise ValueError(
                "CC4 raw LWF penalty "
                "must be <= 0"
            )

        return float(
            -self.lambda_time
            * delay_penalty
            +
            self.lambda_failure
            * raw_penalty
        )


# ============================================================
# Local work event
# ============================================================

@dataclass(frozen=True)
class LocalWorkFailure:
    """
    一个已经确认失败的 GreenLocalWork。

    raw_lwf_penalty 必须使用 CC4 当前
    mission phase / subnet 对应的 LWF 值。

    本类不接受 ASF / RIA / team reward。
    """

    hostname: str
    raw_lwf_penalty: float

    def __post_init__(
        self,
    ) -> None:
        hostname = str(
            self.hostname
        ).strip()

        if not hostname:
            raise ValueError(
                "hostname must not be empty"
            )

        raw = float(
            self.raw_lwf_penalty
        )

        if (
            not isfinite(raw)
            or raw > 0.0
        ):
            raise ValueError(
                "raw_lwf_penalty must "
                "be finite and <= 0"
            )


# ============================================================
# Incident event
# ============================================================

@dataclass
class IncidentEvent:
    """
    一个 host-level compromise event。

    HIDDEN TRUTH bookkeeping only。

    绝不能进入：
        FormalStateEncoder
        LLM input
        PPO input
        CEM / UG-CEM planner state
    """

    incident_event_id: str
    agent_name: str
    host_id: str
    ordinal: int

    t_compromise: int
    t_normal: int | None = None

    incident_host_lwf_count: int = 0
    incident_host_lwf_raw_penalty: float = 0.0

    @property
    def active(
        self,
    ) -> bool:
        return self.t_normal is None

    @property
    def attack_eradication_time(
        self,
    ) -> int | None:
        if self.t_normal is None:
            return None

        return int(
            self.t_normal
            - self.t_compromise
        )

    def to_jsonable(
        self,
    ) -> dict:
        return {
            "incident_event_id":
                self.incident_event_id,

            "agent_name":
                self.agent_name,

            "host_id":
                self.host_id,

            "ordinal":
                int(self.ordinal),

            "t_compromise":
                int(self.t_compromise),

            "t_normal":
                (
                    None
                    if self.t_normal is None
                    else int(self.t_normal)
                ),

            "attack_eradication_time":
                self.attack_eradication_time,

            "incident_host_lwf_count":
                int(
                    self.incident_host_lwf_count
                ),

            "incident_host_lwf_raw_penalty":
                float(
                    self
                    .incident_host_lwf_raw_penalty
                ),
        }


# ============================================================
# One CC4 tick accounting
# ============================================================

@dataclass(frozen=True)
class IncidentTickAccounting:
    """
    一个真实 CC4 global tick 的
    incident-specific accounting。

    incident_event_ids / host_ids:
        当前 interval 相关的 events，
        包括：
        - interval 开始时 active；
        - 本 tick 新开始；
        - 本 tick 恢复。

    active_event_ids:
        真正贡献 eradication-time reward
        的事件。
    """

    global_tick_start: int
    global_tick_end: int

    incident_event_ids: tuple[str, ...]
    incident_host_ids: tuple[str, ...]

    active_event_ids: tuple[str, ...]
    active_host_ids: tuple[str, ...]

    started_event_ids: tuple[str, ...]
    recovered_event_ids: tuple[str, ...]

    counted_lwf_event_ids: tuple[str, ...]
    counted_lwf_host_ids: tuple[str, ...]

    incident_active_ticks: int
    incident_delay_penalty: float

    incident_host_lwf_count: int
    incident_host_lwf_raw_penalty: float

    response_reward: float

    def to_replay_kwargs(
        self,
    ) -> dict:
        """
        直接传给：

            DecisionEpochReplayCollector.record_tick(...)
        """

        return {
            "incident_event_ids":
                self.incident_event_ids,

            "incident_host_ids":
                self.incident_host_ids,

            "incident_active_ticks":
                self.incident_active_ticks,

            "incident_delay_penalty":
                self.incident_delay_penalty,

            "incident_host_lwf_count":
                self.incident_host_lwf_count,

            "incident_host_lwf_raw_penalty":
                self
                .incident_host_lwf_raw_penalty,

            "response_reward":
                self.response_reward,
        }


# ============================================================
# Formal incident bookkeeper
# ============================================================

class IncidentResponseBookkeeper:
    """
    每个 Blue agent / region 一个实例。

    tracked hosts 应来自该 agent 的
    observable reset inventory。

    然后 reward/evaluation side
    使用 hidden truth 判断这些 host
    是否存在 active Red session。

    因此：

        host scope
            <- observable inventory

        compromise truth
            <- hidden bookkeeping only

    避免 hidden truth 参与 planner state。
    """

    def __init__(
        self,
        *,
        episode_seed: int,
        agent_name: str,
        reward_config: (
            ResponseRewardConfig | None
        ) = None,
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
                "episode_seed must be int"
            )

        agent_name = str(
            agent_name
        )

        if agent_name not in BLUE_AGENTS:
            raise ValueError(
                "unsupported Blue agent: "
                f"{agent_name!r}"
            )

        self.episode_seed = int(
            episode_seed
        )

        self.agent_name = (
            agent_name
        )

        self.reward_config = (
            reward_config
            if reward_config is not None
            else ResponseRewardConfig()
        )

        self._presence: dict[
            str,
            bool,
        ] = {}

        self._active_by_host: dict[
            str,
            IncidentEvent,
        ] = {}

        self._host_ordinals: dict[
            str,
            int,
        ] = {}

        self._events: list[
            IncidentEvent
        ] = []

        self._last_global_tick = 0
        self._reset_done = False

        self.total_incident_active_ticks = 0
        self.total_incident_delay_penalty = 0.0
        self.total_incident_host_lwf_count = 0
        self.total_incident_host_lwf_raw_penalty = 0.0
        self.total_response_reward = 0.0

    # --------------------------------------------------------
    # Event creation
    # --------------------------------------------------------

    def _new_event(
        self,
        *,
        hostname: str,
        t_compromise: int,
    ) -> IncidentEvent:
        ordinal = (
            self._host_ordinals.get(
                hostname,
                0,
            )
            + 1
        )

        self._host_ordinals[
            hostname
        ] = ordinal

        event_id = (
            f"seed{self.episode_seed}:"
            f"{self.agent_name}:"
            f"{hostname}:"
            f"incident{ordinal}"
        )

        event = IncidentEvent(
            incident_event_id=event_id,
            agent_name=self.agent_name,
            host_id=hostname,
            ordinal=ordinal,
            t_compromise=int(
                t_compromise
            ),
        )

        self._events.append(
            event
        )

        self._active_by_host[
            hostname
        ] = event

        return event

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset(
        self,
        *,
        initial_red_presence_by_host: Mapping[
            str,
            bool,
        ],
        global_tick: int = 0,
    ) -> None:
        tick = _nonnegative_int(
            "global_tick",
            global_tick,
        )

        presence = _presence_mapping(
            initial_red_presence_by_host
        )

        self._presence = dict(
            presence
        )

        self._active_by_host = {}
        self._host_ordinals = {}
        self._events = []

        self.total_incident_active_ticks = 0
        self.total_incident_delay_penalty = 0.0
        self.total_incident_host_lwf_count = 0
        self.total_incident_host_lwf_raw_penalty = 0.0
        self.total_response_reward = 0.0

        self._last_global_tick = tick
        self._reset_done = True

        # 如果 episode 开始时 host 已经 compromised，
        # 将 episode 起点视作 t_compromise。
        #
        # 这样至少不会丢掉 episode 内
        # 从 start 到 recovery 的持续时间。
        for hostname in sorted(
            presence
        ):
            if presence[
                hostname
            ]:
                self._new_event(
                    hostname=hostname,
                    t_compromise=tick,
                )

    # --------------------------------------------------------
    # Tick
    # --------------------------------------------------------

    def record_tick(
        self,
        *,
        global_tick_end: int,
        red_presence_after: Mapping[
            str,
            bool,
        ],
        local_work_failures: Sequence[
            LocalWorkFailure
        ] = (),
    ) -> IncidentTickAccounting:
        if not self._reset_done:
            raise RuntimeError(
                "reset() must be called first"
            )

        tick_end = _nonnegative_int(
            "global_tick_end",
            global_tick_end,
        )

        tick_start = int(
            self._last_global_tick
        )

        # 正式 CC4 collector 每次只记录一个 global tick。
        if tick_end != tick_start + 1:
            raise ValueError(
                "record_tick must advance "
                "exactly one global tick"
            )

        presence_after = (
            _presence_mapping(
                red_presence_after
            )
        )

        if (
            set(presence_after)
            != set(self._presence)
        ):
            raise ValueError(
                "tracked host set changed "
                "within episode"
            )

        # ====================================================
        # 1. Active duration for interval [t, t+1]
        #
        # 使用 interval 开始时的 presence。
        #
        # False -> True at t+1:
        #     本 interval 不计 active tick。
        #
        # True -> False at t+1:
        #     本 interval 仍计 active tick。
        #
        # 因此累计 active ticks 精确等于：
        #
        #     t_normal - t_compromise
        # ====================================================

        active_before = sorted(
            self._active_by_host.values(),
            key=lambda event:
                event.incident_event_id,
        )

        active_event_ids = tuple(
            event.incident_event_id
            for event in active_before
        )

        active_host_ids = tuple(
            event.host_id
            for event in active_before
        )

        incident_active_ticks = len(
            active_before
        )

        # Final-paper delay term:
        #
        #   sum_i (t - t_compromise_i) I(c_i,t = 1)
        #
        # Each event that was active during this interval contributes its
        # compromise age at the interval end. This preserves the existing
        # convention that the recovery interval is counted.
        incident_delay_penalty = float(
            sum(
                tick_end - event.t_compromise
                for event in active_before
            )
        )

        # ====================================================
        # 2. Incident-host LWF
        # ====================================================

        lwf_count = 0
        lwf_raw_penalty = 0.0

        counted_lwf_event_ids: set[
            str
        ] = set()

        counted_lwf_host_ids: set[
            str
        ] = set()

        for failure in (
            local_work_failures
        ):
            if not isinstance(
                failure,
                LocalWorkFailure,
            ):
                raise TypeError(
                    "local_work_failures must "
                    "contain LocalWorkFailure"
                )

            hostname = str(
                failure.hostname
            )

            if (
                hostname
                not in self._presence
            ):
                raise ValueError(
                    "LWF host is outside "
                    "this Blue agent scope: "
                    f"{hostname}"
                )

            # The final paper defines operation failure over the whole Blue
            # scope, including failures caused by attacks or response actions.
            # Keep the legacy replay field name for compatibility, but count
            # every LWF in this agent's tracked host inventory.
            event = (
                self._active_by_host.get(
                    hostname
                )
            )

            raw = float(
                failure.raw_lwf_penalty
            )

            lwf_count += 1
            lwf_raw_penalty += raw

            if event is not None:
                event.incident_host_lwf_count += 1

                event.incident_host_lwf_raw_penalty += (
                    raw
                )

                counted_lwf_event_ids.add(
                    event.incident_event_id
                )

            counted_lwf_host_ids.add(
                hostname
            )

        # ====================================================
        # 3. Response reward
        # ====================================================

        response_reward = (
            self.reward_config.compute(
                incident_delay_penalty=(
                    incident_delay_penalty
                ),
                incident_host_lwf_raw_penalty=(
                    lwf_raw_penalty
                ),
            )
        )

        # ====================================================
        # 4. Presence transition at tick_end
        # ====================================================

        started_events: list[
            IncidentEvent
        ] = []

        recovered_events: list[
            IncidentEvent
        ] = []

        for hostname in sorted(
            self._presence
        ):
            before = self._presence[
                hostname
            ]

            after = presence_after[
                hostname
            ]

            # normal -> compromised
            if (
                not before
                and after
            ):
                if (
                    hostname
                    in self._active_by_host
                ):
                    raise RuntimeError(
                        "host already has "
                        "active incident"
                    )

                started_events.append(
                    self._new_event(
                        hostname=hostname,
                        t_compromise=tick_end,
                    )
                )

            # compromised -> normal
            elif (
                before
                and not after
            ):
                event = (
                    self._active_by_host.pop(
                        hostname,
                        None,
                    )
                )

                if event is None:
                    raise RuntimeError(
                        "active presence without "
                        "active incident event"
                    )

                event.t_normal = (
                    tick_end
                )

                recovered_events.append(
                    event
                )

        # ====================================================
        # 5. Related event identities for replay audit
        # ====================================================

        related_events = {
            event.incident_event_id:
                event
            for event in (
                list(active_before)
                + started_events
                + recovered_events
            )
        }

        related_event_ids = tuple(
            sorted(
                related_events.keys()
            )
        )

        related_host_ids = tuple(
            sorted(
                {
                    event.host_id
                    for event
                    in related_events.values()
                }
            )
        )

        # ====================================================
        # 6. Totals
        # ====================================================

        self.total_incident_active_ticks += (
            incident_active_ticks
        )

        self.total_incident_delay_penalty += (
            incident_delay_penalty
        )

        self.total_incident_host_lwf_count += (
            lwf_count
        )

        self.total_incident_host_lwf_raw_penalty += (
            lwf_raw_penalty
        )

        self.total_response_reward += (
            response_reward
        )

        self._presence = dict(
            presence_after
        )

        self._last_global_tick = (
            tick_end
        )

        return IncidentTickAccounting(
            global_tick_start=(
                tick_start
            ),

            global_tick_end=(
                tick_end
            ),

            incident_event_ids=(
                related_event_ids
            ),

            incident_host_ids=(
                related_host_ids
            ),

            active_event_ids=(
                active_event_ids
            ),

            active_host_ids=(
                active_host_ids
            ),

            started_event_ids=tuple(
                sorted(
                    event.incident_event_id
                    for event
                    in started_events
                )
            ),

            recovered_event_ids=tuple(
                sorted(
                    event.incident_event_id
                    for event
                    in recovered_events
                )
            ),

            counted_lwf_event_ids=tuple(
                sorted(
                    counted_lwf_event_ids
                )
            ),

            counted_lwf_host_ids=tuple(
                sorted(
                    counted_lwf_host_ids
                )
            ),

            incident_active_ticks=(
                incident_active_ticks
            ),

            incident_delay_penalty=(
                incident_delay_penalty
            ),

            incident_host_lwf_count=(
                lwf_count
            ),

            incident_host_lwf_raw_penalty=(
                float(
                    lwf_raw_penalty
                )
            ),

            response_reward=(
                float(
                    response_reward
                )
            ),
        )

    # --------------------------------------------------------
    # Read-only snapshots
    # --------------------------------------------------------

    @property
    def last_global_tick(
        self,
    ) -> int:
        return int(
            self._last_global_tick
        )

    @property
    def all_events(
        self,
    ) -> tuple[
        IncidentEvent,
        ...,
    ]:
        return tuple(
            replace(event)
            for event in self._events
        )

    @property
    def active_events(
        self,
    ) -> tuple[
        IncidentEvent,
        ...,
    ]:
        return tuple(
            replace(event)
            for event in sorted(
                self._active_by_host.values(),
                key=lambda item:
                    item.incident_event_id,
            )
        )

    @property
    def completed_events(
        self,
    ) -> tuple[
        IncidentEvent,
        ...,
    ]:
        return tuple(
            replace(event)
            for event in self._events
            if event.t_normal is not None
        )

    @property
    def unresolved_event_count(
        self,
    ) -> int:
        return len(
            self._active_by_host
        )

    @property
    def mean_completed_attack_eradication_time(
        self,
    ) -> float | None:
        values = [
            event.attack_eradication_time
            for event
            in self._events
            if (
                event.attack_eradication_time
                is not None
            )
        ]

        if not values:
            return None

        return float(
            sum(values)
            / len(values)
        )
