from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import (
    Any,
    Mapping,
    Optional,
)

import numpy as np


# ============================================================
# Formal state contract
# ============================================================

BLUE_AGENTS = (
    "blue_agent_0",
    "blue_agent_1",
    "blue_agent_2",
    "blue_agent_3",
    "blue_agent_4",
)

META_KEYS = {
    "success",
    "action",
    "message",
}


FORMAL_STATE_FEATURE_NAMES = (
    # agent identity
    "agent_blue_0",
    "agent_blue_1",
    "agent_blue_2",
    "agent_blue_3",
    "agent_blue_4",

    # global / inventory
    "global_tick_fraction",
    "inventory_size",

    # host-level evidence coverage
    "suspicious_host_fraction",
    "current_evidence_host_fraction",
    "process_evidence_host_fraction",
    "connection_evidence_host_fraction",
    "file_evidence_host_fraction",

    # current observation intensity
    "current_process_event_density",
    "current_connection_event_density",
    "current_file_event_density",

    # observable target summary
    "max_host_score",
    "mean_positive_host_score",
    "any_observable_target",

    # top-ranked observable host
    "top_host_has_process",
    "top_host_has_connection",
    "top_host_has_file",
    "top_host_observation_hits",
    "top_host_recency",

    # previous action result
    "success_false",
    "success_true",
    "success_unknown",
    "success_in_progress",
)


FORMAL_STATE_DIM = len(
    FORMAL_STATE_FEATURE_NAMES
)


if FORMAL_STATE_DIM != 27:
    raise RuntimeError(
        "formal state dimension contract "
        f"changed unexpectedly: "
        f"{FORMAL_STATE_DIM}"
    )


# ============================================================
# Evidence state
# ============================================================

@dataclass
class HostEvidence:
    """
    一个 Blue agent 对一台 host 的
    planner-visible evidence。

    注意：
        所有字段只能由 Blue observation 更新。

    禁止来源：
        controller true state
        Red sessions
        compromise truth
        attack labels
        reward ground truth
    """

    hostname: str

    process_events: int = 0
    connection_events: int = 0
    file_events: int = 0

    observation_hits: int = 0

    first_seen_tick: Optional[int] = None
    last_seen_tick: Optional[int] = None

    def has_process(
        self,
    ) -> bool:
        return (
            self.process_events > 0
        )

    def has_connection(
        self,
    ) -> bool:
        return (
            self.connection_events > 0
        )

    def has_file(
        self,
    ) -> bool:
        return (
            self.file_events > 0
        )

    def has_any_evidence(
        self,
    ) -> bool:
        return (
            self.process_events > 0
            or self.connection_events > 0
            or self.file_events > 0
        )


@dataclass(frozen=True)
class CurrentObservationStats:
    """
    当前一次 post-reset Blue observation
    中看到的 evidence。

    与累计 HostEvidence 分离。
    """

    process_events: int
    connection_events: int
    file_events: int
    evidence_hosts: tuple[str, ...]


# ============================================================
# Helpers
# ============================================================

def _observation_hosts(
    observation: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:

    out: dict[
        str,
        Mapping[str, Any],
    ] = {}

    for raw_key, raw_value in (
        observation.items()
    ):
        key = str(
            raw_key
        )

        if key in META_KEYS:
            continue

        if not isinstance(
            raw_value,
            Mapping,
        ):
            continue

        out[
            key
        ] = raw_value

    return out


def _count_host_evidence(
    data: Mapping[str, Any],
) -> tuple[
    int,
    int,
    int,
]:
    """
    从一次 Blue-observable host record 中统计：

        process events
        network connection events
        file evidence

    reset 时不会调用本函数更新 threat state，
    因为 A4.1a 已确认 reset Processes 是 baseline。
    """

    raw_processes = data.get(
        "Processes",
        [],
    )

    raw_files = data.get(
        "Files",
        [],
    )

    process_count = 0
    connection_count = 0
    file_count = 0

    if isinstance(
        raw_processes,
        (list, tuple),
    ):
        for process in raw_processes:

            if not isinstance(
                process,
                Mapping,
            ):
                continue

            process_count += 1

            raw_connections = (
                process.get(
                    "Connections",
                    [],
                )
            )

            if isinstance(
                raw_connections,
                (list, tuple),
            ):
                connection_count += sum(
                    1
                    for item
                    in raw_connections
                    if isinstance(
                        item,
                        Mapping,
                    )
                )

    if isinstance(
        raw_files,
        (list, tuple),
    ):
        file_count = sum(
            1
            for item
            in raw_files
            if isinstance(
                item,
                Mapping,
            )
        )

    return (
        int(
            process_count
        ),
        int(
            connection_count
        ),
        int(
            file_count
        ),
    )


def _squash_nonnegative(
    value: float,
) -> float:
    """
    x -> x / (1 + x)

    将非负无界量映射至 [0, 1)。

    不依赖 seed / agent-specific max，
    后续更适合统一 WM normalization。
    """

    x = float(
        value
    )

    if (
        not isfinite(
            x
        )
        or x < 0.0
    ):
        raise ValueError(
            "value must be a finite "
            f"non-negative number: {value!r}"
        )

    if x == 0.0:
        return 0.0

    return float(
        x / (
            1.0 + x
        )
    )


def _success_name(
    raw_success: Any,
) -> str:
    """
    兼容：

        TernaryEnum.TRUE
        <TernaryEnum.TRUE: 1>
        "TRUE"
        bool

    输出固定四类：
        FALSE
        TRUE
        UNKNOWN
        IN_PROGRESS
    """

    if isinstance(
        raw_success,
        bool,
    ):
        return (
            "TRUE"
            if raw_success
            else "FALSE"
        )

    enum_name = getattr(
        raw_success,
        "name",
        None,
    )

    if enum_name is not None:
        text = str(
            enum_name
        ).upper()

    else:
        text = str(
            raw_success
        ).upper()

    for name in (
        "IN_PROGRESS",
        "UNKNOWN",
        "FALSE",
        "TRUE",
    ):
        if name in text:
            return name

    # 缺失或未来 wrapper 未知值
    # 保守编码为 UNKNOWN。
    return "UNKNOWN"


def _success_one_hot(
    raw_success: Any,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    name = _success_name(
        raw_success
    )

    order = (
        "FALSE",
        "TRUE",
        "UNKNOWN",
        "IN_PROGRESS",
    )

    return tuple(
        1.0
        if item == name
        else 0.0
        for item in order
    )


# ============================================================
# Observable evidence tracker
# ============================================================

class ObservableHostEvidenceTracker:
    """
    A4 正式 planner-visible host evidence tracker。

    生命周期：

        tracker = ObservableHostEvidenceTracker(agent)

        tracker.reset(reset_observation)

        tracker.update(
            observation=...,
            global_tick=...,
            completed_action_family=...,
            completed_target_host=...,
            completed_action_success=...,
        )

    最重要的 contract：

    RESET:
        只建立 inventory + IP mapping。
        RESET 中的 Processes 不算 threat evidence。

    POST-RESET:
        Processes
        Connections
        Files
        才进入 evidence。

    Ground truth:
        永远不读取。
    """

    def __init__(
        self,
        agent_name: str,
    ):
        agent_name = str(
            agent_name
        )

        if (
            agent_name
            not in BLUE_AGENTS
        ):
            raise ValueError(
                "unsupported Blue agent: "
                f"{agent_name!r}"
            )

        self.agent_name = (
            agent_name
        )

        self._inventory: tuple[
            str,
            ...,
        ] = tuple()

        self._inventory_set: set[
            str
        ] = set()

        self._ip_to_hostname: dict[
            str,
            str,
        ] = {}

        self._evidence: dict[
            str,
            HostEvidence,
        ] = {}

        self._current_stats = (
            CurrentObservationStats(
                process_events=0,
                connection_events=0,
                file_events=0,
                evidence_hosts=tuple(),
            )
        )

        self._reset_done = False
        self._last_global_tick = 0

    # --------------------------------------------------------
    # public properties
    # --------------------------------------------------------

    @property
    def inventory(
        self,
    ) -> tuple[str, ...]:
        return self._inventory

    @property
    def inventory_size(
        self,
    ) -> int:
        return len(
            self._inventory
        )

    @property
    def ip_to_hostname(
        self,
    ) -> dict[str, str]:
        return dict(
            self._ip_to_hostname
        )

    @property
    def current_stats(
        self,
    ) -> CurrentObservationStats:
        return self._current_stats

    @property
    def last_global_tick(
        self,
    ) -> int:
        return int(
            self._last_global_tick
        )

    # --------------------------------------------------------
    # reset
    # --------------------------------------------------------

    def reset(
        self,
        reset_observation: Mapping[
            str,
            Any,
        ],
    ) -> None:
        """
        只从 Blue reset observation 建 inventory。

        不从 controller 获取 hostname。
        """

        if not isinstance(
            reset_observation,
            Mapping,
        ):
            raise TypeError(
                "reset_observation "
                "must be a mapping"
            )

        hosts = (
            _observation_hosts(
                reset_observation
            )
        )

        hostnames: set[
            str
        ] = set()

        ip_map: dict[
            str,
            str,
        ] = {}

        for raw_key, data in (
            hosts.items()
        ):
            system_info = (
                data.get(
                    "System info",
                    {},
                )
            )

            hostname = None

            if isinstance(
                system_info,
                Mapping,
            ):
                value = (
                    system_info.get(
                        "Hostname"
                    )
                )

                if value is not None:
                    hostname = str(
                        value
                    )

            if not hostname:
                hostname = str(
                    raw_key
                )

            hostnames.add(
                hostname
            )

            raw_interfaces = (
                data.get(
                    "Interface",
                    [],
                )
            )

            if isinstance(
                raw_interfaces,
                (list, tuple),
            ):
                for interface in (
                    raw_interfaces
                ):
                    if not isinstance(
                        interface,
                        Mapping,
                    ):
                        continue

                    ip = interface.get(
                        "ip_address"
                    )

                    if ip is None:
                        continue

                    ip_map[
                        str(ip)
                    ] = hostname

        if not hostnames:
            raise ValueError(
                f"{self.agent_name}: "
                "reset observation produced "
                "empty host inventory"
            )

        self._inventory = tuple(
            sorted(
                hostnames
            )
        )

        self._inventory_set = set(
            self._inventory
        )

        self._ip_to_hostname = (
            ip_map
        )

        # ====================================================
        # CRITICAL
        #
        # reset Processes are baseline,
        # NOT threat evidence.
        # ====================================================
        self._evidence = {
            hostname:
                HostEvidence(
                    hostname=hostname
                )
            for hostname
            in self._inventory
        }

        self._current_stats = (
            CurrentObservationStats(
                process_events=0,
                connection_events=0,
                file_events=0,
                evidence_hosts=tuple(),
            )
        )

        self._last_global_tick = 0
        self._reset_done = True

    # --------------------------------------------------------
    # hostname mapping
    # --------------------------------------------------------

    def canonical_hostname(
        self,
        raw_key: Any,
        data: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> Optional[str]:
        """
        hostname key 或 IP key -> reset inventory hostname。

        未知 host 不进入 planner evidence。
        """

        text = str(
            raw_key
        )

        if (
            text
            in self._inventory_set
        ):
            return text

        mapped = (
            self._ip_to_hostname.get(
                text
            )
        )

        if mapped is not None:
            return mapped

        if isinstance(
            data,
            Mapping,
        ):
            system_info = (
                data.get(
                    "System info",
                    {}
                )
            )

            if isinstance(
                system_info,
                Mapping,
            ):
                hostname = (
                    system_info.get(
                        "Hostname"
                    )
                )

                if hostname is not None:
                    hostname = str(
                        hostname
                    )

                    if (
                        hostname
                        in self._inventory_set
                    ):
                        return hostname

        return None

    # --------------------------------------------------------
    # evidence maintenance
    # --------------------------------------------------------

    def clear_host(
        self,
        hostname: str,
    ) -> None:
        """
        清除 planner-visible historical evidence。

        正式语义：
            successful Restore 可以调用。

        Remove success 不调用，
        因为 Remove != host definitely normal。
        """

        hostname = str(
            hostname
        )

        if (
            hostname
            not in self._inventory_set
        ):
            return

        self._evidence[
            hostname
        ] = HostEvidence(
            hostname=hostname
        )

    def update(
        self,
        *,
        observation: Mapping[
            str,
            Any,
        ],
        global_tick: int,
        completed_action_family: Optional[
            str
        ] = None,
        completed_target_host: Optional[
            str
        ] = None,
        completed_action_success: Optional[
            bool
        ] = None,
    ) -> None:
        """
        用一次真实 Blue post-reset observation
        更新 evidence。

        completed_* 仅表示 Blue 自己刚执行的动作，
        不是 hidden ground truth。

        Restore success:
            清除旧 evidence，
            然后再处理当前 observation 中
            新出现的 Monitor / Analyse evidence。
        """

        if not self._reset_done:
            raise RuntimeError(
                "tracker.reset() must "
                "be called before update()"
            )

        if not isinstance(
            observation,
            Mapping,
        ):
            raise TypeError(
                "observation must "
                "be a mapping"
            )

        if (
            isinstance(
                global_tick,
                bool,
            )
            or not isinstance(
                global_tick,
                int,
            )
            or global_tick < 0
        ):
            raise ValueError(
                "global_tick must be "
                "a non-negative int"
            )

        if (
            global_tick
            < self._last_global_tick
        ):
            raise ValueError(
                "global_tick must be "
                "monotonic"
            )

        # ====================================================
        # Restore semantics
        # ====================================================

        if (
            completed_action_family
            == "Restore"
            and completed_target_host
            is not None
            and completed_action_success
            is True
        ):
            self.clear_host(
                str(
                    completed_target_host
                )
            )

        current_processes = 0
        current_connections = 0
        current_files = 0

        current_hosts: set[
            str
        ] = set()

        for raw_key, data in (
            _observation_hosts(
                observation
            ).items()
        ):
            hostname = (
                self.canonical_hostname(
                    raw_key,
                    data,
                )
            )

            if hostname is None:
                # 不允许 observation 中
                # 未知 host 被自动扩展进 inventory。
                continue

            (
                process_count,
                connection_count,
                file_count,
            ) = _count_host_evidence(
                data
            )

            current_processes += (
                process_count
            )

            current_connections += (
                connection_count
            )

            current_files += (
                file_count
            )

            has_evidence = (
                process_count > 0
                or connection_count > 0
                or file_count > 0
            )

            if not has_evidence:
                continue

            current_hosts.add(
                hostname
            )

            evidence = (
                self._evidence[
                    hostname
                ]
            )

            evidence.process_events += int(
                process_count
            )

            evidence.connection_events += int(
                connection_count
            )

            evidence.file_events += int(
                file_count
            )

            evidence.observation_hits += 1

            if (
                evidence.first_seen_tick
                is None
            ):
                evidence.first_seen_tick = (
                    int(
                        global_tick
                    )
                )

            evidence.last_seen_tick = int(
                global_tick
            )

        self._current_stats = (
            CurrentObservationStats(
                process_events=int(
                    current_processes
                ),
                connection_events=int(
                    current_connections
                ),
                file_events=int(
                    current_files
                ),
                evidence_hosts=tuple(
                    sorted(
                        current_hosts
                    )
                ),
            )
        )

        self._last_global_tick = int(
            global_tick
        )

    # --------------------------------------------------------
    # evidence snapshots
    # --------------------------------------------------------

    def evidence_for(
        self,
        hostname: str,
    ) -> HostEvidence:

        hostname = str(
            hostname
        )

        if (
            hostname
            not in self._evidence
        ):
            raise KeyError(
                hostname
            )

        item = (
            self._evidence[
                hostname
            ]
        )

        # 返回复制，避免调用方绕过 tracker
        # 修改内部状态。
        return HostEvidence(
            hostname=item.hostname,
            process_events=int(
                item.process_events
            ),
            connection_events=int(
                item.connection_events
            ),
            file_events=int(
                item.file_events
            ),
            observation_hits=int(
                item.observation_hits
            ),
            first_seen_tick=(
                item.first_seen_tick
            ),
            last_seen_tick=(
                item.last_seen_tick
            ),
        )

    def suspicious_hosts(
        self,
    ) -> tuple[str, ...]:

        return tuple(
            sorted(
                hostname
                for (
                    hostname,
                    evidence,
                ) in self._evidence.items()
                if evidence.has_any_evidence()
            )
        )

    # --------------------------------------------------------
    # host scoring
    # --------------------------------------------------------

    @staticmethod
    def _raw_evidence_score(
        evidence: HostEvidence,
        *,
        global_tick: int,
    ) -> float:
        """
        该 score 仅用于：

            Analyse / Remove / Restore
            的 observable target ranking。

        它不是：
            attack probability
            PPO reward
            world-model value

        权重表达证据强度顺序：

            process    1
            connection 2
            file       3

        observation_hits 只提供小幅累积证据；
        recency 只提供小幅时效信息。

        所有方法共享同一规则。
        """

        if (
            not evidence
            .has_any_evidence()
        ):
            return 0.0

        base = (
            1.0
            * float(
                evidence.process_events
            )
            +
            2.0
            * float(
                evidence.connection_events
            )
            +
            3.0
            * float(
                evidence.file_events
            )
        )

        hit_bonus = (
            0.25
            * float(
                evidence.observation_hits
            )
        )

        recency_bonus = 0.0

        if (
            evidence.last_seen_tick
            is not None
        ):
            age = max(
                0,
                int(
                    global_tick
                )
                -
                int(
                    evidence.last_seen_tick
                ),
            )

            recency_bonus = (
                1.0
                / float(
                    1 + age
                )
            )

        return float(
            base
            + hit_bonus
            + recency_bonus
        )

    def observable_host_scores(
        self,
        global_tick: Optional[
            int
        ] = None,
    ) -> dict[
        str,
        float,
    ]:
        """
        A3 CybORGActionAdapter 的正式输入。
        """

        if not self._reset_done:
            raise RuntimeError(
                "tracker.reset() must "
                "be called first"
            )

        if global_tick is None:
            global_tick = (
                self._last_global_tick
            )

        if (
            isinstance(
                global_tick,
                bool,
            )
            or not isinstance(
                global_tick,
                int,
            )
            or global_tick < 0
        ):
            raise ValueError(
                "global_tick must be "
                "a non-negative int"
            )

        result = {}

        for hostname, evidence in (
            self._evidence.items()
        ):
            score = (
                self._raw_evidence_score(
                    evidence,
                    global_tick=global_tick,
                )
            )

            if score > 0.0:
                result[
                    hostname
                ] = float(
                    score
                )

        return result

    def ranked_host_evidence(
        self,
        *,
        global_tick: Optional[
            int
        ] = None,
        limit: Optional[
            int
        ] = None,
    ) -> list[
        dict[str, Any]
    ]:
        """
        后续 Gate B 可由这里生成
        LLM textual incident summary。

        现在先冻结 observable-only source。
        """

        if global_tick is None:
            global_tick = (
                self._last_global_tick
            )

        scores = (
            self.observable_host_scores(
                global_tick
            )
        )

        ranked = sorted(
            scores.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        if limit is not None:
            if (
                isinstance(
                    limit,
                    bool,
                )
                or not isinstance(
                    limit,
                    int,
                )
                or limit < 0
            ):
                raise ValueError(
                    "limit must be "
                    "a non-negative int "
                    "or None"
                )

            ranked = ranked[
                :limit
            ]

        out = []

        for hostname, score in ranked:

            evidence = (
                self._evidence[
                    hostname
                ]
            )

            if (
                evidence.last_seen_tick
                is None
            ):
                age = None
            else:
                age = max(
                    0,
                    int(
                        global_tick
                    )
                    -
                    int(
                        evidence.last_seen_tick
                    ),
                )

            out.append(
                {
                    "hostname":
                        hostname,

                    "score":
                        float(
                            score
                        ),

                    "process_events":
                        int(
                            evidence
                            .process_events
                        ),

                    "connection_events":
                        int(
                            evidence
                            .connection_events
                        ),

                    "file_events":
                        int(
                            evidence
                            .file_events
                        ),

                    "observation_hits":
                        int(
                            evidence
                            .observation_hits
                        ),

                    "age_ticks":
                        age,
                }
            )

        return out


# ============================================================
# Formal state encoder
# ============================================================

class FormalStateEncoder:
    """
    A4.1b 正式固定维度 state encoder。

    输出：
        np.ndarray shape=(27,)
        dtype=float32

    输入只来自：
        ObservableHostEvidenceTracker
        当前 Blue observation
        当前 global tick
        episode_steps

    不允许 env / controller 参数。
    """

    state_dim = (
        FORMAL_STATE_DIM
    )

    feature_names = (
        FORMAL_STATE_FEATURE_NAMES
    )

    @staticmethod
    def _agent_one_hot(
        agent_name: str,
    ) -> list[float]:

        if (
            agent_name
            not in BLUE_AGENTS
        ):
            raise ValueError(
                f"unsupported agent: "
                f"{agent_name!r}"
            )

        return [
            1.0
            if agent_name == item
            else 0.0
            for item in BLUE_AGENTS
        ]

    def encode(
        self,
        *,
        tracker: (
            ObservableHostEvidenceTracker
        ),
        observation: Mapping[
            str,
            Any,
        ],
        global_tick: int,
        episode_steps: int,
    ) -> np.ndarray:

        if not isinstance(
            tracker,
            ObservableHostEvidenceTracker,
        ):
            raise TypeError(
                "tracker must be "
                "ObservableHostEvidenceTracker"
            )

        if not isinstance(
            observation,
            Mapping,
        ):
            raise TypeError(
                "observation must "
                "be a mapping"
            )

        if (
            isinstance(
                global_tick,
                bool,
            )
            or not isinstance(
                global_tick,
                int,
            )
            or global_tick < 0
        ):
            raise ValueError(
                "global_tick must be "
                "a non-negative int"
            )

        if (
            isinstance(
                episode_steps,
                bool,
            )
            or not isinstance(
                episode_steps,
                int,
            )
            or episode_steps <= 0
        ):
            raise ValueError(
                "episode_steps must "
                "be a positive int"
            )

        inventory_size = (
            tracker.inventory_size
        )

        if inventory_size <= 0:
            raise RuntimeError(
                "tracker inventory "
                "must not be empty"
            )

        inventory_den = float(
            inventory_size
        )

        scores = (
            tracker.observable_host_scores(
                global_tick
            )
        )

        suspicious_hosts = (
            tracker.suspicious_hosts()
        )

        process_hosts = 0
        connection_hosts = 0
        file_hosts = 0

        for hostname in (
            tracker.inventory
        ):
            evidence = (
                tracker.evidence_for(
                    hostname
                )
            )

            if evidence.has_process():
                process_hosts += 1

            if evidence.has_connection():
                connection_hosts += 1

            if evidence.has_file():
                file_hosts += 1

        current = (
            tracker.current_stats
        )

        # ----------------------------------------------------
        # top observable host
        # ----------------------------------------------------

        top_hostname = None
        top_score = 0.0
        top_evidence = None

        if scores:
            top_hostname = min(
                scores.keys(),
                key=lambda hostname: (
                    -scores[
                        hostname
                    ],
                    hostname,
                ),
            )

            top_score = float(
                scores[
                    top_hostname
                ]
            )

            top_evidence = (
                tracker.evidence_for(
                    top_hostname
                )
            )

        positive_scores = list(
            scores.values()
        )

        if positive_scores:
            mean_score = float(
                np.mean(
                    np.asarray(
                        positive_scores,
                        dtype=np.float64,
                    )
                )
            )
        else:
            mean_score = 0.0

        # ----------------------------------------------------
        # recency of top host
        # ----------------------------------------------------

        top_recency = 0.0

        if (
            top_evidence is not None
            and
            top_evidence.last_seen_tick
            is not None
        ):
            age = max(
                0,
                global_tick
                -
                int(
                    top_evidence
                    .last_seen_tick
                ),
            )

            top_recency = float(
                1.0
                / float(
                    1 + age
                )
            )

        # ----------------------------------------------------
        # success
        # ----------------------------------------------------

        (
            success_false,
            success_true,
            success_unknown,
            success_in_progress,
        ) = _success_one_hot(
            observation.get(
                "success"
            )
        )

        # ----------------------------------------------------
        # vector
        # ----------------------------------------------------

        features: list[
            float
        ] = []

        # 0..4
        features.extend(
            self._agent_one_hot(
                tracker.agent_name
            )
        )

        # 5
        features.append(
            float(
                min(
                    max(
                        global_tick
                        / float(
                            episode_steps
                        ),
                        0.0,
                    ),
                    1.0,
                )
            )
        )

        # 6
        features.append(
            _squash_nonnegative(
                float(
                    inventory_size
                )
            )
        )

        # 7
        features.append(
            float(
                len(
                    suspicious_hosts
                )
                / inventory_den
            )
        )

        # 8
        features.append(
            float(
                len(
                    current
                    .evidence_hosts
                )
                / inventory_den
            )
        )

        # 9
        features.append(
            float(
                process_hosts
                / inventory_den
            )
        )

        # 10
        features.append(
            float(
                connection_hosts
                / inventory_den
            )
        )

        # 11
        features.append(
            float(
                file_hosts
                / inventory_den
            )
        )

        # 12
        features.append(
            _squash_nonnegative(
                current.process_events
                / inventory_den
            )
        )

        # 13
        features.append(
            _squash_nonnegative(
                current.connection_events
                / inventory_den
            )
        )

        # 14
        features.append(
            _squash_nonnegative(
                current.file_events
                / inventory_den
            )
        )

        # 15
        features.append(
            _squash_nonnegative(
                top_score
            )
        )

        # 16
        features.append(
            _squash_nonnegative(
                mean_score
            )
        )

        # 17
        features.append(
            1.0
            if scores
            else 0.0
        )

        # 18..22
        if top_evidence is None:
            features.extend(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
        else:
            features.extend(
                [
                    (
                        1.0
                        if (
                            top_evidence
                            .has_process()
                        )
                        else 0.0
                    ),
                    (
                        1.0
                        if (
                            top_evidence
                            .has_connection()
                        )
                        else 0.0
                    ),
                    (
                        1.0
                        if (
                            top_evidence
                            .has_file()
                        )
                        else 0.0
                    ),
                    _squash_nonnegative(
                        float(
                            top_evidence
                            .observation_hits
                        )
                    ),
                    float(
                        top_recency
                    ),
                ]
            )

        # 23..26
        features.extend(
            [
                success_false,
                success_true,
                success_unknown,
                success_in_progress,
            ]
        )

        vector = np.asarray(
            features,
            dtype=np.float32,
        )

        if (
            vector.shape
            != (
                FORMAL_STATE_DIM,
            )
        ):
            raise RuntimeError(
                "formal state shape mismatch: "
                f"{vector.shape}"
            )

        if not np.all(
            np.isfinite(
                vector
            )
        ):
            raise RuntimeError(
                "formal state contains "
                "non-finite values"
            )

        return vector