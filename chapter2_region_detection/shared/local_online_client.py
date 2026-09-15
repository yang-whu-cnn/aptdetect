from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from shared.action_contract import get_action
from shared.local_action_adapter import (
    LocalActionAdapter,
    LocalActionAdapterConfig,
    LocalThreatState,
)


@dataclass(frozen=True)
class LocalOnlineConfig:
    """
    Gate A 开发期共享 local-online 环境配置。

    这些参数只负责构造可交互 transition environment，
    不是最终论文 reward 参数。
    """

    max_steps: int = 100
    seed: int = 0
    region_id: int = 0

    # analyse 提升的 visibility 会逐步回落，
    # 避免一次 analyse 获得永久完全观测能力。
    visibility_floor: float = 0.20
    visibility_decay: float = 0.85

    # 自然威胁演化。
    threat_drift: float = 0.018
    threat_noise_std: float = 0.012
    burst_probability: float = 0.10
    burst_scale: float = 0.05

    # observation generation。
    event_rate: float = 2.50
    max_events_per_type: int = 4


@dataclass(frozen=True)
class _RegionProfile:
    """
    local simulator 私有 region profile。

    仅用于产生不同区域的初始攻击面和自然演化差异。
    不暴露给 planner。
    """

    name: str

    initial_auth: float
    initial_scan: float
    initial_process: float
    initial_outbound: float

    auth_drift_scale: float
    scan_drift_scale: float
    process_drift_scale: float
    outbound_drift_scale: float


_REGION_PROFILES = {
    0: _RegionProfile(
        name="hq",
        initial_auth=0.30,
        initial_scan=0.42,
        initial_process=0.28,
        initial_outbound=0.26,
        auth_drift_scale=1.00,
        scan_drift_scale=1.20,
        process_drift_scale=0.95,
        outbound_drift_scale=0.90,
    ),
    1: _RegionProfile(
        name="edge_a",
        initial_auth=0.28,
        initial_scan=0.34,
        initial_process=0.43,
        initial_outbound=0.32,
        auth_drift_scale=0.95,
        scan_drift_scale=0.95,
        process_drift_scale=1.20,
        outbound_drift_scale=1.00,
    ),
    2: _RegionProfile(
        name="edge_b",
        initial_auth=0.30,
        initial_scan=0.32,
        initial_process=0.30,
        initial_outbound=0.50,
        auth_drift_scale=1.00,
        scan_drift_scale=0.90,
        process_drift_scale=0.95,
        outbound_drift_scale=1.25,
    ),
    3: _RegionProfile(
        name="branch",
        initial_auth=0.42,
        initial_scan=0.34,
        initial_process=0.32,
        initial_outbound=0.28,
        auth_drift_scale=1.20,
        scan_drift_scale=1.00,
        process_drift_scale=1.00,
        outbound_drift_scale=0.95,
    ),
}


_EVENT_TYPES = (
    "auth_fail",
    "port_scan",
    "proc_spawn",
    "outbound_conn",
)


class SharedLocalOnlineClient:
    """
    Gate A 使用的共享 local-online transition environment。

    它与旧 src.cc4_client.OnlineCC4Client 分离。

    设计原则：

    1. Ours / UG-CEM 后续共用；
    2. action ID 只作为类别 ID；
    3. 五类动作通过 LocalActionAdapter 产生不同语义；
    4. latent threat state 不暴露给 planner；
    5. risk_proxy 从可观测 events 计算，
       不直接返回真实 latent risk；
    6. 当前 reward 固定为 0.0；
    7. 正式 reward 留到 Gate B。
    """

    def __init__(
        self,
        cfg: LocalOnlineConfig | None = None,
        action_cfg: LocalActionAdapterConfig | None = None,
    ):
        self.cfg = (
            cfg
            if cfg is not None
            else LocalOnlineConfig()
        )

        self._validate_config()

        self.profile = _REGION_PROFILES[
            self.cfg.region_id
        ]

        self.action_adapter = LocalActionAdapter(
            cfg=action_cfg
        )

        self._rng = np.random.RandomState(
            self.cfg.seed
        )

        self.t = 0
        self.done = False

        self._latent_state = (
            self._make_initial_latent_state()
        )

        self._last_obs: Dict[str, Any] = {}

    def _validate_config(
        self,
    ) -> None:
        if int(self.cfg.max_steps) <= 0:
            raise ValueError(
                "max_steps must be > 0"
            )

        if int(self.cfg.region_id) not in (
            _REGION_PROFILES
        ):
            raise ValueError(
                "region_id must be one of "
                f"{sorted(_REGION_PROFILES)}"
            )

        if not (
            0.0
            <= float(
                self.cfg.visibility_floor
            )
            <= 1.0
        ):
            raise ValueError(
                "visibility_floor "
                "must be in [0, 1]"
            )

        if not (
            0.0
            <= float(
                self.cfg.visibility_decay
            )
            <= 1.0
        ):
            raise ValueError(
                "visibility_decay "
                "must be in [0, 1]"
            )

        if float(
            self.cfg.threat_drift
        ) < 0.0:
            raise ValueError(
                "threat_drift must be >= 0"
            )

        if float(
            self.cfg.threat_noise_std
        ) < 0.0:
            raise ValueError(
                "threat_noise_std must be >= 0"
            )

        if not (
            0.0
            <= float(
                self.cfg.burst_probability
            )
            <= 1.0
        ):
            raise ValueError(
                "burst_probability "
                "must be in [0, 1]"
            )

        if float(
            self.cfg.burst_scale
        ) < 0.0:
            raise ValueError(
                "burst_scale must be >= 0"
            )

        if float(
            self.cfg.event_rate
        ) < 0.0:
            raise ValueError(
                "event_rate must be >= 0"
            )

        if int(
            self.cfg.max_events_per_type
        ) <= 0:
            raise ValueError(
                "max_events_per_type "
                "must be > 0"
            )

    @staticmethod
    def _clip01(
        value: float,
    ) -> float:
        return float(
            np.clip(
                float(value),
                0.0,
                1.0,
            )
        )

    def _make_initial_latent_state(
        self,
    ) -> LocalThreatState:
        """
        构造当前 region 的初始私有状态。
        """

        return LocalThreatState(
            auth_pressure=(
                self.profile.initial_auth
            ),
            scan_pressure=(
                self.profile.initial_scan
            ),
            process_pressure=(
                self.profile.initial_process
            ),
            outbound_pressure=(
                self.profile.initial_outbound
            ),
            visibility=(
                self.cfg.visibility_floor
            ),
        )

    def reset(
        self,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        重置 episode。

        同一 region + 同一 seed
        必须得到可复现轨迹。
        """

        base_seed = (
            self.cfg.seed
            if seed is None
            else int(seed)
        )

        actual_seed = (
            int(base_seed)
            + int(self.cfg.region_id) * 1009
        )

        self._rng = np.random.RandomState(
            actual_seed
        )

        self.t = 0
        self.done = False

        self._latent_state = (
            self._make_initial_latent_state()
        )

        obs = self._make_observation()
        self._last_obs = obs

        return obs

    def _natural_evolve(
        self,
        state: LocalThreatState,
    ) -> LocalThreatState:
        """
        action 执行之前的自然攻击演化。

        该过程与 planner 无关，也不依赖 action ID。
        """

        current = np.array(
            [
                state.auth_pressure,
                state.scan_pressure,
                state.process_pressure,
                state.outbound_pressure,
            ],
            dtype=np.float64,
        )

        drift_scale = np.array(
            [
                self.profile.auth_drift_scale,
                self.profile.scan_drift_scale,
                self.profile.process_drift_scale,
                self.profile.outbound_drift_scale,
            ],
            dtype=np.float64,
        )

        drift = (
            float(self.cfg.threat_drift)
            * drift_scale
        )

        noise = self._rng.normal(
            loc=0.0,
            scale=float(
                self.cfg.threat_noise_std
            ),
            size=4,
        )

        burst = np.zeros(
            4,
            dtype=np.float64,
        )

        if (
            self._rng.rand()
            < float(
                self.cfg.burst_probability
            )
        ):
            burst_idx = int(
                self._rng.randint(
                    0,
                    4,
                )
            )

            burst[
                burst_idx
            ] = float(
                self.cfg.burst_scale
            )

        next_pressure = np.clip(
            current
            + drift
            + noise
            + burst,
            0.0,
            1.0,
        )

        # analyse 的信息增益不是永久的。
        # visibility 会逐步回落到 floor。
        next_visibility = (
            float(
                self.cfg.visibility_floor
            )
            +
            float(
                self.cfg.visibility_decay
            )
            * (
                float(state.visibility)
                -
                float(
                    self.cfg.visibility_floor
                )
            )
        )

        next_visibility = self._clip01(
            next_visibility
        )

        return LocalThreatState(
            auth_pressure=float(
                next_pressure[0]
            ),
            scan_pressure=float(
                next_pressure[1]
            ),
            process_pressure=float(
                next_pressure[2]
            ),
            outbound_pressure=float(
                next_pressure[3]
            ),
            visibility=next_visibility,
        )

    def _rand_ip(
        self,
    ) -> str:
        return (
            f"10."
            f"{int(self._rng.randint(0, 256))}."
            f"{int(self._rng.randint(0, 256))}."
            f"{int(self._rng.randint(1, 255))}"
        )

    def _generate_events(
        self,
    ) -> List[Dict[str, Any]]:
        """
        根据 latent state + visibility
        生成 planner 可以看到的事件。

        visibility 影响“能看到多少异常”，
        而不是直接改变真实威胁。
        """

        state = self._latent_state

        pressures = (
            state.auth_pressure,
            state.scan_pressure,
            state.process_pressure,
            state.outbound_pressure,
        )

        visibility_factor = (
            0.15
            +
            0.85
            * float(state.visibility)
        )

        events: List[
            Dict[str, Any]
        ] = []

        for event_type, pressure in zip(
            _EVENT_TYPES,
            pressures,
        ):
            rate = (
                float(
                    self.cfg.event_rate
                )
                * float(pressure)
                * visibility_factor
            )

            count = int(
                self._rng.poisson(
                    lam=max(
                        rate,
                        0.0,
                    )
                )
            )

            count = min(
                count,
                int(
                    self.cfg
                    .max_events_per_type
                ),
            )

            for _ in range(
                count
            ):
                severity = (
                    float(pressure)
                    +
                    float(
                        self._rng.normal(
                            0.0,
                            0.06,
                        )
                    )
                )

                severity = self._clip01(
                    severity
                )

                events.append(
                    {
                        "type": event_type,
                        "severity": severity,
                        "src": self._rand_ip(),
                        "dst": self._rand_ip(),
                        "time": int(self.t),
                    }
                )

        return events

    def _risk_proxy_from_events(
        self,
        events: List[
            Dict[str, Any]
        ],
    ) -> float:
        """
        只根据可观测事件构造 risk_proxy。

        这里故意不读取真实 latent threat
        的平均值作为 risk_proxy，
        避免把隐藏环境状态直接泄漏给 planner。
        """

        if not events:
            return 0.0

        severities = np.array(
            [
                float(
                    event.get(
                        "severity",
                        0.0,
                    )
                )
                for event in events
            ],
            dtype=np.float64,
        )

        mean_severity = float(
            np.mean(
                severities
            )
        )

        max_severity = float(
            np.max(
                severities
            )
        )

        max_events = (
            len(_EVENT_TYPES)
            * int(
                self.cfg
                .max_events_per_type
            )
        )

        density = min(
            float(
                len(events)
            )
            /
            float(
                max_events
            ),
            1.0,
        )

        risk_proxy = (
            0.50
            * mean_severity
            +
            0.35
            * max_severity
            +
            0.15
            * density
        )

        return self._clip01(
            risk_proxy
        )

    def _make_observation(
        self,
    ) -> Dict[str, Any]:
        """
        构造外部可见 observation。

        不允许加入 latent state。
        """

        events = self._generate_events()

        risk_proxy = (
            self._risk_proxy_from_events(
                events
            )
        )

        return {
            "t": int(
                self.t
            ),
            "events": events,
            "risk_proxy": float(
                risk_proxy
            ),
            "region_id": int(
                self.cfg.region_id
            ),
            "region_name": (
                self.profile.name
            ),
            "meta": {
                "sim": (
                    "shared_local_online_gate_a"
                )
            },
        }

    def step(
        self,
        action_id: int,
    ) -> Tuple[
        Dict[str, Any],
        float,
        bool,
        Dict[str, Any],
    ]:
        """
        一个 transition：

        s_t
          -> 自然威胁演化
          -> 应用 action
          -> 生成新的 observation
          -> s_{t+1}

        当前 reward 固定为 0.0。
        Gate B 冻结正式 reward 后再接入。
        """

        if self.done:
            raise RuntimeError(
                "episode already finished; "
                "call reset() before step()"
            )

        action = get_action(
            action_id
        )

        evolved = self._natural_evolve(
            self._latent_state
        )

        self._latent_state = (
            self.action_adapter.apply(
                evolved,
                action.action_id,
            )
        )

        self.t += 1

        obs = self._make_observation()
        self._last_obs = obs

        self.done = (
            self.t
            >= int(
                self.cfg.max_steps
            )
        )

        # Gate B 前不定义正式 reward。
        reward = 0.0

        info = {
            "t": int(
                self.t
            ),
            "max_steps": int(
                self.cfg.max_steps
            ),
            "action_id": int(
                action.action_id
            ),
            "action_name": (
                action.name
            ),
            "reward_status": (
                "placeholder_zero_until_gate_b"
            ),
        }

        return (
            obs,
            float(reward),
            bool(self.done),
            info,
        )