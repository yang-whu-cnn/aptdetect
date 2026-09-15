from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

from shared.action_contract import get_action


@dataclass(frozen=True)
class LocalThreatState:
    """
    local-online simulator 内部使用的隐状态。

    注意：
    这些变量不能直接暴露给 planner。

    planner 仍然只能看到统一的 8 维
    StateSummary。
    """

    auth_pressure: float
    scan_pressure: float
    process_pressure: float
    outbound_pressure: float
    visibility: float


@dataclass(frozen=True)
class LocalActionAdapterConfig:
    """
    A2 开发期动作效果参数。

    这些数值只用于 local-online 联调，
    不是论文最终 reward 参数。
    """

    analyse_visibility_gain: float = 0.35

    control_scan_multiplier: float = 0.70
    control_outbound_multiplier: float = 0.25

    remove_auth_multiplier: float = 0.25
    remove_process_multiplier: float = 0.60

    restore_auth_multiplier: float = 0.10
    restore_process_multiplier: float = 0.10
    restore_outbound_multiplier: float = 0.40


class LocalActionAdapter:
    """
    将 Gate A1 的统一高层动作
    映射成 local simulator 中的
    transition semantics。

    本模块：

    - 不计算 reward；
    - 不依赖 planner；
    - 不依赖 CEM；
    - 不依赖 PPO；
    - 不依赖 uncertainty；
    - 不把 action ID 当连续强度。
    """

    def __init__(
        self,
        cfg: LocalActionAdapterConfig | None = None,
    ):
        self.cfg = (
            cfg
            if cfg is not None
            else LocalActionAdapterConfig()
        )

        self._validate_config()

    @staticmethod
    def _clip01(
        value: float,
    ) -> float:
        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )

    def _validate_config(
        self,
    ) -> None:
        values = {
            "analyse_visibility_gain":
                self.cfg.analyse_visibility_gain,

            "control_scan_multiplier":
                self.cfg.control_scan_multiplier,

            "control_outbound_multiplier":
                self.cfg.control_outbound_multiplier,

            "remove_auth_multiplier":
                self.cfg.remove_auth_multiplier,

            "remove_process_multiplier":
                self.cfg.remove_process_multiplier,

            "restore_auth_multiplier":
                self.cfg.restore_auth_multiplier,

            "restore_process_multiplier":
                self.cfg.restore_process_multiplier,

            "restore_outbound_multiplier":
                self.cfg.restore_outbound_multiplier,
        }

        for name, value in values.items():
            if not isfinite(
                float(value)
            ):
                raise ValueError(
                    f"{name} must be finite"
                )

        if not (
            0.0
            <= self.cfg.analyse_visibility_gain
            <= 1.0
        ):
            raise ValueError(
                "analyse_visibility_gain "
                "must be in [0, 1]"
            )

        for name, value in values.items():
            if name == (
                "analyse_visibility_gain"
            ):
                continue

            if not (
                0.0
                <= float(value)
                <= 1.0
            ):
                raise ValueError(
                    f"{name} must be in [0, 1]"
                )

    def _validate_state(
        self,
        state: LocalThreatState,
    ) -> None:
        values = (
            state.auth_pressure,
            state.scan_pressure,
            state.process_pressure,
            state.outbound_pressure,
            state.visibility,
        )

        if not all(
            isfinite(float(v))
            for v in values
        ):
            raise ValueError(
                "local threat state "
                "must be finite"
            )

        if not all(
            0.0 <= float(v) <= 1.0
            for v in values
        ):
            raise ValueError(
                "local threat state values "
                "must be in [0, 1]"
            )

    def apply(
        self,
        state: LocalThreatState,
        action_id: int,
    ) -> LocalThreatState:
        """
        对当前 local latent state
        应用一次高层动作效果。

        这里只处理 action effect。

        threat 的自然增长、随机攻击过程、
        observation generation 等由后续
        local environment 负责。
        """

        self._validate_state(
            state
        )

        action = get_action(
            action_id
        )

        name = action.name

        # --------------------------------
        # 0. no_op
        # --------------------------------
        if name == "no_op":
            return state

        # --------------------------------
        # 1. analyse
        #
        # 调查提高可见性，
        # 不直接清除威胁。
        # --------------------------------
        if name == "analyse":
            return replace(
                state,
                visibility=self._clip01(
                    state.visibility
                    +
                    self.cfg
                    .analyse_visibility_gain
                ),
            )

        # --------------------------------
        # 2. control_traffic
        #
        # 主要作用于网络活动。
        # --------------------------------
        if name == "control_traffic":
            return replace(
                state,

                scan_pressure=self._clip01(
                    state.scan_pressure
                    *
                    self.cfg
                    .control_scan_multiplier
                ),

                outbound_pressure=self._clip01(
                    state.outbound_pressure
                    *
                    self.cfg
                    .control_outbound_multiplier
                ),
            )

        # --------------------------------
        # 3. remove
        #
        # 用户级失陷清除：
        # 主要针对认证/会话，
        # 对进程级失陷提供部分缓解。
        # --------------------------------
        if name == "remove":
            return replace(
                state,

                auth_pressure=self._clip01(
                    state.auth_pressure
                    *
                    self.cfg
                    .remove_auth_multiplier
                ),

                process_pressure=self._clip01(
                    state.process_pressure
                    *
                    self.cfg
                    .remove_process_multiplier
                ),
            )

        # --------------------------------
        # 4. restore
        #
        # host reimaging：
        # 对主机级失陷进行强恢复。
        #
        # 不清除 scan_pressure，
        # 因为外部攻击者仍可继续扫描。
        # --------------------------------
        if name == "restore":
            return replace(
                state,

                auth_pressure=self._clip01(
                    state.auth_pressure
                    *
                    self.cfg
                    .restore_auth_multiplier
                ),

                process_pressure=self._clip01(
                    state.process_pressure
                    *
                    self.cfg
                    .restore_process_multiplier
                ),

                outbound_pressure=self._clip01(
                    state.outbound_pressure
                    *
                    self.cfg
                    .restore_outbound_multiplier
                ),
            )

        raise RuntimeError(
            f"unhandled action name: "
            f"{name}"
        )