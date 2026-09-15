from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

from shared.action_contract import get_action


@dataclass(frozen=True)
class LocalThreatState:
    """
    local-online simulator 内部使用的隐状态。

    这些变量不能直接暴露给 planner。
    """

    auth_pressure: float
    scan_pressure: float
    process_pressure: float
    outbound_pressure: float
    visibility: float


@dataclass(frozen=True)
class LocalActionAdapterConfig:
    """
    A2R 开发期动作效果参数。

    这些参数只用于 local-online 联调，
    不是正式论文 reward 参数。
    """

    analyse_visibility_gain: float = 0.35

    remove_auth_multiplier: float = 0.25
    remove_process_multiplier: float = 0.60

    restore_auth_multiplier: float = 0.10
    restore_process_multiplier: float = 0.10
    restore_outbound_multiplier: float = 0.40


class LocalActionAdapter:
    """
    将四类统一高层动作映射到
    local simulator transition semantics。

    本模块：
    - 不计算 reward；
    - 不依赖 CEM / PPO / uncertainty；
    - 不把 action ID 当连续强度；
    - 不包含已删除的 control_traffic。
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
        对 local latent state 应用一次
        高层动作语义。
        """

        self._validate_state(
            state
        )

        action = get_action(
            action_id
        )

        name = action.name

        # 0. no_op
        if name == "no_op":
            return state

        # 1. analyse
        #
        # 调查提升可见性，
        # 不直接消除威胁。
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

        # 2. remove
        #
        # 用户级 compromise removal。
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

        # 3. restore
        #
        # host reimaging。
        #
        # 不直接消除 scan_pressure，
        # 因为外部扫描源仍可能存在。
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