from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import (
    Mapping,
    Optional,
)

from shared.action_contract import (
    get_action,
)

from shared.cyborg_target_resolver import (
    ValidCyborgAction,
    resolve_sleep_action,
    resolve_target_action,
)


@dataclass(frozen=True)
class CybORGActionResolution:
    """
    一次 high-level action
    到 CC4 low-level action 的解析结果。

    同时保留 requested / executed 信息，
    后续可以直接进入：

    - formal replay；
    - fallback rate；
    - valid-target rate；
    - action distribution；
    - debugging / audit。
    """

    agent_name: str

    requested_action_id: int
    requested_action_name: str
    requested_cyborg_family: str

    executed_index: int
    executed_label: str
    executed_action_family: str

    target_host: Optional[str]

    fallback: bool
    fallback_reason: Optional[str]


class CybORGActionAdapter:
    """
    A3 正式共享四动作 CC4 adapter。

    后续：
        Ours
        UG-CEM
        CEM

    必须共用该 adapter + resolver。

    本阶段只负责：

        high-level categorical action
                    ↓
        wrapper action_labels + action_mask
                    ↓
        deterministic target resolution
                    ↓
        low-level action index

    本模块明确不负责：

    - reward；
    - formal state encoding；
    - world model；
    - PPO；
    - multi-tick decision epoch；
    - hidden compromise detection。
    """

    @staticmethod
    def _snapshot(
        env,
        agent_name: str,
    ) -> tuple[
        list[object],
        list[object],
    ]:
        """
        每次 resolve 时重新读取当前 wrapper。

        不跨 reset / seed 永久缓存，
        因为 A3.1 已确认不同 seed 下
        valid host 数会变化。
        """

        labels = list(
            env.action_labels(
                agent_name
            )
        )

        mask = list(
            env.action_mask(
                agent_name
            )
        )

        if len(labels) != len(mask):
            raise ValueError(
                f"{agent_name}: "
                "action_labels/action_mask "
                "length mismatch"
            )

        if not labels:
            raise ValueError(
                f"{agent_name}: "
                "empty action space"
            )

        return (
            labels,
            mask,
        )

    @staticmethod
    def _build_result(
        *,
        agent_name: str,
        requested_action_id: int,
        requested_action_name: str,
        requested_cyborg_family: str,
        selected: ValidCyborgAction,
        fallback: bool,
        fallback_reason: Optional[str],
    ) -> CybORGActionResolution:

        return CybORGActionResolution(
            agent_name=str(
                agent_name
            ),

            requested_action_id=int(
                requested_action_id
            ),

            requested_action_name=str(
                requested_action_name
            ),

            requested_cyborg_family=str(
                requested_cyborg_family
            ),

            executed_index=int(
                selected.index
            ),

            executed_label=str(
                selected.label
            ),

            executed_action_family=str(
                selected.family
            ),

            target_host=(
                selected.target_host
            ),

            fallback=bool(
                fallback
            ),

            fallback_reason=(
                fallback_reason
            ),
        )

    def resolve(
        self,
        env,
        agent_name: str,
        action_id: int,
        observable_host_scores: Mapping[
            str,
            Real,
        ] | None = None,
    ) -> CybORGActionResolution:
        """
        将统一 categorical action
        映射到当前真实 CC4 wrapper index。

        targeted action 的 host score
        必须来自当前 Blue observable evidence。

        如果：
            Analyse / Remove / Restore

        当前没有合法 observable target：

            fallback -> valid Sleep
        """

        action = get_action(
            action_id
        )

        labels, mask = (
            self._snapshot(
                env,
                agent_name,
            )
        )

        # =====================================================
        # A3.2
        #
        # 0 = no_op
        #     -> explicit Sleep
        #
        # 不能：
        # - 映射 Monitor；
        # - 写死 index=49；
        # - 写死 index=145。
        # =====================================================
        if (
            action.cyborg_action
            == "Sleep"
        ):
            selected = (
                resolve_sleep_action(
                    labels,
                    mask,
                )
            )

            return self._build_result(
                agent_name=(
                    agent_name
                ),
                requested_action_id=(
                    action.action_id
                ),
                requested_action_name=(
                    action.name
                ),
                requested_cyborg_family=(
                    action.cyborg_action
                ),
                selected=selected,
                fallback=False,
                fallback_reason=None,
            )

        # =====================================================
        # A3.3
        #
        # Analyse / Remove / Restore
        # 共用同一个 deterministic resolver。
        # =====================================================
        selected = (
            resolve_target_action(
                family=(
                    action.cyborg_action
                ),
                labels=labels,
                mask=mask,
                observable_host_scores=(
                    observable_host_scores
                ),
            )
        )

        if selected is not None:
            return self._build_result(
                agent_name=(
                    agent_name
                ),
                requested_action_id=(
                    action.action_id
                ),
                requested_action_name=(
                    action.name
                ),
                requested_cyborg_family=(
                    action.cyborg_action
                ),
                selected=selected,
                fallback=False,
                fallback_reason=None,
            )

        # =====================================================
        # 无合法 observable target
        # -> fallback Sleep
        # =====================================================
        sleep = resolve_sleep_action(
            labels,
            mask,
        )

        return self._build_result(
            agent_name=(
                agent_name
            ),
            requested_action_id=(
                action.action_id
            ),
            requested_action_name=(
                action.name
            ),
            requested_cyborg_family=(
                action.cyborg_action
            ),
            selected=sleep,
            fallback=True,
            fallback_reason=(
                "no_valid_observable_target"
            ),
        )