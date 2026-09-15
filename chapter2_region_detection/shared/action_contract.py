from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Dict, Tuple


@dataclass(frozen=True)
class ActionContract:
    """
    LWM-RL / UG-CEM / CEM-APT
    正式公平比较共用的高层动作契约。

    action_id:
        离散 categorical ID。

    name:
        论文与代码统一动作名称。

    description:
        新小论文中的动作语义。

    cyborg_action:
        计划映射到的 CC4 底层动作类型。

        本层只冻结动作类型，
        host target 由 A3 adapter 负责解析。

    duration_ticks:
        CC4 动作持续时间 metadata。

    注意：
        action_id 只是类别索引，
        不表示连续强度或动作等级。
    """

    action_id: int
    name: str
    description: str
    cyborg_action: str
    duration_ticks: int


ACTION_CONTRACTS: Tuple[
    ActionContract,
    ...,
] = (
    ActionContract(
        action_id=0,
        name="no_op",
        description=(
            "no operation; "
            "observe without active intervention"
        ),
        cyborg_action="Sleep",
        duration_ticks=1,
    ),
    ActionContract(
        action_id=1,
        name="analyse",
        description=(
            "intrusion investigation"
        ),
        cyborg_action="Analyse",
        duration_ticks=2,
    ),
    ActionContract(
        action_id=2,
        name="remove",
        description=(
            "user-level compromise removal"
        ),
        cyborg_action="Remove",
        duration_ticks=3,
    ),
    ActionContract(
        action_id=3,
        name="restore",
        description=(
            "host reimaging"
        ),
        cyborg_action="Restore",
        duration_ticks=5,
    ),
)


def _validate_contracts() -> None:
    """
    模块加载时验证动作契约本身。

    防止后续修改时出现：
    - 重复 ID；
    - 重复名称；
    - ID 不连续；
    - 非法 duration。
    """

    ids = [
        action.action_id
        for action in ACTION_CONTRACTS
    ]

    names = [
        action.name
        for action in ACTION_CONTRACTS
    ]

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "duplicate action_id "
            "in ACTION_CONTRACTS"
        )

    if len(names) != len(set(names)):
        raise RuntimeError(
            "duplicate action name "
            "in ACTION_CONTRACTS"
        )

    expected_ids = list(
        range(
            len(ACTION_CONTRACTS)
        )
    )

    if ids != expected_ids:
        raise RuntimeError(
            "action IDs must be "
            "contiguous categorical IDs "
            f"{expected_ids}; got {ids}"
        )

    for action in ACTION_CONTRACTS:
        if (
            not isinstance(
                action.duration_ticks,
                Integral,
            )
            or isinstance(
                action.duration_ticks,
                bool,
            )
            or action.duration_ticks <= 0
        ):
            raise RuntimeError(
                "duration_ticks must be "
                "a positive integer: "
                f"{action}"
            )


_validate_contracts()


N_ACTIONS = len(
    ACTION_CONTRACTS
)


ID2ACTION: Dict[
    int,
    ActionContract,
] = {
    action.action_id: action
    for action in ACTION_CONTRACTS
}


NAME2ID: Dict[
    str,
    int,
] = {
    action.name: action.action_id
    for action in ACTION_CONTRACTS
}


def get_action(
    action_id: int,
) -> ActionContract:
    """
    根据 categorical action ID
    获取统一动作定义。
    """

    if (
        not isinstance(
            action_id,
            Integral,
        )
        or isinstance(
            action_id,
            bool,
        )
    ):
        raise ValueError(
            "action_id must be "
            "an integer categorical ID"
        )

    action_id = int(
        action_id
    )

    if action_id not in ID2ACTION:
        raise ValueError(
            f"invalid action_id="
            f"{action_id}; "
            f"expected 0.."
            f"{N_ACTIONS - 1}"
        )

    return ID2ACTION[
        action_id
    ]


def get_action_id(
    name: str,
) -> int:
    """
    根据统一动作名称获取 categorical ID。
    """

    if name not in NAME2ID:
        raise ValueError(
            f"unknown action name: "
            f"{name}"
        )

    return NAME2ID[
        name
    ]


def get_cyborg_action_type(
    action_id: int,
) -> str:
    """
    获取计划映射到的 CC4 动作类型。

    host target 由后续 A3 adapter
    负责解析。
    """

    return get_action(
        action_id
    ).cyborg_action


def get_action_duration(
    action_id: int,
) -> int:
    """
    获取当前动作对应的 CC4 duration metadata。
    """

    return get_action(
        action_id
    ).duration_ticks