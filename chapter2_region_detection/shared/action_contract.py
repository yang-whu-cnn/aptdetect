from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ActionContract:
    """
    第二章正式公平比较使用的统一高层动作契约。

    这里定义的是“论文层面的动作语义”，
    与旧版 src/action_space.py 分离。

    action_id:
        离散类别 ID。

    name:
        论文和代码中统一使用的动作名称。

    description:
        论文动作语义。

    cyborg_action:
        CybORG / CC4 中计划映射到的
        底层动作类型。

        这里只冻结动作类型名称；
        host / subnet 等具体参数由后续
        adapter 负责选择。

    duration_ticks:
        对应官方 CC4 动作持续时间。

    注意：
        action_id 只是类别索引，
        不能把 ID 大小解释为连续动作强度。
    """

    action_id: int
    name: str
    description: str
    cyborg_action: str
    duration_ticks: int


ACTION_CONTRACTS: Tuple[
    ActionContract,
    ...
] = (
    ActionContract(
        action_id=0,
        name="no_op",
        description=(
            "no operation / monitor; "
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
        name="control_traffic",
        description=(
            "traffic blocking"
        ),
        cyborg_action="BlockTraffic",
        duration_ticks=1,
    ),
    ActionContract(
        action_id=3,
        name="remove",
        description=(
            "user-level compromise removal"
        ),
        cyborg_action="Remove",
        duration_ticks=3,
    ),
    ActionContract(
        action_id=4,
        name="restore",
        description=(
            "host reimaging"
        ),
        cyborg_action="Restore",
        duration_ticks=5,
    ),
)


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
    根据动作 ID 获取统一动作定义。
    """

    action_id = int(action_id)

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
    根据统一动作名称获得 ID。
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
    获取该高层动作对应的
    CybORG 动作类型。

    注意：
    这里只返回类型，
    不负责 host/subnet 参数选择。
    """

    return get_action(
        action_id
    ).cyborg_action


def get_action_duration(
    action_id: int,
) -> int:
    """
    获取 CC4 对应动作持续时间。
    """

    return get_action(
        action_id
    ).duration_ticks