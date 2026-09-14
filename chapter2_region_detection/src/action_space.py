# src/action_space.py
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ActionDef:
    """离散高层动作定义（单区域）"""
    name: str
    intensity: int  # 用于“动作激进程度”判断（误报风险惩罚）
    cost: float     # 用于动作开销惩罚


# 统一动作空间（第二章：单区域检测）
# 说明：真实CC4环境的动作接口可能不同，你只需要在 OnlineCC4Client 内做映射
ACTION_SPACE: List[ActionDef] = [
    ActionDef("monitor", intensity=0, cost=0.05),        # 继续监测
    ActionDef("light_evidence", intensity=1, cost=0.20), # 轻量补证
    ActionDef("heavy_evidence", intensity=2, cost=0.45), # 重度补证
    ActionDef("local_mitigate", intensity=3, cost=0.75), # 局部处置
    ActionDef("strong_mitigate", intensity=4, cost=1.10) # 强处置
]

NAME2ID: Dict[str, int] = {a.name: i for i, a in enumerate(ACTION_SPACE)}


def action_cost(action_id: int) -> float:
    return ACTION_SPACE[action_id].cost


def action_intensity(action_id: int) -> int:
    return ACTION_SPACE[action_id].intensity
