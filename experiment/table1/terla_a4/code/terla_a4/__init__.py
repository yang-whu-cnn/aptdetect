"""TERLA-A4 structural adaptation; production reward/training remains gated."""

from .graph import HeteroGraph, TERLAGraphBuilder
from .model import TERLAPolicy, TERLASharedController
from .reward import (CC4HiddenRewardAdapter, HiddenRewardRecord,
                     TERLARewardChannel, TERLAProductionGate)
from .targeting import TargetDecision, resolve_terla_action

__all__ = [
    "HeteroGraph", "TERLAGraphBuilder", "TERLAPolicy", "TERLASharedController",
    "CC4HiddenRewardAdapter", "HiddenRewardRecord", "TERLARewardChannel", "TERLAProductionGate",
    "TargetDecision", "resolve_terla_action",
]
