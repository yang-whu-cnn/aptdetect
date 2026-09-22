from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

from .reward import caics_reward


class ConditionalNodeModel(Protocol):
    def fit(self, parents, targets) -> Any: ...
    def predict(self, parents) -> Any: ...


DAG_PARENTS = {
    "action": ("observable_evidence",),
    "target_role": ("observable_evidence", "action"),
    "post_incident_proxy": ("pre_incident_proxy", "observable_evidence", "action", "target_role"),
    "service_failure": ("post_incident_proxy", "action"),
    "service_availability": ("post_incident_proxy", "service_failure", "action"),
    "reward": ("pre_incident_proxy", "post_incident_proxy", "action", "target_role",
               "service_failure", "service_availability"),
}


@dataclass(frozen=True)
class SCMRecord:
    observable_evidence: float
    action: int
    target_role: int
    pre_incident_proxy: int
    post_incident_proxy: int
    service_failure: int
    service_availability: float
    reward: float


class CausalRewardSCM:
    """Training-only causal reward facade with injectable linear/forest nodes."""
    def __init__(self, node_models: dict[str, ConditionalNodeModel] | None = None):
        self.node_models = dict(node_models or {}); self.fitted = False

    def fit(self, records: list[SCMRecord]) -> None:
        if not records: raise ValueError("SCM requires train records")
        for name, model in self.node_models.items():
            parents = [[getattr(row, field) for field in DAG_PARENTS[name]] for row in records]
            model.fit(parents, [getattr(row, name) for row in records])
        self.fitted = True

    def intervene_action(self, record: SCMRecord, action: int) -> SCMRecord:
        if not self.fitted: raise RuntimeError("SCM must be fitted before intervention")
        reward = caics_reward(active_incidents=record.post_incident_proxy,
                              incident_change=record.post_incident_proxy-record.pre_incident_proxy,
                              action_id=action)
        return replace(record, action=action, reward=reward)
