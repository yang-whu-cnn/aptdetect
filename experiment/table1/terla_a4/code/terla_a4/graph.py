from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


FORBIDDEN_POLICY_FIELDS = frozenset({
    "compromised", "true_compromise", "red_session", "red_sessions",
    "ground_truth", "infected", "escalated",
})


@dataclass(frozen=True)
class HeteroGraph:
    node_features: dict[str, torch.Tensor]
    edges: dict[tuple[str, str, str], torch.Tensor]
    host_names: tuple[str, ...]
    host_scores: torch.Tensor

    def to(self, device: torch.device | str) -> "HeteroGraph":
        return HeteroGraph(
            {name: value.to(device) for name, value in self.node_features.items()},
            {name: value.to(device) for name, value in self.edges.items()},
            self.host_names,
            self.host_scores.to(device),
        )


def _reject_hidden(value: Any) -> None:
    if isinstance(value, Mapping):
        overlap = FORBIDDEN_POLICY_FIELDS.intersection(str(key).lower() for key in value)
        if overlap:
            raise ValueError(f"hidden truth fields are forbidden in TERLA graph: {sorted(overlap)}")
        for nested in value.values():
            _reject_hidden(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_hidden(nested)


def _visible_event(data: Mapping[str, Any], normalized: str, raw: str) -> float:
    if normalized in data:
        return float(bool(data[normalized]))
    rows = data.get(raw, ())
    return float(isinstance(rows, (list, tuple)) and len(rows) > 0)


class TERLAGraphBuilder:
    """Build mission/subnet/host graph solely from a Blue-visible observation."""

    def build(self, observation: Mapping[str, Any], *, mission_phase: int,
              default_subnet: str = "local") -> HeteroGraph:
        _reject_hidden(observation)
        if mission_phase not in (0, 1, 2):
            raise ValueError("mission phase must be 0, 1, or 2")
        nested = observation.get("observation")
        hosts = nested if isinstance(nested, Mapping) else observation
        rows: list[tuple[str, str, float, float]] = []
        for raw_host, data in hosts.items():
            if str(raw_host).lower() in {"success", "mission_phase"} or not isinstance(data, Mapping):
                continue
            host = str(data.get("hostname", raw_host))
            subnet = str(data.get("subnet", default_subnet))
            process = _visible_event(data, "malicious_process_event", "Processes")
            network = _visible_event(data, "malicious_network_event", "Connections")
            rows.append((host, subnet, process, network))
        # Canonical order makes the graph invariant to observation dictionary order.
        rows.sort(key=lambda item: (item[1], item[0]))
        subnet_names = tuple(sorted({item[1] for item in rows} or {default_subnet}))
        subnet_index = {name: i for i, name in enumerate(subnet_names)}
        mission = torch.zeros((1, 3), dtype=torch.float32)
        mission[0, mission_phase] = 1.0
        subnet_features = torch.ones((len(subnet_names), 1), dtype=torch.float32)
        host_features = torch.tensor([[p, n] for _h, _s, p, n in rows], dtype=torch.float32)
        if not rows:
            host_features = torch.empty((0, 2), dtype=torch.float32)
        host_scores = torch.tensor([2 * p + n for _h, _s, p, n in rows], dtype=torch.float32)

        ms = torch.tensor([[0, i] for i in range(len(subnet_names))], dtype=torch.long).T
        sh = torch.tensor([[subnet_index[s], i] for i, (_h, s, _p, _n) in enumerate(rows)],
                          dtype=torch.long).T
        if not rows:
            sh = torch.empty((2, 0), dtype=torch.long)
        edges = {
            ("mission", "contains", "subnet"): ms,
            ("subnet", "in_mission", "mission"): ms.flip(0),
            ("subnet", "contains", "host"): sh,
            ("host", "in_subnet", "subnet"): sh.flip(0),
        }
        return HeteroGraph(
            {"mission": mission, "subnet": subnet_features, "host": host_features},
            edges, tuple(item[0] for item in rows), host_scores)
