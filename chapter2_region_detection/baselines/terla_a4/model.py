from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .graph import HeteroGraph


NODE_TYPES = ("mission", "subnet", "host")
RELATIONS = (
    ("mission", "contains", "subnet"),
    ("subnet", "in_mission", "mission"),
    ("subnet", "contains", "host"),
    ("host", "in_subnet", "subnet"),
)


class TypedAttentionLayer(nn.Module):
    """Auditable pure-torch HGT-style relation/type-specific attention."""

    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.query = nn.ModuleDict({t: nn.Linear(hidden, hidden, bias=False) for t in NODE_TYPES})
        self.key = nn.ModuleDict({t: nn.Linear(hidden, hidden, bias=False) for t in NODE_TYPES})
        self.value = nn.ModuleDict({t: nn.Linear(hidden, hidden, bias=False) for t in NODE_TYPES})
        self.relation = nn.ParameterDict({
            "__".join(r): nn.Parameter(torch.eye(hidden)) for r in RELATIONS})
        self.self_projection = nn.ModuleDict({t: nn.Linear(hidden, hidden) for t in NODE_TYPES})

    def forward(self, x: dict[str, torch.Tensor],
                edges: dict[tuple[str, str, str], torch.Tensor]) -> dict[str, torch.Tensor]:
        aggregated = {t: torch.zeros_like(x[t]) for t in NODE_TYPES}
        for relation in RELATIONS:
            src_type, _name, dst_type = relation
            edge = edges[relation]
            if edge.numel() == 0:
                continue
            src, dst = edge[0], edge[1]
            q = self.query[dst_type](x[dst_type][dst])
            k = self.key[src_type](x[src_type][src]) @ self.relation["__".join(relation)]
            v = self.value[src_type](x[src_type][src])
            logits = (q * k).sum(-1) / math.sqrt(self.hidden)
            for destination in torch.unique(dst):
                mask = dst == destination
                weights = torch.softmax(logits[mask], dim=0)
                aggregated[dst_type][destination] += (weights[:, None] * v[mask]).sum(0)
        return {t: torch.relu(self.self_projection[t](x[t]) + aggregated[t]) for t in NODE_TYPES}


class TERLAPolicy(nn.Module):
    """Two typed-attention layers, ReLU, host sum pooling, PPO MLP [120,120]."""

    def __init__(self, hidden: int = 60):
        super().__init__()
        if hidden != 60:
            raise ValueError("TERLA-A4 hidden size is frozen at (2 host features + 4 A4)*10 = 60")
        self.hidden = hidden
        self.input_projection = nn.ModuleDict({
            "mission": nn.Linear(3, hidden), "subnet": nn.Linear(1, hidden),
            "host": nn.Linear(2, hidden),
        })
        self.hgt1 = TypedAttentionLayer(hidden)
        self.hgt2 = TypedAttentionLayer(hidden)
        self.trunk = nn.Sequential(nn.Linear(hidden, 120), nn.ReLU(),
                                   nn.Linear(120, 120), nn.ReLU())
        self.actor = nn.Linear(120, 4)
        self.critic = nn.Linear(120, 1)

    def encode(self, graph: HeteroGraph) -> torch.Tensor:
        x = {t: torch.relu(self.input_projection[t](graph.node_features[t])) for t in NODE_TYPES}
        x = self.hgt1(x, graph.edges)
        x = self.hgt2(x, graph.edges)
        # Paper pooling is global sum over host features. Empty graphs map to zero.
        return x["host"].sum(dim=0) if len(x["host"]) else x["host"].new_zeros(self.hidden)

    def forward(self, graph: HeteroGraph) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.trunk(self.encode(graph))
        return self.actor(latent), self.critic(latent).squeeze(-1)


@dataclass
class AgentContext:
    decisions: int = 0
    last_action: int | None = None


class TERLASharedController:
    """One shared policy object; per-agent runtime context never aliases."""

    def __init__(self, policy: TERLAPolicy | None = None):
        self.policy = policy or TERLAPolicy()
        self._contexts: dict[str, AgentContext] = {}

    def context(self, agent: str) -> AgentContext:
        if agent not in self._contexts:
            self._contexts[agent] = AgentContext()
        return self._contexts[agent]

    def record_action(self, agent: str, action: int) -> None:
        context = self.context(agent)
        context.decisions += 1
        context.last_action = int(action)
