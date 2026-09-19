from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class UncertaintyTriple:
    world_model: float
    progress: float
    prior: float

    def weighted(self, weights: tuple[float, float, float]) -> float:
        values = (self.world_model, self.progress, self.prior)
        if any(not math.isfinite(x) or x < 0 for x in (*values, *weights)):
            raise ValueError("uncertainties and weights must be finite and non-negative")
        return float(sum(w * x for w, x in zip(weights, values)))


@dataclass
class ActionStatistics:
    n: int = 0
    w: float = 0.0
    q: float = 0.0
    prior: float = 0.0
    uncertainty: UncertaintyTriple = field(
        default_factory=lambda: UncertaintyTriple(0.0, 0.0, 0.0))

    def backup(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError("backup value must be finite")
        self.n += 1
        self.w += float(value)
        self.q = self.w / self.n


def hybrid_score(*, q: float, uncertainty: float, prior: float, node_visits: int,
                 action_visits: int, c_puct: float, rho: float, c_u: float) -> float:
    """Paper Eq. 5, with already-scheduled rho(phi) and c_u(phi)."""
    values = (q, uncertainty, prior, c_puct, rho, c_u)
    if any(not math.isfinite(x) for x in values):
        raise ValueError("hybrid score inputs must be finite")
    if uncertainty < 0 or prior < 0 or node_visits < 0 or action_visits < 0:
        raise ValueError("hybrid score counts/probabilities/uncertainty must be non-negative")
    return float(
        q - rho * uncertainty
        + c_puct * prior * math.sqrt(node_visits) / (1 + action_visits)
        + c_u * uncertainty
    )


@dataclass
class SearchNode:
    state: tuple[float, ...]
    n: int = 0
    actions: dict[int, ActionStatistics] = field(default_factory=dict)
    children: dict[int, "SearchNode"] = field(default_factory=dict)
    elapsed_ticks: int = 0

    def backup(self, action: int, value: float) -> None:
        self.actions[action].backup(value)
        self.n += 1

    def select(self, *, c_puct: float, rho: float, c_u: float,
               uncertainty_weights: tuple[float, float, float]) -> int:
        if not self.actions:
            raise RuntimeError("cannot select from an unexpanded node")
        scored = []
        for action, stats in self.actions.items():
            u = stats.uncertainty.weighted(uncertainty_weights)
            score = hybrid_score(q=stats.q, uncertainty=u, prior=stats.prior,
                                 node_visits=self.n, action_visits=stats.n,
                                 c_puct=c_puct, rho=rho, c_u=c_u)
            scored.append((score, stats.prior, -action, action))
        return max(scored)[-1]
