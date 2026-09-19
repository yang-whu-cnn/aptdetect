from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Hashable, Iterable

from .adapter import AlertToken, DCAConfig


def _distance(a: AlertToken, b: AlertToken, *, time_window: int) -> float:
    """Auditable mixed alert distance in [0,1].

    Host identity is the dominant physical feature; evidence type is the cyber
    feature; time is bounded by the configured temporal window.
    """
    host = 0.0 if a.host == b.host else 0.6
    kind = 0.0 if a.evidence_type == b.evidence_type else 0.1
    time = 0.3 * min(abs(a.tick - b.tick) / float(time_window), 1.0)
    return host + kind + time


def cluster_alerts(tokens: Iterable[AlertToken], config: DCAConfig) -> tuple[tuple[AlertToken, ...], ...]:
    """Deterministic DBSCAN over Blue-visible alert tokens, ordered temporally."""
    points = tuple(sorted(tokens))
    n = len(points)
    if n == 0:
        return ()
    neighbours = [
        tuple(j for j, other in enumerate(points)
              if _distance(point, other, time_window=config.time_window) <= config.epsilon)
        for point in points
    ]
    labels: list[int | None] = [None] * n
    cluster_id = 0
    for i in range(n):
        if labels[i] is not None:
            continue
        if len(neighbours[i]) < config.min_pts:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        queue = list(neighbours[i])
        seen = set(queue)
        while queue:
            j = queue.pop(0)
            if labels[j] == -1:
                labels[j] = cluster_id
            if labels[j] is not None:
                continue
            labels[j] = cluster_id
            if len(neighbours[j]) >= config.min_pts:
                for candidate in neighbours[j]:
                    if candidate not in seen:
                        seen.add(candidate)
                        queue.append(candidate)
        cluster_id += 1
    clusters = []
    for cid in range(cluster_id):
        clusters.append(tuple(points[i] for i, label in enumerate(labels) if label == cid))
    return tuple(clusters)


@dataclass(frozen=True)
class PathInference:
    state: Hashable
    attacker_action: Hashable | None
    value: float


class TabularAttackPathModel:
    """Finite deterministic model with value iteration and Dyna replay updates."""

    def __init__(self, *, gamma: float = 0.95, dyna_steps: int = 20, seed: int = 0):
        if not 0 <= gamma < 1 or dyna_steps < 0:
            raise ValueError("invalid path-model gamma or Dyna steps")
        self.gamma = float(gamma)
        self.dyna_steps = int(dyna_steps)
        self.rng = random.Random(int(seed))
        self.model: dict[tuple[Hashable, Hashable], tuple[Hashable, float]] = {}
        self.values: dict[Hashable, float] = {}
        self.q_values: dict[tuple[Hashable, Hashable], float] = {}

    def update(self, state: Hashable, attacker_action: Hashable, next_state: Hashable,
               *, service_impact: float, n_steps: int, objective_reached: bool,
               objective_bonus: float = 10.0) -> float:
        if n_steps <= 0 or not 0 <= float(service_impact) <= 1:
            raise ValueError("path reward inputs are outside their finite contract")
        reward = float(service_impact) + 1.0 / int(n_steps)
        if objective_reached:
            reward += float(objective_bonus)
        self.model[(state, attacker_action)] = (next_state, reward)
        self.values.setdefault(state, 0.0); self.values.setdefault(next_state, 0.0)
        self._backup(state, attacker_action)
        keys = tuple(sorted(self.model, key=lambda item: (repr(item[0]), repr(item[1]))))
        for _ in range(self.dyna_steps):
            sampled = keys[self.rng.randrange(len(keys))]
            self._backup(*sampled)
        return reward

    def _backup(self, state: Hashable, action: Hashable) -> float:
        next_state, reward = self.model[(state, action)]
        q = reward + self.gamma * self.values.get(next_state, 0.0)
        self.q_values[(state, action)] = q
        candidates = [v for (s, _a), v in self.q_values.items() if s == state]
        self.values[state] = max(candidates)
        return q

    def value_iteration(self, *, tolerance: float = 1e-10, max_iterations: int = 10000) -> int:
        if tolerance <= 0 or max_iterations <= 0:
            raise ValueError("invalid convergence controls")
        for iteration in range(1, max_iterations + 1):
            before = dict(self.values)
            for state, action in sorted(self.model, key=lambda item: (repr(item[0]), repr(item[1]))):
                self._backup(state, action)
            delta = max((abs(self.values[s] - before.get(s, 0.0)) for s in self.values), default=0.0)
            if delta <= tolerance:
                return iteration
        raise RuntimeError("attack path value iteration did not converge")

    def infer(self, state: Hashable) -> PathInference:
        choices = [(a, q) for (s, a), q in self.q_values.items() if s == state]
        if not choices:
            return PathInference(state, None, 0.0)
        action, value = max(choices, key=lambda item: (item[1], repr(item[0])))
        return PathInference(state, action, float(value))


def cluster_state(cluster: Iterable[AlertToken]) -> tuple[str, tuple[str, ...]]:
    alerts = tuple(cluster)
    if not alerts:
        raise ValueError("cluster cannot be empty")
    return alerts[0].host, tuple(sorted({item.evidence_type for item in alerts}))
