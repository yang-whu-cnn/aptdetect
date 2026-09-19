from __future__ import annotations

from dataclasses import dataclass

from .graph import HeteroGraph


DURATIONS = (1, 2, 3, 5)


@dataclass(frozen=True)
class TargetDecision:
    requested_action: int
    executed_action: int
    target: str | None
    duration: int
    reason: str


def resolve_terla_action(requested_action: int, graph: HeteroGraph) -> TargetDecision:
    action = int(requested_action)
    if action not in range(4):
        raise ValueError("TERLA-A4 action must be in [0,3]")
    if action == 0:
        return TargetDecision(action, 0, None, DURATIONS[0], "requested_sleep")
    candidates = [(float(score), index, host) for index, (host, score) in enumerate(
        zip(graph.host_names, graph.host_scores.tolist())) if float(score) > 0]
    if not candidates:
        return TargetDecision(action, 0, None, DURATIONS[0], "no_observable_suspicious_target")
    if action == 1:
        _score, _index, target = min(candidates, key=lambda x: (x[0], x[1]))
    else:
        _score, _index, target = max(candidates, key=lambda x: (x[0], -x[1]))
    return TargetDecision(action, action, target, DURATIONS[action], "observable_score_target")
