from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Mapping


FORBIDDEN_FIELDS = {"compromised", "true_compromise", "red_session", "ground_truth"}


@dataclass(frozen=True, order=True)
class AlertToken:
    tick: int
    host: str
    evidence_type: str
    confidence: float
    removable: bool = True
    persisted_after_remove: bool = False


@dataclass(frozen=True)
class DCAConfig:
    method_name: str = "DCA-CC4 (adapted)"
    min_pts: int = 3
    time_window: int = 10
    high_confidence: float = 0.75
    multi_channel_threshold: int = 2
    epsilon: float = 0.25

    def __post_init__(self) -> None:
        if self.method_name != "DCA-CC4 (adapted)":
            raise ValueError("DCA must be disclosed as adapted")
        if self.min_pts != 3 or self.time_window <= 0 or self.multi_channel_threshold < 2:
            raise ValueError("invalid frozen DCA alert settings")
        if not (0 <= self.epsilon <= 1 and 0 <= self.high_confidence <= 1):
            raise ValueError("DCA thresholds must lie in [0,1]")


def tokenize_alerts(observable_events: Iterable[Mapping[str, object]]) -> tuple[AlertToken, ...]:
    """Construct tokens only from Blue-visible event fields."""
    tokens = []
    for event in observable_events:
        overlap = FORBIDDEN_FIELDS.intersection(map(str, event.keys()))
        if overlap:
            raise ValueError(f"hidden truth fields are forbidden: {sorted(overlap)}")
        host = str(event.get("host", "")).strip()
        kind = str(event.get("evidence_type", event.get("type", ""))).strip()
        if not host or not kind:
            continue
        confidence = float(event.get("confidence", 0.0))
        if not isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("observable confidence must lie in [0,1]")
        tokens.append(AlertToken(int(event.get("tick", 0)), host, kind, confidence,
                                 bool(event.get("removable", True)),
                                 bool(event.get("persisted_after_remove", False))))
    return tuple(sorted(tokens))


@dataclass(frozen=True)
class DCAActionDecision:
    method: str
    requested_action_id: int
    target: str | None
    reason: str
    evidence_count: int
    inferred_attacker_action: object | None = None


class DCAResponseAdapter:
    """Fixed auditable mapping from inferred observable attack evidence to A4."""

    def __init__(self, config: DCAConfig | None = None, *, path_model=None):
        self.config = config or DCAConfig()
        if path_model is None:
            from .path_model import TabularAttackPathModel
            path_model = TabularAttackPathModel()
        self.path_model = path_model

    def act(self, observable_events: Iterable[Mapping[str, object]]) -> DCAActionDecision:
        tokens = tokenize_alerts(observable_events)
        if not tokens:
            return DCAActionDecision(self.config.method_name, 0, None, "no_observable_evidence", 0)
        from .path_model import cluster_alerts, cluster_state
        clusters = cluster_alerts(tokens, self.config)
        if not clusters:
            # DBSCAN noise is still observable evidence. Deterministically choose
            # an observed host for investigation; never reinterpret noise as absence.
            by_host: dict[str, list[AlertToken]] = {}
            for token in tokens:
                by_host.setdefault(token.host, []).append(token)
            target, evidence = max(
                by_host.items(),
                key=lambda item: (
                    max(x.confidence for x in item[1]),
                    len(item[1]),
                    item[0],
                ),
            )
            return DCAActionDecision(
                self.config.method_name,
                1,
                target,
                "unclustered_or_single_observable_evidence",
                len(evidence),
            )
        # Cluster rank consumes the learned attack-path value before observable ties.
        ranked = []
        for evidence in clusters:
            state = cluster_state(evidence)
            inference = self.path_model.infer(state)
            ranked.append((inference.value, max(x.confidence for x in evidence),
                           len({x.evidence_type for x in evidence}), len(evidence),
                           evidence[0].host, evidence, inference))
        *_rank, target, evidence, inference = max(ranked, key=lambda item: item[:5])
        confidence = max(x.confidence for x in evidence)
        channels = len({x.evidence_type for x in evidence})
        persistent = any(x.persisted_after_remove for x in evidence)
        removable = any(x.removable for x in evidence)
        if confidence < self.config.high_confidence or channels < self.config.multi_channel_threshold:
            action, reason = 1, "single_or_low_confidence_evidence"
        elif persistent or not removable:
            action, reason = 3, "persistent_or_nonremovable_multichannel_evidence"
        else:
            action, reason = 2, "high_confidence_removable_multichannel_evidence"
        return DCAActionDecision(self.config.method_name, action, target, reason, len(evidence),
                                 inference.attacker_action)
