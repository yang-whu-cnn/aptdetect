from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from .adapter import AlertToken, DCAConfig, DCAResponseAdapter
from .path_model import TabularAttackPathModel, cluster_alerts, cluster_state


EVIDENCE_FIELDS = (("Processes", "process", 0.55),
                   ("Connections", "connection", 0.75),
                   ("Files", "file", 0.90))


def _host_mapping(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = observation.get("observation")
    return nested if isinstance(nested, Mapping) else observation


class CC4ObservableAlertStream:
    """Translate only current Blue-visible CC4 observation rows into DCA tokens."""

    def __init__(self, tracker):
        self.tracker = tracker
        self.removed_hosts: set[str] = set()

    def note_completed_action(self, *, family: str | None, target: str | None,
                              success: bool | None) -> None:
        if family == "Remove" and target and success is True:
            self.removed_hosts.add(str(target))
        if family == "Restore" and target and success is True:
            self.removed_hosts.discard(str(target))

    def extract(self, observation: Mapping[str, Any], *, tick: int) -> tuple[AlertToken, ...]:
        out: list[AlertToken] = []
        for raw_host, data in _host_mapping(observation).items():
            if raw_host == "success" or not isinstance(data, Mapping):
                continue
            host = self.tracker.canonical_hostname(raw_host, data)
            if host is None:
                continue
            for field, kind, confidence in EVIDENCE_FIELDS:
                rows = data.get(field, ())
                if not isinstance(rows, (list, tuple)):
                    continue
                # Each visible row is an alert event. No controller or Red state is read.
                for _row in rows:
                    out.append(AlertToken(
                        tick=int(tick), host=str(host), evidence_type=kind,
                        confidence=float(confidence), removable=(kind != "file"),
                        persisted_after_remove=(str(host) in self.removed_hosts),
                    ))
        return tuple(sorted(out))


class DCAAgentRuntime:
    def __init__(self, *, tracker, config: DCAConfig | None = None, seed: int = 0):
        self.config = config or DCAConfig()
        self.tracker = tracker
        self.path_model = TabularAttackPathModel(gamma=0.95, dyna_steps=20, seed=seed)
        self.response = DCAResponseAdapter(self.config, path_model=self.path_model)
        self.stream = CC4ObservableAlertStream(tracker)
        self.tokens: list[AlertToken] = []
        self.previous_path_state = None

    def observe(self, observation: Mapping[str, Any], *, tick: int,
                completed_family: str | None = None, completed_target: str | None = None,
                completed_success: bool | None = None) -> tuple[AlertToken, ...]:
        self.stream.note_completed_action(family=completed_family, target=completed_target,
                                          success=completed_success)
        fresh = self.stream.extract(observation, tick=tick)
        self.tokens.extend(fresh)
        # DCA correlates alerts inside a finite temporal window.  Keeping every
        # historical token would let stale evidence influence later actions even
        # though the DBSCAN distance caps temporal separation.  The boundary is
        # inclusive: evidence exactly ``time_window`` ticks old remains eligible.
        cutoff = int(tick) - self.config.time_window
        self.tokens = [token for token in self.tokens if token.tick >= cutoff]
        clusters = cluster_alerts(self.tokens, self.config)
        if clusters:
            selected = max(clusters, key=lambda c: (c[-1].tick, len(c), c[0].host))
            state = cluster_state(selected)
            if self.previous_path_state is not None and state != self.previous_path_state and fresh:
                action = "observe:" + sorted({x.evidence_type for x in fresh})[-1]
                channels = len(state[1])
                self.path_model.update(
                    self.previous_path_state, action, state,
                    service_impact=min(1.0, len(fresh) / 10.0),
                    n_steps=max(1, len(self.tokens)), objective_reached=(channels >= 3),
                )
                self.path_model.value_iteration()
            self.previous_path_state = state
        return fresh

    def act(self):
        return self.response.act(asdict(item) for item in self.tokens)
