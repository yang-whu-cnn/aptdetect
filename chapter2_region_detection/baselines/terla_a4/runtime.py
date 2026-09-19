from __future__ import annotations

from typing import Any, Mapping

from .graph import HeteroGraph, TERLAGraphBuilder


def public_mission_phase(tick: int, episode_ticks: int) -> int:
    if episode_ticks <= 0 or tick < 0:
        raise ValueError("invalid public mission clock")
    return min(2, (3 * int(tick)) // int(episode_ticks))


class ObservableTERLAState:
    """Post-reset Blue evidence only; reset process inventory is not an alert."""

    def __init__(self, tracker):
        self.tracker = tracker
        self.flags = {host: [0, 0] for host in tracker.inventory}
        self.builder = TERLAGraphBuilder()

    def update(self, observation: Mapping[str, Any], *, restored_target: str | None = None) -> None:
        if restored_target in self.flags:
            self.flags[str(restored_target)] = [0, 0]
        nested = observation.get("observation")
        hosts = nested if isinstance(nested, Mapping) else observation
        for raw_host, data in hosts.items():
            if not isinstance(data, Mapping):
                continue
            host = self.tracker.canonical_hostname(raw_host, data)
            if host not in self.flags:
                continue
            processes = data.get("Processes", ())
            process = int(isinstance(processes, (list, tuple)) and bool(processes))
            connection = 0
            if isinstance(processes, (list, tuple)):
                connection = int(any(isinstance(row, Mapping) and bool(row.get("Connections", ()))
                                     for row in processes))
            connection |= int(bool(data.get("Connections", ())))
            self.flags[host][0] |= process
            self.flags[host][1] |= connection

    def graph(self, *, tick: int, episode_ticks: int) -> HeteroGraph:
        normalized = {host: {"malicious_process_event": values[0],
                             "malicious_network_event": values[1]}
                      for host, values in self.flags.items()}
        return self.builder.build(normalized,
                                  mission_phase=public_mission_phase(tick, episode_ticks))
