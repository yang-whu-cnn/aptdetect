from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class HiddenRewardRecord:
    """Environment-side label channel. Never pass this object to policy code."""
    red_sessions_by_host: tuple[int, ...]
    service_unreliability_by_host: tuple[tuple[float, ...], ...]
    ot_host: tuple[bool, ...]


class TERLARewardChannel:
    @staticmethod
    def health(record: HiddenRewardRecord) -> float:
        n = len(record.red_sessions_by_host)
        if len(record.service_unreliability_by_host) != n or len(record.ot_host) != n:
            raise ValueError("TERLA hidden reward host arrays must have equal length")
        total = 0.0
        for sessions, unreliability, is_ot in zip(
                record.red_sessions_by_host, record.service_unreliability_by_host, record.ot_host):
            if int(sessions) < 0:
                raise ValueError("red session count cannot be negative")
            values = tuple(float(x) for x in unreliability)
            if any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
                raise ValueError("service unreliability must lie in [0,1]")
            total += int(sessions) + sum(values) * (2.0 if is_ot else 1.0)
        return -total

    @classmethod
    def transition_reward(cls, previous: HiddenRewardRecord,
                          current: HiddenRewardRecord) -> float:
        """Paper wrapper rewards the change in segment health per timestep."""
        return cls.health(current) - cls.health(previous)


def _is_agent(agent_name: object, role: str) -> bool:
    return f"{role}_agent" in str(agent_name).lower()


def _is_ot_service(service_name: object) -> bool:
    name = getattr(service_name, "name", str(service_name)).upper()
    return name == "OTSERVICE" or name.endswith(".OTSERVICE")


class CC4HiddenRewardAdapter:
    """Exact environment-side TERLA reward extractor; forbidden to policy code."""

    RED_SESSION_PATH = "controller.state.hosts[*].sessions -> state.sessions[*][*].active"
    GREEN_SESSION_PATH = "controller.state.hosts[*].sessions -> state.sessions[*][*].active"
    SERVICE_PATH = "controller.state.hosts[*].services[*].active/get_service_reliability()"
    OT_PATH = "controller.state.hosts[*].services key == ProcessName.OTSERVICE"

    @classmethod
    def record(cls, controller, hostnames: Iterable[str]) -> HiddenRewardRecord:
        red_counts: list[int] = []
        green_unreliability: list[tuple[float, ...]] = []
        ot_flags: list[bool] = []
        state = controller.state
        for hostname in tuple(hostnames):
            host = state.hosts[str(hostname)]
            red = 0
            green = 0
            for agent_name, session_ids in host.sessions.items():
                sessions = state.sessions.get(agent_name, {})
                active = sum(1 for session_id in session_ids
                             if sessions.get(session_id) is not None
                             and bool(sessions[session_id].active))
                if _is_agent(agent_name, "red"):
                    red += active
                elif _is_agent(agent_name, "green"):
                    green += active
            active_services = [service for service in host.services.values()
                               if bool(service.active)]
            if active_services:
                failure_probability = sum(
                    1.0 - float(service.get_service_reliability()) / 100.0
                    for service in active_services) / len(active_services)
            else:
                failure_probability = 1.0
            red_counts.append(red)
            green_unreliability.append(tuple(failure_probability for _ in range(green)))
            ot_flags.append(any(_is_ot_service(name) for name in host.services))
        return HiddenRewardRecord(tuple(red_counts), tuple(green_unreliability), tuple(ot_flags))


@dataclass(frozen=True)
class TERLAProductionGate:
    exact_red_session_channel: bool = False
    exact_green_service_reliability_channel: bool = False
    exact_ot_host_mapping: bool = False
    policy_reward_leak_test_passed: bool = False

    def require_ready(self) -> None:
        missing = [name for name, ready in self.__dict__.items() if not ready]
        if missing:
            raise RuntimeError("TERLA training gate blocked: " + ", ".join(missing))

    @classmethod
    def verified_reward_channel(cls) -> "TERLAProductionGate":
        """Reward fields verified; policy leakage remains an independent gate."""
        return cls(True, True, True, False)

    @classmethod
    def development_runner_verified(cls) -> "TERLAProductionGate":
        return cls(True, True, True, True)
