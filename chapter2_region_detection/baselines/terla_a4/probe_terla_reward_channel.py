from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CYBORG_ROOT = os.environ.get("CYBORG_ROOT", r"D:\python1\cage-challenge-4")
if CYBORG_ROOT and CYBORG_ROOT not in sys.path:
    sys.path.insert(0, CYBORG_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CybORG import CybORG  # type: ignore
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent  # type: ignore
from CybORG.Simulator.Actions import Sleep  # type: ignore
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator  # type: ignore

from baselines.terla_a4.reward import CC4HiddenRewardAdapter, TERLARewardChannel
from shared.formal_state import BLUE_AGENTS, ObservableHostEvidenceTracker


def _hash(payload) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def run_probe(*, seed: int, ticks: int) -> dict:
    generator = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent, green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent, steps=ticks + 1)
    env = CybORG(scenario_generator=generator, seed=seed)
    env.reset(seed=seed)
    controller = env.environment_controller
    inventories = {}
    for agent in BLUE_AGENTS:
        tracker = ObservableHostEvidenceTracker(agent)
        tracker.reset(env.get_observation(agent))
        inventories[agent] = tuple(tracker.inventory)

    types = {name: set() for name in (
        "host_sessions", "session_active", "host_services", "service_active",
        "service_reliability", "service_key")}
    role = {"operational_hosts": 0, "ot_service_hosts": 0,
            "operational_without_ot": 0, "nonoperational_with_ot": 0}
    all_hosts = sorted({host for values in inventories.values() for host in values})
    for hostname in all_hosts:
        host = controller.state.hosts[hostname]
        types["host_sessions"].add(type(host.sessions).__name__)
        types["host_services"].add(type(host.services).__name__)
        has_ot = False
        for name, service in host.services.items():
            types["service_key"].add(type(name).__name__)
            types["service_active"].add(type(service.active).__name__)
            types["service_reliability"].add(type(service.get_service_reliability()).__name__)
            key = getattr(name, "name", str(name)).upper()
            has_ot |= key == "OTSERVICE" or key.endswith(".OTSERVICE")
        operational = "operational_zone" in hostname
        role["operational_hosts"] += int(operational)
        role["ot_service_hosts"] += int(has_ot)
        role["operational_without_ot"] += int(operational and not has_ot)
        role["nonoperational_with_ot"] += int(not operational and has_ot)
        for agent_name, ids in host.sessions.items():
            sessions = controller.state.sessions.get(agent_name, {})
            for session_id in ids:
                if sessions.get(session_id) is not None:
                    types["session_active"].add(type(sessions[session_id].active).__name__)

    samples = []
    for tick in range(1, ticks + 1):
        before = int(controller.step_count)
        env.parallel_step(actions={agent: Sleep() for agent in BLUE_AGENTS})
        after = int(controller.step_count)
        by_agent = {}
        for agent, hosts in inventories.items():
            record = CC4HiddenRewardAdapter.record(controller, hosts)
            by_agent[agent] = {
                "red_sessions": sum(record.red_sessions_by_host),
                "green_sessions": sum(len(x) for x in record.service_unreliability_by_host),
                "service_unreliability_sum": round(sum(map(sum, record.service_unreliability_by_host)), 8),
                "ot_hosts": sum(record.ot_host),
                "health": round(TERLARewardChannel.health(record), 8),
            }
        samples.append({"tick": tick, "step_delta": after - before, "agents": by_agent})

    totals = {key: sum(row["agents"][agent][key] for row in samples for agent in BLUE_AGENTS)
              for key in ("red_sessions", "green_sessions", "service_unreliability_sum", "ot_hosts")}
    exact = (all(row["step_delta"] == 1 for row in samples)
             and role["nonoperational_with_ot"] == 0
             and role["ot_service_hosts"] > 0
             and all(types.values()))
    return {
        "schema": "terla_reward_channel_probe_v1", "formal_result_eligible": False,
        "seed": seed, "requested_ticks": ticks, "executed_steps": int(controller.step_count),
        "field_paths": {"red_sessions": CC4HiddenRewardAdapter.RED_SESSION_PATH,
                        "green_sessions": CC4HiddenRewardAdapter.GREEN_SESSION_PATH,
                        "service_reliability": CC4HiddenRewardAdapter.SERVICE_PATH,
                        "ot_role": CC4HiddenRewardAdapter.OT_PATH},
        "field_types": {key: sorted(value) for key, value in types.items()},
        "role_consistency": role, "aggregate_over_ticks": totals,
        "tick_aggregate_sha256": _hash(samples), "exact_channel_verified": exact,
        "policy_channel_used": False,
        "notes": ["GreenLocalWork uniformly samples active services; mean failure probability is exact.",
                  "OT role is defined by actual OTSERVICE presence; operational zones also contain non-OT router roles.",
                  "No host/session/user/IP/service identifiers are emitted."],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3199)
    parser.add_argument("--ticks", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_probe(seed=args.seed, ticks=args.ticks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"exact_channel_verified": result["exact_channel_verified"],
                      "executed_steps": result["executed_steps"], "out": str(args.out)}))
    if not result["exact_channel_verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
