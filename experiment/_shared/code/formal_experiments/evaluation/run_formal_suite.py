"""Unified entrypoint for development smoke and formal-result validation.

This Phase-0 runner intentionally does not launch training or a long formal run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from formal_experiments.common.run_manifest import PROTOCOL_VERSION
from formal_experiments.evaluation.metrics_v3 import aggregate_repeat
from formal_experiments.evaluation.validate_formal_run import validate_run_directory


DEV_SMOKE_SEEDS = (39000, 39001)
DEV_SMOKE_TICKS = 20


def failure_owner_from_inventories(
    hostname: str, inventories: dict[str, tuple[str, ...]]
) -> str | None:
    """Resolve Blue jurisdiction from agent inventories, never inferred subnets."""
    owners = [agent for agent, hosts in inventories.items() if hostname in hosts]
    if len(owners) > 1:
        raise RuntimeError(
            f"LWF host {hostname} appears in multiple Blue inventories: {owners}"
        )
    return owners[0] if owners else None


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def collect_fixed_sleep_episode(*, seed: int, ticks: int) -> dict[str, Any]:
    """Collect one real local CC4 episode with a fixed Sleep policy.

    Hidden state is read only by this external evaluator for incident labels and
    LWF audit events; it is never supplied to the policy.
    """
    import os
    import sys

    cyborg_root = os.environ.get("CYBORG_ROOT", r"D:\python1\cage-challenge-4")
    if cyborg_root and cyborg_root not in sys.path:
        sys.path.insert(0, cyborg_root)
    from CybORG import CybORG  # type: ignore
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent  # type: ignore
    from CybORG.Shared.BlueRewardMachine import BlueRewardMachine  # type: ignore
    from CybORG.Simulator.Actions import Sleep  # type: ignore
    from CybORG.Simulator.Actions.GreenActions.GreenLocalWork import GreenLocalWork  # type: ignore
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator  # type: ignore
    from shared.formal_state import BLUE_AGENTS, ObservableHostEvidenceTracker

    # CC4 reset is controller tick zero; steps=ticks+1 permits exactly `ticks`
    # post-reset environment transitions for this development contract.
    generator = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent, green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent, steps=ticks + 1,
    )
    env = CybORG(scenario_generator=generator, seed=seed)
    env.reset(seed=seed)
    controller = env.environment_controller

    inventories: dict[str, tuple[str, ...]] = {}
    for agent in BLUE_AGENTS:
        tracker = ObservableHostEvidenceTracker(agent)
        tracker.reset(env.get_observation(agent))
        inventories[agent] = tuple(tracker.inventory)

    all_hosts = sorted({host for hosts in inventories.values() for host in hosts})
    def red_present(host: str) -> bool:
        state_host = controller.state.hosts[host]
        for agent_name, session_ids in state_host.sessions.items():
            if "red_agent" not in str(agent_name):
                continue
            sessions = controller.state.sessions.get(agent_name, {})
            if any(sessions.get(session_id) is not None and sessions[session_id].active for session_id in session_ids):
                return True
        return False

    def local_work_failures() -> list[tuple[str, float]]:
        found: list[tuple[str, float]] = []
        rewards_by_phase = BlueRewardMachine("Blue").get_phase_rewards(int(controller.state.mission_phase))
        for agent_name, executed in controller.action.items():
            if "green_agent" not in str(agent_name) or not executed:
                continue
            observation_set = controller.observation.get(agent_name)
            if observation_set is None or not observation_set.observations:
                continue
            if observation_set.observations[0].data.get("success") != False:
                continue
            for action in executed:
                if isinstance(action, GreenLocalWork):
                    host = str(controller.state.ip_addresses[action.ip_address])
                    subnet = controller.state.hostname_subnet_map[host]
                    subnet_name = str(getattr(subnet, "value", subnet))
                    found.append((host, float(rewards_by_phase[subnet_name]["LWF"])))
        return found

    previous = {host: red_present(host) for host in all_hosts}
    open_incidents: dict[str, dict[str, Any]] = {}
    incidents: list[dict[str, Any]] = []
    rewards: list[dict[str, float]] = []
    failures: list[dict[str, Any]] = []
    excluded_failures: list[dict[str, Any]] = []

    for _ in range(ticks):
        before = int(controller.step_count)
        raw_observations, raw_rewards, _dones, _info = env.parallel_step(
            actions={agent: Sleep() for agent in BLUE_AGENTS}
        )
        after = int(controller.step_count)
        if after != before + 1:
            raise RuntimeError("CC4 did not advance exactly one tick")
        tick_rewards = {
            agent: float(sum(raw_rewards[agent].values())) for agent in BLUE_AGENTS
        }
        rewards.append(tick_rewards)
        for hostname, raw_penalty in local_work_failures():
            owner = failure_owner_from_inventories(hostname, inventories)
            if owner is None:
                excluded_failures.append({
                    "tick": after,
                    "host": hostname,
                    "raw_penalty": raw_penalty,
                    "reason": "outside_blue_jurisdiction",
                })
                continue
            failures.append({
                "tick": after,
                "agent_name": owner,
                "host": hostname,
                "raw_penalty": raw_penalty,
            })
        current = {host: red_present(host) for host in all_hosts}
        for host in all_hosts:
            if current[host] and not previous[host]:
                item = {"incident_id": f"{seed}:{host}:{after}", "host": host, "t_compromise": after, "t_recovered": None}
                incidents.append(item)
                open_incidents[host] = item
            elif previous[host] and not current[host]:
                if host not in open_incidents:
                    raise RuntimeError(f"recovery without tracked incident for {host}")
                open_incidents.pop(host)["t_recovered"] = after
        previous = current

    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "episode_seed": int(seed),
        "tick_count": int(ticks),
        "episode_end_tick": int(ticks),
        "tick_team_rewards": rewards,
        "operation_failure_events": failures,
        "excluded_non_blue_failure_events": excluded_failures,
        "recovery_actions": [],
        "incidents": incidents,
        "development_only": True,
        "formal_result_eligible": False,
        "policy": "fixed_sleep",
    }


def run_dev_smoke(output: str | Path, *, ticks: int = DEV_SMOKE_TICKS) -> dict[str, Any]:
    if ticks != DEV_SMOKE_TICKS:
        raise ValueError("Phase-0 development smoke is frozen at 20 ticks")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    try:
        episodes = [collect_fixed_sleep_episode(seed=seed, ticks=ticks) for seed in DEV_SMOKE_SEEDS]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CC4 development smoke is unavailable; the local CC4 runtime dependency "
            f"{exc.name!r} is missing. No synthetic episode data was written."
        ) from exc
    metrics = aggregate_repeat(episodes, expected_ticks=ticks).to_dict()
    manifest = {
        "schema_version": 1, "protocol_version": PROTOCOL_VERSION, "method": "fixed-sleep-smoke",
        "repeat_index": 1, "training_seed": 0, "test_episode_seeds": list(DEV_SMOKE_SEEDS),
        "episode_ticks": ticks, "code_commit": "development-uncommitted", "config_sha256": "0" * 64,
        "paper_sha256": "0" * 64, "upstream_commit": None, "python_version": "development",
        "dependencies": {}, "hardware": {}, "model_sha256": None, "run_mode": "dev",
        "formal_result_eligible": False,
    }
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "metrics.json", metrics)
    with (root / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, sort_keys=True) + "\n")
    validation = validate_run_directory(root, formal=False)
    _write_json(root / "validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError(f"development smoke failed evaluator gate: {validation['errors']}")
    return {"formal_result_eligible": False, "episodes": 2, "ticks": ticks, "validation": validation}


def main() -> None:
    parser = argparse.ArgumentParser(description="CC4 v3 development smoke and formal validation")
    parser.add_argument("--mode", required=True, choices=("dev-smoke", "validate-formal"))
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if args.mode == "dev-smoke":
        result = run_dev_smoke(args.run_dir)
    else:
        result = validate_run_directory(args.run_dir, formal=True)
        if not result["passed"]:
            raise SystemExit(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
