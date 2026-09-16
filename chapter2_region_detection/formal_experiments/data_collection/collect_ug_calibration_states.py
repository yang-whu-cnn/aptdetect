from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from formal_experiments.data_collection.collect_cc4_formal_replay import (
    collect_episode,
)
from shared.formal_state import BLUE_AGENTS, FORMAL_STATE_DIM


THIS = Path(__file__).resolve()
PROJ = THIS.parents[2]

CALIBRATION_SEEDS = tuple(range(3000, 3008))
EPISODE_STEPS = 100
PAD_SPACES = False

DEFAULT_OUT_DIR = (
    "outputs/ug_cem_v2/step6"
)

STATE_RECORD_FIELDS = (
    "split",
    "episode_seed",
    "agent_name",
    "decision_index",
    "global_tick_start",
    "state",
)


def state_only_record(transition) -> dict:
    seed = int(transition.episode_seed)
    if seed not in CALIBRATION_SEEDS:
        raise ValueError("Step-6 collector only accepts calibration seeds")

    agent = str(transition.agent_name)
    if agent not in BLUE_AGENTS:
        raise ValueError(f"unexpected Blue agent: {agent}")

    state = np.asarray(transition.state, dtype=np.float32)
    if state.shape != (FORMAL_STATE_DIM,):
        raise ValueError("invalid formal state shape")
    if not np.isfinite(state).all():
        raise ValueError("non-finite formal state")

    return {
        "split": "calibration",
        "episode_seed": seed,
        "agent_name": agent,
        "decision_index": int(transition.decision_index),
        "global_tick_start": int(transition.global_tick_start),
        "state": state.tolist(),
    }


def build_summary(records: list[dict]) -> dict:
    if not records:
        raise ValueError("calibration state collection is empty")

    per_agent = Counter(str(item["agent_name"]) for item in records)
    per_seed = Counter(int(item["episode_seed"]) for item in records)

    missing_agents = [
        agent for agent in BLUE_AGENTS
        if int(per_agent.get(agent, 0)) < 100
    ]
    if missing_agents:
        raise RuntimeError(
            "each agent needs at least 100 calibration states; "
            f"insufficient={missing_agents}"
        )

    missing_seeds = [
        seed for seed in CALIBRATION_SEEDS
        if int(per_seed.get(seed, 0)) <= 0
    ]
    if missing_seeds:
        raise RuntimeError(f"missing calibration seeds: {missing_seeds}")

    for item in records:
        if tuple(item.keys()) != STATE_RECORD_FIELDS:
            raise RuntimeError("state-only record schema changed")

    return {
        "split": "calibration",
        "seeds": list(CALIBRATION_SEEDS),
        "episode_steps": EPISODE_STEPS,
        "pad_spaces": PAD_SPACES,
        "state_dim": FORMAL_STATE_DIM,
        "record_fields": list(STATE_RECORD_FIELDS),
        "contains_hidden_truth": False,
        "contains_reward_labels": False,
        "record_count": len(records),
        "per_agent_counts": {
            agent: int(per_agent.get(agent, 0))
            for agent in BLUE_AGENTS
        },
        "per_seed_counts": {
            str(seed): int(per_seed.get(seed, 0))
            for seed in CALIBRATION_SEEDS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect planner-visible state-only data from frozen "
            "CC4 calibration seeds for Step-6 UG normalizer warm-up."
        )
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
    )
    args = parser.parse_args()

    records: list[dict] = []

    for position, seed in enumerate(CALIBRATION_SEEDS, start=1):
        print("=" * 80)
        print(
            "[STEP6 COLLECT]",
            f"seed={seed}",
            f"episode={position}/{len(CALIBRATION_SEEDS)}",
        )

        transitions, episode_summary = collect_episode(
            seed=seed,
            steps=EPISODE_STEPS,
            pad_spaces=PAD_SPACES,
        )

        projected = [state_only_record(item) for item in transitions]
        records.extend(projected)

        print("  transitions:", len(transitions))
        print("  per_agent:", episode_summary["per_agent_transitions"])

    summary = build_summary(records)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (PROJ / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    states_path = out_dir / "calibration_states.jsonl"
    summary_path = out_dir / "calibration_states_summary.json"

    with states_path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False))
            handle.write("\n")

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[STEP6 CALIBRATION STATE SUMMARY]")
    print("records:", summary["record_count"])
    print("per_agent:", summary["per_agent_counts"])
    print("per_seed:", summary["per_seed_counts"])
    print("hidden_truth:", summary["contains_hidden_truth"])
    print("reward_labels:", summary["contains_reward_labels"])
    print("[OK] states:", states_path)
    print("[OK] summary:", summary_path)


if __name__ == "__main__":
    main()
