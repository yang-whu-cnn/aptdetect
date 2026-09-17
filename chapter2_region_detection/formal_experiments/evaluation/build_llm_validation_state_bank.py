from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from formal_experiments.data_collection.collect_cc4_formal_replay import (
    DEFAULT_SPLIT_SEEDS,
)
from formal_experiments.data_collection.decision_replay import DecisionEpochTransition
from formal_experiments.evaluation.validate_bootstrap_world_model import (
    load_replay_jsonl,
    validate_frozen_split,
)
from shared.formal_state import BLUE_AGENTS, FORMAL_STATE_DIM


DEFAULT_VALIDATION_REPLAY = "outputs/formal_replay_v2/validation.jsonl"
DEFAULT_OUT = "outputs/lwm_rl_v2/b2/validation_state_bank.jsonl"
DEFAULT_SUMMARY = "outputs/lwm_rl_v2/b2/validation_state_bank_summary.json"
TARGET_STATES_PER_CELL = 6
FEATURE17_INDEX = 17


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def state_sha256(state: np.ndarray) -> str:
    array = np.asarray(state, dtype="<f4")
    if array.shape != (FORMAL_STATE_DIM,) or not np.isfinite(array).all():
        raise ValueError("state must be finite D27")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _even_pick(items: Sequence[DecisionEpochTransition], count: int) -> list[DecisionEpochTransition]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return []
    if len(items) < count:
        raise ValueError(f"insufficient states: need {count}, have {len(items)}")
    if count == 1:
        return [items[(len(items) - 1) // 2]]
    positions = np.linspace(0, len(items) - 1, num=count)
    indices = np.rint(positions).astype(np.int64)
    # np.linspace+rint is unique when len(items) >= count, but lock the invariant.
    if len(set(int(x) for x in indices)) != count:
        raise RuntimeError("deterministic even sampling produced duplicate indices")
    return [items[int(index)] for index in indices]


def _feature17(transition: DecisionEpochTransition) -> int:
    state = np.asarray(transition.state, dtype=np.float32)
    if state.shape != (FORMAL_STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError("invalid transition state")
    value = float(state[FEATURE17_INDEX])
    if value not in (0.0, 1.0):
        raise ValueError(f"feature17 must be binary; got {value}")
    return int(value)


def select_cell(
    transitions: Sequence[DecisionEpochTransition],
    *,
    count: int = TARGET_STATES_PER_CELL,
) -> list[DecisionEpochTransition]:
    items = [item for item in transitions if bool(item.action_completed)]
    items.sort(key=lambda item: (int(item.decision_index), int(item.global_tick_start)))
    if len(items) < count:
        raise ValueError(f"cell has only {len(items)} completed states; need {count}")

    by_feature = {
        0: [item for item in items if _feature17(item) == 0],
        1: [item for item in items if _feature17(item) == 1],
    }

    if by_feature[0] and by_feature[1]:
        left = count // 2
        right = count - left
        if len(by_feature[0]) >= left and len(by_feature[1]) >= right:
            selected = _even_pick(by_feature[0], left) + _even_pick(by_feature[1], right)
        else:
            # Coverage is mandatory; if one class is scarce, take at least one and
            # fill the rest evenly from the remaining ordered cell without duplicates.
            scarce_value = 0 if len(by_feature[0]) < len(by_feature[1]) else 1
            scarce_take = min(max(1, len(by_feature[scarce_value])), count - 1)
            selected = _even_pick(by_feature[scarce_value], scarce_take)
            selected_ids = {(x.decision_index, x.global_tick_start) for x in selected}
            remaining = [
                item for item in items
                if (item.decision_index, item.global_tick_start) not in selected_ids
            ]
            selected += _even_pick(remaining, count - len(selected))
    else:
        selected = _even_pick(items, count)

    selected.sort(key=lambda item: (int(item.decision_index), int(item.global_tick_start)))
    if len(selected) != count:
        raise RuntimeError("state-bank cell count mismatch")
    if by_feature[0] and by_feature[1]:
        observed = {_feature17(item) for item in selected}
        if observed != {0, 1}:
            raise RuntimeError("state-bank selection failed feature17 coverage")
    return selected


def build_state_bank(
    transitions: Sequence[DecisionEpochTransition],
) -> tuple[list[dict], dict]:
    validate_frozen_split(transitions, "validation")
    groups: dict[tuple[int, str], list[DecisionEpochTransition]] = defaultdict(list)
    for item in transitions:
        key = (int(item.episode_seed), str(item.agent_name))
        groups[key].append(item)

    expected_keys = {
        (int(seed), str(agent))
        for seed in DEFAULT_SPLIT_SEEDS["validation"]
        for agent in BLUE_AGENTS
    }
    if set(groups) != expected_keys:
        missing = sorted(expected_keys - set(groups))
        extra = sorted(set(groups) - expected_keys)
        raise ValueError(f"validation state-bank group mismatch: missing={missing}, extra={extra}")

    records: list[dict] = []
    source_feature_coverage: dict[str, dict[str, list[int]]] = {}
    for seed in DEFAULT_SPLIT_SEEDS["validation"]:
        seed_key = str(int(seed))
        source_feature_coverage[seed_key] = {}
        for agent in BLUE_AGENTS:
            cell = groups[(int(seed), str(agent))]
            completed = [item for item in cell if bool(item.action_completed)]
            source_values = sorted({_feature17(item) for item in completed})
            source_feature_coverage[seed_key][str(agent)] = source_values
            selected = select_cell(cell, count=TARGET_STATES_PER_CELL)
            for item in selected:
                state = np.asarray(item.state, dtype=np.float32)
                records.append(
                    {
                        "split": "validation",
                        "episode_seed": int(item.episode_seed),
                        "agent_name": str(item.agent_name),
                        "decision_index": int(item.decision_index),
                        "global_tick_start": int(item.global_tick_start),
                        "feature17_any_valid_observable_target": _feature17(item),
                        "state_sha256": state_sha256(state),
                        "state": [float(x) for x in state.tolist()],
                    }
                )

    records.sort(
        key=lambda row: (
            int(row["episode_seed"]),
            str(row["agent_name"]),
            int(row["decision_index"]),
            int(row["global_tick_start"]),
        )
    )
    expected_total = len(DEFAULT_SPLIT_SEEDS["validation"]) * len(BLUE_AGENTS) * TARGET_STATES_PER_CELL
    if len(records) != expected_total:
        raise RuntimeError(f"expected {expected_total} bank states, got {len(records)}")

    canonical_lines = [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in records]
    bank_sha256 = hashlib.sha256(("\n".join(canonical_lines) + "\n").encode("utf-8")).hexdigest()

    per_seed = Counter(int(row["episode_seed"]) for row in records)
    per_agent = Counter(str(row["agent_name"]) for row in records)
    per_cell = Counter((int(row["episode_seed"]), str(row["agent_name"])) for row in records)
    feature_counts = Counter(int(row["feature17_any_valid_observable_target"]) for row in records)

    summary = {
        "status": "PASS",
        "split": "validation",
        "source_seeds": list(DEFAULT_SPLIT_SEEDS["validation"]),
        "agents": list(BLUE_AGENTS),
        "states_per_seed_agent_cell": TARGET_STATES_PER_CELL,
        "record_count": len(records),
        "expected_record_count": expected_total,
        "state_dim": FORMAL_STATE_DIM,
        "feature17_index": FEATURE17_INDEX,
        "per_seed": {str(k): int(v) for k, v in sorted(per_seed.items())},
        "per_agent": {str(k): int(v) for k, v in sorted(per_agent.items())},
        "per_cell": {f"{seed}:{agent}": int(value) for (seed, agent), value in sorted(per_cell.items())},
        "feature17_counts": {str(k): int(v) for k, v in sorted(feature_counts.items())},
        "source_feature17_coverage": source_feature_coverage,
        "hidden_truth_fields": False,
        "reward_fields": False,
        "future_state_fields": False,
        "bank_sha256": bank_sha256,
        "pass": True,
    }
    return records, summary


def write_state_bank(records: Iterable[dict], summary: dict, *, out: str | Path, summary_out: str | Path) -> None:
    out_path = resolve_path(out)
    summary_path = resolve_path(summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen B2 validation LLM state bank")
    parser.add_argument("--validation-replay", default=DEFAULT_VALIDATION_REPLAY)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    transitions = load_replay_jsonl(resolve_path(args.validation_replay))
    records, summary = build_state_bank(transitions)
    write_state_bank(records, summary, out=args.out, summary_out=args.summary_out)

    print("=" * 80)
    print("[GATE B2 VALIDATION STATE BANK]")
    print("record_count:", summary["record_count"])
    print("source_seeds:", summary["source_seeds"])
    print("agents:", summary["agents"])
    print("feature17_counts:", summary["feature17_counts"])
    print("bank_sha256:", summary["bank_sha256"])
    print("pass:", summary["pass"])
    print("[OK] bank:", resolve_path(args.out))
    print("[OK] summary:", resolve_path(args.summary_out))


if __name__ == "__main__":
    main()
