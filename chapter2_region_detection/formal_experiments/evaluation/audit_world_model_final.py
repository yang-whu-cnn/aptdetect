from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from formal_experiments.data_collection.decision_replay import DecisionEpochTransition
from formal_experiments.evaluation.validate_bootstrap_world_model import (
    build_rollout_windows,
    complete,
    load_replay_jsonl,
    mae,
    quantile_table,
    rmse,
    sample_rmse,
    spearman,
    validate_frozen_split,
)
from formal_experiments.training.bootstrap_world_model import (
    BootstrapProbabilisticWorldModel,
    build_world_model_dataset,
)
from shared.action_contract import ACTION_CONTRACTS


DEFAULT_TRAIN_REPLAY = "outputs/formal_replay_v2/train.jsonl"
DEFAULT_VALIDATION_REPLAY = "outputs/formal_replay_v2/validation.jsonl"
DEFAULT_WORLD_MODEL = "outputs/world_model_v2/a4_5b/world_model_absolute.pt"
DEFAULT_OUT = "outputs/world_model_v2/b0_1/per_action_audit.json"

MIN_VALIDATION_COUNT_PER_ACTION = 150
MAX_ACTION_RMSE_PERSISTENCE_RATIO = 1.25
MIN_ACTIONS_BEATING_PERSISTENCE = 3
MAX_H4_FIRST_ACTION_RATIO_BEFORE_REVIEW = 2.0
FROZEN_METRIC_ABS_TOLERANCE = 5e-4

FROZEN_REFERENCE = {
    "one_step_rmse": 0.132231702,
    "one_step_persistence_rmse": 0.180830250,
    "h4_rmse": 0.191052066,
    "h4_persistence_rmse": 0.219133339,
    "h4_uncertainty_error_spearman": 0.687620634,
}

ACTION_ID_TO_NAME = {int(item.action_id): str(item.name) for item in ACTION_CONTRACTS}
FAMILY_TO_NAME = {str(item.cyborg_action): str(item.name) for item in ACTION_CONTRACTS}
ACTION_NAMES = tuple(ACTION_ID_TO_NAME[index] for index in sorted(ACTION_ID_TO_NAME))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _finite(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("audit metric must be finite")
    return value


def reference_from_a4_5b_report(path: str | Path) -> dict[str, float]:
    source = resolve_path(path)
    report = json.loads(source.read_text(encoding="utf-8"))
    selected = report.get("selection", {}).get("selected_target_mode")
    if selected != "absolute":
        raise ValueError("B0.1 reference report must select absolute target mode")
    metrics = report.get("modes", {}).get("absolute", {})
    one = metrics.get("one_step", {})
    h4 = metrics.get("rollout_h4", {})
    reference = {
        "one_step_rmse": one.get("rmse"),
        "one_step_persistence_rmse": one.get("persistence_rmse"),
        "h4_rmse": h4.get("rmse"),
        "h4_persistence_rmse": h4.get("persistence_rmse"),
        "h4_uncertainty_error_spearman": h4.get("epistemic_error_spearman"),
    }
    if set(reference) != set(FROZEN_REFERENCE):
        raise ValueError("B0.1 reference metric keys changed")
    return {name: _finite(value) for name, value in reference.items()}


def _uncertainty_from_prediction(result: dict[str, np.ndarray]) -> np.ndarray:
    epistemic = np.asarray(result["epistemic_variance"], dtype=np.float64)
    if epistemic.ndim != 2:
        raise ValueError("epistemic variance must have shape [N,D]")
    return np.mean(np.sqrt(np.maximum(epistemic, 0.0)), axis=1)


def _one_step_arrays(
    model: BootstrapProbabilisticWorldModel,
    transitions: Sequence[DecisionEpochTransition],
) -> dict[str, object]:
    items = complete(transitions)
    dataset = build_world_model_dataset(items)
    prediction = model.predict_ensemble(dataset.states, dataset.actions)
    predicted = np.asarray(prediction["mean"], dtype=np.float64)
    target = np.asarray(dataset.next_states, dtype=np.float64)
    states = np.asarray(dataset.states, dtype=np.float64)
    errors = sample_rmse(predicted, target)
    uncertainty = _uncertainty_from_prediction(prediction)
    families = np.asarray([item.executed_action_family for item in items], dtype=object)
    agents = np.asarray([item.agent_name for item in items], dtype=object)
    return {
        "items": items,
        "predicted": predicted,
        "target": target,
        "states": states,
        "errors": errors,
        "uncertainty": uncertainty,
        "families": families,
        "agents": agents,
    }


def _bucket_metrics(
    *,
    predicted: np.ndarray,
    target: np.ndarray,
    persistence: np.ndarray,
    uncertainty: np.ndarray,
) -> dict[str, object]:
    count = int(predicted.shape[0])
    if count <= 0:
        raise ValueError("audit bucket must not be empty")
    model_rmse = rmse(predicted, target)
    persistence_rmse = rmse(persistence, target)
    ratio = float("inf") if persistence_rmse <= 0 else model_rmse / persistence_rmse
    errors = sample_rmse(predicted, target)
    return {
        "count": count,
        "rmse": _finite(model_rmse),
        "mae": _finite(mae(predicted, target)),
        "persistence_rmse": _finite(persistence_rmse),
        "persistence_mae": _finite(mae(persistence, target)),
        "rmse_over_persistence": _finite(ratio),
        "beats_persistence": bool(model_rmse <= persistence_rmse),
        "uncertainty_error_spearman": _finite(spearman(uncertainty, errors)),
        "uncertainty_quartiles": quantile_table(uncertainty, errors, bins=4),
    }


def action_one_step_audit(
    model: BootstrapProbabilisticWorldModel,
    transitions: Sequence[DecisionEpochTransition],
) -> dict[str, dict[str, object]]:
    arrays = _one_step_arrays(model, transitions)
    result: dict[str, dict[str, object]] = {}
    for family, name in FAMILY_TO_NAME.items():
        mask = arrays["families"] == family
        if not bool(np.any(mask)):
            result[name] = {"count": 0, "missing": True}
            continue
        result[name] = _bucket_metrics(
            predicted=arrays["predicted"][mask],
            target=arrays["target"][mask],
            persistence=arrays["states"][mask],
            uncertainty=arrays["uncertainty"][mask],
        )
    return result


def per_agent_action_audit(
    model: BootstrapProbabilisticWorldModel,
    transitions: Sequence[DecisionEpochTransition],
) -> dict[str, dict[str, object]]:
    arrays = _one_step_arrays(model, transitions)
    output: dict[str, dict[str, object]] = {}
    for agent in sorted(set(arrays["agents"].tolist())):
        agent_result: dict[str, object] = {}
        for family, name in FAMILY_TO_NAME.items():
            mask = (arrays["agents"] == agent) & (arrays["families"] == family)
            if not bool(np.any(mask)):
                agent_result[name] = {"count": 0}
                continue
            agent_result[name] = {
                "count": int(np.sum(mask)),
                "rmse": _finite(rmse(arrays["predicted"][mask], arrays["target"][mask])),
                "persistence_rmse": _finite(rmse(arrays["states"][mask], arrays["target"][mask])),
            }
        output[str(agent)] = agent_result
    return output


def _rollout_arrays(
    model: BootstrapProbabilisticWorldModel,
    transitions: Sequence[DecisionEpochTransition],
    *,
    horizon: int = 4,
) -> dict[str, np.ndarray]:
    windows = build_rollout_windows(transitions, horizon)
    initial = np.stack([item.initial_state for item in windows]).astype(np.float32)
    actions = np.stack([item.actions for item in windows]).astype(np.int64)
    target = np.stack([item.target_state for item in windows]).astype(np.float32)

    member_final = []
    for member_index in range(len(model.models)):
        current = torch.as_tensor(initial, dtype=torch.float32, device=model.device)
        for step in range(horizon):
            action_tensor = torch.as_tensor(
                actions[:, step], dtype=torch.long, device=model.device
            )
            current = model.predict_member_mean_tensor(
                member_index, current, action_tensor
            )
        member_final.append(current.detach().cpu().numpy())

    members = np.stack(member_final, axis=0)
    predicted = np.mean(members, axis=0)
    uncertainty = np.mean(np.std(members, axis=0, ddof=0), axis=1)
    return {
        "initial": initial,
        "actions": actions,
        "target": target,
        "predicted": predicted,
        "uncertainty": uncertainty,
    }


def first_action_h4_audit(
    model: BootstrapProbabilisticWorldModel,
    transitions: Sequence[DecisionEpochTransition],
) -> dict[str, dict[str, object]]:
    arrays = _rollout_arrays(model, transitions, horizon=4)
    first_actions = arrays["actions"][:, 0]
    result: dict[str, dict[str, object]] = {}
    for action_id, name in ACTION_ID_TO_NAME.items():
        mask = first_actions == int(action_id)
        if not bool(np.any(mask)):
            result[name] = {"count": 0, "missing": True}
            continue
        result[name] = _bucket_metrics(
            predicted=arrays["predicted"][mask],
            target=arrays["target"][mask],
            persistence=arrays["initial"][mask],
            uncertainty=arrays["uncertainty"][mask],
        )
    return result


def aggregate_reproduction(
    model: BootstrapProbabilisticWorldModel,
    validation_transitions: Sequence[DecisionEpochTransition],
) -> dict[str, float]:
    one = _one_step_arrays(model, validation_transitions)
    roll = _rollout_arrays(model, validation_transitions, horizon=4)
    one_rmse = rmse(one["predicted"], one["target"])
    one_persistence = rmse(one["states"], one["target"])
    h4_rmse = rmse(roll["predicted"], roll["target"])
    h4_persistence = rmse(roll["initial"], roll["target"])
    h4_errors = sample_rmse(roll["predicted"], roll["target"])
    h4_spearman = spearman(roll["uncertainty"], h4_errors)
    return {
        "one_step_rmse": _finite(one_rmse),
        "one_step_persistence_rmse": _finite(one_persistence),
        "h4_rmse": _finite(h4_rmse),
        "h4_persistence_rmse": _finite(h4_persistence),
        "h4_uncertainty_error_spearman": _finite(h4_spearman),
    }


def completed_action_counts(
    transitions: Sequence[DecisionEpochTransition],
) -> dict[str, int]:
    counts = Counter(item.executed_action_family for item in complete(transitions))
    return {name: int(counts.get(family, 0)) for family, name in FAMILY_TO_NAME.items()}


def h4_catastrophic_review(
    first_action_h4: dict[str, dict[str, object]],
    *,
    max_ratio: float = MAX_H4_FIRST_ACTION_RATIO_BEFORE_REVIEW,
) -> dict[str, object]:
    actions: dict[str, dict[str, object]] = {}
    review_required = False
    for name in ACTION_NAMES:
        metrics = first_action_h4.get(name, {})
        count = int(metrics.get("count", 0))
        ratio = metrics.get("rmse_over_persistence")
        missing = count <= 0 or bool(metrics.get("missing", False))
        finite = bool(ratio is not None and np.isfinite(float(ratio)))
        catastrophic = bool(missing or not finite or float(ratio) > float(max_ratio))
        review_required = review_required or catastrophic
        actions[name] = {
            "count": count,
            "rmse_over_persistence": None if ratio is None else float(ratio),
            "missing": missing,
            "catastrophic": catastrophic,
        }
    return {
        "max_h4_first_action_ratio_before_review": float(max_ratio),
        "actions": actions,
        "manual_review_required": bool(review_required),
    }


def b0_1_gate(
    *,
    validation_per_action: dict[str, dict[str, object]],
    first_action_h4: dict[str, dict[str, object]],
    aggregate: dict[str, float],
    reference: Mapping[str, float] | None = None,
    min_validation_count: int = MIN_VALIDATION_COUNT_PER_ACTION,
    max_ratio: float = MAX_ACTION_RMSE_PERSISTENCE_RATIO,
    min_beating: int = MIN_ACTIONS_BEATING_PERSISTENCE,
    frozen_tolerance: float = FROZEN_METRIC_ABS_TOLERANCE,
) -> dict[str, object]:
    expected_reference = dict(FROZEN_REFERENCE if reference is None else reference)
    if set(expected_reference) != set(FROZEN_REFERENCE):
        raise ValueError("B0.1 reference metric keys changed")
    action_checks: dict[str, dict[str, object]] = {}
    beating = 0
    all_count = True
    all_ratio = True

    for name in ACTION_NAMES:
        metrics = validation_per_action.get(name, {})
        count = int(metrics.get("count", 0))
        ratio = float(metrics.get("rmse_over_persistence", float("inf")))
        beats = bool(metrics.get("beats_persistence", False))
        count_ok = count >= int(min_validation_count)
        ratio_ok = bool(np.isfinite(ratio) and ratio <= float(max_ratio))
        all_count = all_count and count_ok
        all_ratio = all_ratio and ratio_ok
        beating += int(beats)
        action_checks[name] = {
            "count": count,
            "count_ok": count_ok,
            "rmse_over_persistence": ratio,
            "ratio_ok": ratio_ok,
            "beats_persistence": beats,
        }

    reproduction = {}
    for key, expected in expected_reference.items():
        actual = float(aggregate[key])
        delta = abs(actual - float(expected))
        reproduction[key] = {
            "actual": actual,
            "expected": float(expected),
            "abs_delta": delta,
            "within_tolerance": bool(delta <= float(frozen_tolerance)),
        }

    reproduction_ok = all(item["within_tolerance"] for item in reproduction.values())
    h4_review = h4_catastrophic_review(first_action_h4)
    result = {
        "thresholds": {
            "min_validation_count_per_action": int(min_validation_count),
            "max_action_rmse_persistence_ratio": float(max_ratio),
            "min_actions_beating_persistence": int(min_beating),
            "max_h4_first_action_ratio_before_review": float(
                MAX_H4_FIRST_ACTION_RATIO_BEFORE_REVIEW
            ),
            "frozen_metric_abs_tolerance": float(frozen_tolerance),
        },
        "actions": action_checks,
        "all_actions_have_min_count": bool(all_count),
        "all_actions_ratio_within_limit": bool(all_ratio),
        "actions_beating_persistence": int(beating),
        "enough_actions_beat_persistence": bool(beating >= int(min_beating)),
        "h4_catastrophic_review": h4_review,
        "frozen_aggregate_reproduction": reproduction,
        "frozen_aggregate_reproduction_ok": bool(reproduction_ok),
    }
    result["pass"] = bool(
        all_count
        and all_ratio
        and beating >= int(min_beating)
        and reproduction_ok
        and not h4_review["manual_review_required"]
    )
    return result


def run_audit(
    *,
    train_replay: str | Path,
    validation_replay: str | Path,
    checkpoint: str | Path,
    reference_report: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, object]:
    train = load_replay_jsonl(resolve_path(train_replay))
    validation = load_replay_jsonl(resolve_path(validation_replay))
    validate_frozen_split(train, "train")
    validate_frozen_split(validation, "validation")

    model = BootstrapProbabilisticWorldModel.load_checkpoint(
        resolve_path(checkpoint), device=device
    )
    if model.config.target_mode != "absolute":
        raise RuntimeError("B0.1 requires the frozen absolute WM checkpoint")
    if model.config.state_dim != 27 or model.config.n_actions != 4:
        raise RuntimeError("B0.1 checkpoint violates D27/A4 contract")
    if model.config.ensemble_size != 5 or len(model.models) != 5:
        raise RuntimeError("B0.1 checkpoint violates frozen M=5 ensemble contract")

    train_counts = completed_action_counts(train)
    validation_counts = completed_action_counts(validation)
    per_action = action_one_step_audit(model, validation)
    h4_first_action = first_action_h4_audit(model, validation)
    aggregate = aggregate_reproduction(model, validation)
    reference = (
        dict(FROZEN_REFERENCE)
        if reference_report is None
        else reference_from_a4_5b_report(reference_report)
    )
    gate = b0_1_gate(
        validation_per_action=per_action,
        first_action_h4=h4_first_action,
        aggregate=aggregate,
        reference=reference,
    )

    return {
        "contract": {
            "phase": "gate_b0_1_world_model_final_audit",
            "training_performed": False,
            "train_split_only_for_counts": True,
            "validation_used_for_quality": True,
            "calibration_test_used": False,
            "state_dim": int(model.config.state_dim),
            "n_actions": int(model.config.n_actions),
            "target_mode": str(model.config.target_mode),
            "ensemble_size": int(model.config.ensemble_size),
            "horizon": 4,
        },
        "artifacts": {
            "train_replay": str(resolve_path(train_replay)),
            "validation_replay": str(resolve_path(validation_replay)),
            "world_model": str(resolve_path(checkpoint)),
            "reference_report": (
                None if reference_report is None else str(resolve_path(reference_report))
            ),
        },
        "train_completed_action_counts": train_counts,
        "validation_completed_action_counts": validation_counts,
        "validation_per_action_one_step": per_action,
        "validation_first_action_h4": h4_first_action,
        "validation_per_agent_action_one_step": per_agent_action_audit(model, validation),
        "aggregate_reproduction": aggregate,
        "quality_gate": gate,
        "pass": bool(gate["pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate B0.1 frozen WM per-action final audit"
    )
    parser.add_argument("--train-replay", default=DEFAULT_TRAIN_REPLAY)
    parser.add_argument("--validation-replay", default=DEFAULT_VALIDATION_REPLAY)
    parser.add_argument("--checkpoint", default=DEFAULT_WORLD_MODEL)
    parser.add_argument(
        "--reference-report",
        default=None,
        help="A4.5b report from the same trained checkpoint; omit only for historical frozen-reference reproduction.",
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    report = run_audit(
        train_replay=args.train_replay,
        validation_replay=args.validation_replay,
        checkpoint=args.checkpoint,
        reference_report=args.reference_report,
        device=args.device,
    )

    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[GATE B0.1 WM FINAL AUDIT]")
    print("train_counts:", report["train_completed_action_counts"])
    print("validation_counts:", report["validation_completed_action_counts"])
    for name in ACTION_NAMES:
        metrics = report["validation_per_action_one_step"][name]
        h4 = report["validation_first_action_h4"][name]
        print(
            f"{name}: n={metrics['count']} "
            f"rmse={metrics.get('rmse')} "
            f"persistence={metrics.get('persistence_rmse')} "
            f"ratio={metrics.get('rmse_over_persistence')} "
            f"spearman={metrics.get('uncertainty_error_spearman')} "
            f"h4_n={h4.get('count')} h4_ratio={h4.get('rmse_over_persistence')}"
        )
    print("aggregate_reproduction:", report["aggregate_reproduction"])
    print("actions_beating_persistence:", report["quality_gate"]["actions_beating_persistence"])
    print(
        "h4_manual_review_required:",
        report["quality_gate"]["h4_catastrophic_review"]["manual_review_required"],
    )
    print("pass:", report["pass"])
    print("[OK] report:", out)

    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
