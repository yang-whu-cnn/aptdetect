from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from formal_experiments.evaluation.audit_world_model_final import (
    DEFAULT_TRAIN_REPLAY,
    DEFAULT_VALIDATION_REPLAY,
    DEFAULT_WORLD_MODEL,
    _uncertainty_from_prediction,
)
from formal_experiments.evaluation.validate_bootstrap_world_model import (
    complete,
    load_replay_jsonl,
    mae,
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


PROJ = Path(__file__).resolve().parents[2]
DEFAULT_PROVISIONAL_ROOT = "outputs/lwm_rl_v2/b4/provisional"
DEFAULT_OUT_ROOT = "outputs/lwm_rl_v2/b0_2"
KNN_K = 5
VALIDATION_PERCENTILE = 99.0
MAX_OOD_FRACTION = 0.10
MAX_PROBE_VALIDATION_RMSE_RATIO = 1.25
FROZEN_VARIANTS = (
    "llm_h_gpt56_sol",
    "llm_m_gpt54_mini",
    "llm_l_gemini35_flash_lite",
)
ACTION_NAMES = {int(item.action_id): str(item.name) for item in ACTION_CONTRACTS}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJ / path).resolve()


def load_probe_jsonl(path: str | Path) -> list[dict]:
    source = resolve_path(path)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"probe file is empty: {source}")
    for row in rows:
        if int(row.get("episode_seed", -1)) not in range(1000, 1032):
            raise ValueError("B0.2 probe contains non-train seed")
        for name in ("state", "next_state"):
            value = np.asarray(row.get(name), dtype=np.float32)
            if value.shape != (27,) or not np.isfinite(value).all():
                raise ValueError(f"probe row contains invalid {name}")
        if not np.isfinite(float(row.get("response_reward"))):
            raise ValueError("probe response_reward must be finite")
        if int(row.get("canonical_action_id", -1)) not in range(4):
            raise ValueError("probe canonical action outside A4")
    return rows


def _state_matrix(transitions) -> np.ndarray:
    matrix = np.stack([np.asarray(item.state, dtype=np.float32) for item in transitions])
    if matrix.ndim != 2 or matrix.shape[1] != 27 or not np.isfinite(matrix).all():
        raise ValueError("replay state matrix violates D27")
    return matrix


def _knn_distances(query: np.ndarray, reference: np.ndarray, *, k: int = KNN_K, chunk: int = 128) -> np.ndarray:
    q = np.asarray(query, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if q.ndim != 2 or r.ndim != 2 or q.shape[1] != r.shape[1]:
        raise ValueError("kNN matrices must be 2D with same feature dimension")
    if int(k) < 1 or r.shape[0] < int(k):
        raise ValueError("invalid kNN k/reference size")
    r_norm = np.sum(r * r, axis=1)
    output = []
    for start in range(0, q.shape[0], int(chunk)):
        part = q[start : start + int(chunk)]
        d2 = (
            np.sum(part * part, axis=1, keepdims=True)
            + r_norm.reshape(1, -1)
            - 2.0 * part @ r.T
        )
        d2 = np.maximum(d2, 0.0)
        kth = np.partition(d2, int(k) - 1, axis=1)[:, int(k) - 1]
        output.append(np.sqrt(kth))
    return np.concatenate(output, axis=0).astype(np.float64)


def build_ood_reference(*, model, train_transitions, validation_transitions) -> dict:
    normalizer = model._require_fitted()
    train_state = _state_matrix(train_transitions)
    validation_state = _state_matrix(validation_transitions)
    train_z = normalizer.normalize_np(train_state)
    validation_z = normalizer.normalize_np(validation_state)
    validation_knn = _knn_distances(validation_z, train_z, k=KNN_K)
    threshold = float(np.percentile(validation_knn, VALIDATION_PERCENTILE))
    if not np.isfinite(threshold) or threshold < 0.0:
        raise RuntimeError("invalid frozen OOD threshold")
    return {
        "train_state_count": int(train_z.shape[0]),
        "validation_state_count": int(validation_z.shape[0]),
        "knn_k": int(KNN_K),
        "validation_percentile": float(VALIDATION_PERCENTILE),
        "threshold": threshold,
        "validation_knn_mean": float(np.mean(validation_knn)),
        "validation_knn_p95": float(np.percentile(validation_knn, 95)),
        "train_z": train_z,
    }


def _probe_one_step(model, rows: list[dict]) -> dict:
    completed = [row for row in rows if bool(row.get("action_completed"))]
    if not completed:
        raise ValueError("probe has zero completed actions for dynamics audit")
    states = np.stack([np.asarray(row["state"], dtype=np.float32) for row in completed])
    next_states = np.stack([np.asarray(row["next_state"], dtype=np.float32) for row in completed])
    actions = np.asarray([int(row["canonical_action_id"]) for row in completed], dtype=np.int64)
    prediction = model.predict_ensemble(states, actions)
    predicted = np.asarray(prediction["mean"], dtype=np.float64)
    target = next_states.astype(np.float64)
    errors = sample_rmse(predicted, target)
    uncertainty = _uncertainty_from_prediction(prediction)

    per_action = {}
    counts = Counter(int(x) for x in actions.tolist())
    for action_id in range(4):
        mask = actions == action_id
        if not bool(np.any(mask)):
            per_action[ACTION_NAMES[action_id]] = {"count": 0, "rmse": None, "mae": None}
        else:
            per_action[ACTION_NAMES[action_id]] = {
                "count": int(np.sum(mask)),
                "rmse": float(rmse(predicted[mask], target[mask])),
                "mae": float(mae(predicted[mask], target[mask])),
            }

    q25 = float(np.quantile(uncertainty, 0.25))
    q75 = float(np.quantile(uncertainty, 0.75))
    bottom = errors[uncertainty <= q25]
    top = errors[uncertainty >= q75]
    if bottom.size == 0 or top.size == 0:
        raise RuntimeError("uncertainty quartiles are empty")

    return {
        "completed_count": int(len(completed)),
        "incomplete_count": int(len(rows) - len(completed)),
        "rmse": float(rmse(predicted, target)),
        "mae": float(mae(predicted, target)),
        "uncertainty_error_spearman": float(spearman(uncertainty, errors)),
        "bottom_uncertainty_quartile_mean_error": float(np.mean(bottom)),
        "top_uncertainty_quartile_mean_error": float(np.mean(top)),
        "per_action": per_action,
        "canonical_action_counts": {str(i): int(counts.get(i, 0)) for i in range(4)},
    }


def _h4_windows(rows: list[dict]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    groups: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("model_alias", "")), int(row["episode_ordinal"]), str(row["agent_name"]))
        groups[key].append(row)
    windows = []
    for items in groups.values():
        ordered = sorted(items, key=lambda row: int(row["decision_index"]))
        for start in range(0, max(0, len(ordered) - 3)):
            seq = ordered[start : start + 4]
            indices = [int(item["decision_index"]) for item in seq]
            if indices != list(range(indices[0], indices[0] + 4)):
                continue
            if not all(bool(item.get("action_completed")) for item in seq):
                continue
            initial = np.asarray(seq[0]["state"], dtype=np.float32)
            actions = np.asarray([int(item["canonical_action_id"]) for item in seq], dtype=np.int64)
            target = np.asarray(seq[-1]["next_state"], dtype=np.float32)
            windows.append((initial, actions, target))
    return windows


def _probe_h4(model, rows: list[dict]) -> dict:
    windows = _h4_windows(rows)
    if not windows:
        return {"window_count": 0, "rmse": None, "uncertainty_error_spearman": None}
    initial = np.stack([item[0] for item in windows])
    actions = np.stack([item[1] for item in windows])
    target = np.stack([item[2] for item in windows]).astype(np.float64)
    member_final = []
    for member_index in range(len(model.models)):
        current = torch.as_tensor(initial, dtype=torch.float32, device=model.device)
        for step in range(4):
            action_t = torch.as_tensor(actions[:, step], dtype=torch.long, device=model.device)
            current = model.predict_member_mean_tensor(member_index, current, action_t)
        member_final.append(current.detach().cpu().numpy())
    members = np.stack(member_final, axis=0).astype(np.float64)
    predicted = np.mean(members, axis=0)
    uncertainty = np.mean(np.std(members, axis=0, ddof=0), axis=1)
    errors = sample_rmse(predicted, target)
    return {
        "window_count": int(len(windows)),
        "rmse": float(rmse(predicted, target)),
        "mae": float(mae(predicted, target)),
        "uncertainty_error_spearman": float(spearman(uncertainty, errors)),
    }


def audit_probe_rows(
    *,
    rows: list[dict],
    model: BootstrapProbabilisticWorldModel,
    reference: dict,
    validation_one_step_rmse: float,
) -> dict:
    probe_states = np.stack([np.asarray(row["state"], dtype=np.float32) for row in rows])
    normalizer = model._require_fitted()
    probe_z = normalizer.normalize_np(probe_states)
    knn = _knn_distances(probe_z, reference["train_z"], k=KNN_K)
    threshold = float(reference["threshold"])
    ood_fraction = float(np.mean(knn > threshold))
    z_rms = np.sqrt(np.mean(np.square(probe_z.astype(np.float64)), axis=1))

    one = _probe_one_step(model, rows)
    h4 = _probe_h4(model, rows)
    ratio = float(one["rmse"] / float(validation_one_step_rmse))
    action_counts = one["canonical_action_counts"]
    coverage_complete = all(int(action_counts[str(i)]) > 0 for i in range(4))

    finite_values = [
        one["rmse"],
        one["mae"],
        one["uncertainty_error_spearman"],
        one["bottom_uncertainty_quartile_mean_error"],
        one["top_uncertainty_quartile_mean_error"],
        ratio,
        ood_fraction,
        float(np.mean(knn)),
        float(np.mean(z_rms)),
    ]
    finite_ok = bool(np.isfinite(np.asarray(finite_values, dtype=np.float64)).all())
    shift_checks = {
        "probe_to_validation_rmse_ratio": ratio,
        "rmse_ratio_ok": bool(ratio <= MAX_PROBE_VALIDATION_RMSE_RATIO),
        "uncertainty_error_spearman": float(one["uncertainty_error_spearman"]),
        "uncertainty_spearman_positive": bool(one["uncertainty_error_spearman"] > 0.0),
        "bottom_uncertainty_quartile_mean_error": float(one["bottom_uncertainty_quartile_mean_error"]),
        "top_uncertainty_quartile_mean_error": float(one["top_uncertainty_quartile_mean_error"]),
        "uncertainty_quartile_order_ok": bool(
            one["top_uncertainty_quartile_mean_error"]
            > one["bottom_uncertainty_quartile_mean_error"]
        ),
        "ood_fraction": ood_fraction,
        "ood_fraction_ok": bool(ood_fraction <= MAX_OOD_FRACTION),
        "finite_ok": finite_ok,
    }
    hard_shift_pass = bool(
        shift_checks["rmse_ratio_ok"]
        and shift_checks["uncertainty_spearman_positive"]
        and shift_checks["uncertainty_quartile_order_ok"]
        and shift_checks["ood_fraction_ok"]
        and shift_checks["finite_ok"]
    )
    if hard_shift_pass and coverage_complete:
        status = "PASS"
    elif hard_shift_pass and not coverage_complete:
        status = "COVERAGE_INCOMPLETE"
    else:
        status = "FAIL"

    return {
        "status": status,
        "pass": status == "PASS",
        "coverage_complete": bool(coverage_complete),
        "probe_record_count": int(len(rows)),
        "validation_one_step_rmse": float(validation_one_step_rmse),
        "one_step": one,
        "h4_diagnostic": h4,
        "ood": {
            "knn_k": int(KNN_K),
            "threshold": threshold,
            "fraction_above_threshold": ood_fraction,
            "knn_mean": float(np.mean(knn)),
            "knn_p95": float(np.percentile(knn, 95)),
            "normalized_z_rms_mean": float(np.mean(z_rms)),
            "normalized_z_rms_p95": float(np.percentile(z_rms, 95)),
        },
        "gate_checks": shift_checks,
    }


def run_b0_2_audit(
    *,
    probe_paths: Iterable[str | Path],
    train_replay: str | Path = DEFAULT_TRAIN_REPLAY,
    validation_replay: str | Path = DEFAULT_VALIDATION_REPLAY,
    world_model_path: str | Path = DEFAULT_WORLD_MODEL,
    device: str = "cpu",
) -> dict:
    train = load_replay_jsonl(resolve_path(train_replay))
    validation = load_replay_jsonl(resolve_path(validation_replay))
    validate_frozen_split(train, "train")
    validate_frozen_split(validation, "validation")
    model = BootstrapProbabilisticWorldModel.load_checkpoint(resolve_path(world_model_path), device=device)
    if model.config.target_mode != "absolute" or model.config.state_dim != 27 or model.config.n_actions != 4:
        raise RuntimeError("B0.2 requires frozen absolute D27/A4 world model")
    if model.config.ensemble_size != 5:
        raise RuntimeError("B0.2 requires frozen M5 ensemble")

    validation_dataset = build_world_model_dataset(complete(validation))
    validation_prediction = model.predict_ensemble(validation_dataset.states, validation_dataset.actions)
    validation_rmse = float(
        rmse(np.asarray(validation_prediction["mean"]), validation_dataset.next_states)
    )
    reference = build_ood_reference(
        model=model,
        train_transitions=train,
        validation_transitions=validation,
    )

    rows = []
    sources = []
    for path in probe_paths:
        resolved = resolve_path(path)
        part = load_probe_jsonl(resolved)
        rows.extend(part)
        sources.append({"path": str(resolved), "records": len(part)})
    result = audit_probe_rows(
        rows=rows,
        model=model,
        reference=reference,
        validation_one_step_rmse=validation_rmse,
    )
    result["contract"] = {
        "phase": "gate_b0_2_policy_induced_ood",
        "train_only_probe": True,
        "knn_k": KNN_K,
        "validation_percentile": VALIDATION_PERCENTILE,
        "max_ood_fraction": MAX_OOD_FRACTION,
        "max_probe_validation_rmse_ratio": MAX_PROBE_VALIDATION_RMSE_RATIO,
        "world_model_target": "absolute",
        "ensemble_size": 5,
    }
    result["sources"] = sources
    result["reference"] = {
        key: value for key, value in reference.items() if key != "train_z"
    }
    return result


def _next_stage(current_count: int) -> int | None:
    for target in (2000, 5000, 10000, 20000):
        if int(target) > int(current_count):
            return int(target)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate B0.2 policy-induced OOD/model-exploitation audit")
    parser.add_argument("--model-alias", choices=FROZEN_VARIANTS)
    parser.add_argument("--union", action="store_true")
    parser.add_argument("--provisional-root", default=DEFAULT_PROVISIONAL_ROOT)
    parser.add_argument("--train-replay", default=DEFAULT_TRAIN_REPLAY)
    parser.add_argument("--validation-replay", default=DEFAULT_VALIDATION_REPLAY)
    parser.add_argument("--world-model", default=DEFAULT_WORLD_MODEL)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if bool(args.union) == bool(args.model_alias):
        raise SystemExit("choose exactly one of --model-alias or --union")
    provisional_root = resolve_path(args.provisional_root)
    if args.union:
        aliases = list(FROZEN_VARIANTS)
        out_name = "union.json"
    else:
        aliases = [str(args.model_alias)]
        out_name = f"{args.model_alias}.json"
    probe_paths = [provisional_root / alias / "probe_transitions.jsonl" for alias in aliases]
    report = run_b0_2_audit(
        probe_paths=probe_paths,
        train_replay=args.train_replay,
        validation_replay=args.validation_replay,
        world_model_path=args.world_model,
        device=args.device,
    )
    if report["status"] == "COVERAGE_INCOMPLETE":
        report["recommended_next_stage"] = _next_stage(report["probe_record_count"])
    else:
        report["recommended_next_stage"] = None

    out = resolve_path(args.out_root) / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("[GATE B0.2 POLICY-INDUCED OOD AUDIT]")
    print("sources:", aliases)
    print("probe records:", report["probe_record_count"])
    print("validation one-step RMSE:", report["validation_one_step_rmse"])
    print("probe one-step RMSE:", report["one_step"]["rmse"])
    print("probe/validation ratio:", report["gate_checks"]["probe_to_validation_rmse_ratio"])
    print("canonical actions:", report["one_step"]["canonical_action_counts"])
    print("uncertainty-error Spearman:", report["gate_checks"]["uncertainty_error_spearman"])
    print("quartile bottom/top:", report["gate_checks"]["bottom_uncertainty_quartile_mean_error"], report["gate_checks"]["top_uncertainty_quartile_mean_error"])
    print("OOD fraction:", report["gate_checks"]["ood_fraction"])
    print("H4 windows/RMSE:", report["h4_diagnostic"]["window_count"], report["h4_diagnostic"]["rmse"])
    print("coverage_complete:", report["coverage_complete"])
    print("status:", report["status"])
    print("recommended_next_stage:", report["recommended_next_stage"])
    print("[OK] report:", out)
    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
