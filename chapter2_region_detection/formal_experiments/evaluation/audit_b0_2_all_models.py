from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from formal_experiments.evaluation.audit_b0_2_policy_ood import (
    DEFAULT_OUT_ROOT,
    DEFAULT_PROVISIONAL_ROOT,
    DEFAULT_TRAIN_REPLAY,
    DEFAULT_VALIDATION_REPLAY,
    DEFAULT_WORLD_MODEL,
    FROZEN_VARIANTS,
    KNN_K,
    _knn_distances,
    _probe_one_step,
    build_ood_reference,
    load_probe_jsonl,
    resolve_path,
    run_b0_2_audit,
)
from formal_experiments.evaluation.validate_bootstrap_world_model import (
    load_replay_jsonl,
    validate_frozen_split,
)
from formal_experiments.training.bootstrap_world_model import BootstrapProbabilisticWorldModel


TAIL_FRACTION = 0.25


def _tail_diagnostic(
    *,
    rows: list[dict],
    model: BootstrapProbabilisticWorldModel,
    reference: dict,
    validation_rmse: float,
) -> dict:
    n = len(rows)
    tail_count = max(1, int(math.ceil(n * TAIL_FRACTION)))
    tail = rows[-tail_count:]
    states = np.stack([np.asarray(row["state"], dtype=np.float32) for row in tail])
    z = model._require_fitted().normalize_np(states)
    knn = _knn_distances(z, reference["train_z"], k=KNN_K)
    one = _probe_one_step(model, tail)
    return {
        "diagnostic_only": True,
        "affects_gate": False,
        "tail_fraction": float(TAIL_FRACTION),
        "tail_record_count": int(tail_count),
        "one_step_rmse": float(one["rmse"]),
        "probe_to_validation_rmse_ratio": float(one["rmse"] / validation_rmse),
        "uncertainty_error_spearman": float(one["uncertainty_error_spearman"]),
        "bottom_uncertainty_quartile_mean_error": float(
            one["bottom_uncertainty_quartile_mean_error"]
        ),
        "top_uncertainty_quartile_mean_error": float(
            one["top_uncertainty_quartile_mean_error"]
        ),
        "canonical_action_counts": dict(one["canonical_action_counts"]),
        "ood_fraction": float(np.mean(knn > float(reference["threshold"]))),
        "knn_mean": float(np.mean(knn)),
        "knn_p95": float(np.percentile(knn, 95)),
    }


def run_all_model_audit(
    *,
    provisional_root: str | Path = DEFAULT_PROVISIONAL_ROOT,
    train_replay: str | Path = DEFAULT_TRAIN_REPLAY,
    validation_replay: str | Path = DEFAULT_VALIDATION_REPLAY,
    world_model_path: str | Path = DEFAULT_WORLD_MODEL,
    device: str = "cpu",
) -> dict:
    root = resolve_path(provisional_root)
    per_model = {}
    probe_paths = {}
    for alias in FROZEN_VARIANTS:
        path = root / alias / "probe_transitions.jsonl"
        probe_paths[alias] = path
        per_model[alias] = run_b0_2_audit(
            probe_paths=[path],
            train_replay=train_replay,
            validation_replay=validation_replay,
            world_model_path=world_model_path,
            device=device,
        )

    # Pooled/union metrics are diagnostic only. They can never override a failing
    # per-model audit because the shared WM must be reliable for every formal LLM.
    pooled = run_b0_2_audit(
        probe_paths=[probe_paths[alias] for alias in FROZEN_VARIANTS],
        train_replay=train_replay,
        validation_replay=validation_replay,
        world_model_path=world_model_path,
        device=device,
    )
    pooled["diagnostic_only"] = True
    pooled["affects_gate"] = False

    statuses = [str(per_model[alias]["status"]) for alias in FROZEN_VARIANTS]
    if any(status == "FAIL" for status in statuses):
        overall = "FAIL"
    elif any(status == "COVERAGE_INCOMPLETE" for status in statuses):
        overall = "COVERAGE_INCOMPLETE"
    elif all(status == "PASS" for status in statuses):
        overall = "PASS"
    else:
        raise RuntimeError(f"unexpected per-model B0.2 statuses: {statuses}")

    train = load_replay_jsonl(resolve_path(train_replay))
    validation = load_replay_jsonl(resolve_path(validation_replay))
    validate_frozen_split(train, "train")
    validate_frozen_split(validation, "validation")
    model = BootstrapProbabilisticWorldModel.load_checkpoint(
        resolve_path(world_model_path), device=device
    )
    reference = build_ood_reference(
        model=model,
        train_transitions=train,
        validation_transitions=validation,
    )
    validation_rmse = float(per_model[FROZEN_VARIANTS[0]]["validation_one_step_rmse"])
    tail = {}
    for alias in FROZEN_VARIANTS:
        rows = load_probe_jsonl(probe_paths[alias])
        tail[alias] = _tail_diagnostic(
            rows=rows,
            model=model,
            reference=reference,
            validation_rmse=validation_rmse,
        )

    return {
        "phase": "gate_b0_2_all_models",
        "shared_world_model_requires_every_variant_pass": True,
        "variants": list(FROZEN_VARIANTS),
        "per_model": per_model,
        "tail_25pct_diagnostic": tail,
        "pooled_union_diagnostic": pooled,
        "status": overall,
        "pass": overall == "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate B0.2 all-model audit: per-model gates plus pooled/tail diagnostics"
    )
    parser.add_argument("--provisional-root", default=DEFAULT_PROVISIONAL_ROOT)
    parser.add_argument("--train-replay", default=DEFAULT_TRAIN_REPLAY)
    parser.add_argument("--validation-replay", default=DEFAULT_VALIDATION_REPLAY)
    parser.add_argument("--world-model", default=DEFAULT_WORLD_MODEL)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    report = run_all_model_audit(
        provisional_root=args.provisional_root,
        train_replay=args.train_replay,
        validation_replay=args.validation_replay,
        world_model_path=args.world_model,
        device=args.device,
    )
    out = resolve_path(args.out_root) / "all_models.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("[GATE B0.2 ALL-MODEL AUDIT]")
    for alias in FROZEN_VARIANTS:
        item = report["per_model"][alias]
        tail = report["tail_25pct_diagnostic"][alias]
        print(
            f"{alias}: status={item['status']} records={item['probe_record_count']} "
            f"rmse_ratio={item['gate_checks']['probe_to_validation_rmse_ratio']:.6f} "
            f"ood={item['gate_checks']['ood_fraction']:.6f} "
            f"tail_rmse_ratio={tail['probe_to_validation_rmse_ratio']:.6f} "
            f"tail_ood={tail['ood_fraction']:.6f}"
        )
    print("pooled_union_diagnostic_status:", report["pooled_union_diagnostic"]["status"])
    print("overall_status:", report["status"])
    print("pass:", report["pass"])
    print("[OK] report:", out)
    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
