"""Strict, read-only UAMCTS artifact gate used by every real runtime."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import (
    FrozenProgressEnsemble,
    FrozenUncertaintyNormalizers,
    UAMCTSPrototypeRetriever,
    VALIDATION_SEEDS,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORLD_MODEL = "outputs/world_model_final_20260917/a4_5b/world_model_absolute.pt"
DEFAULT_REWARD_MODEL = "outputs/world_model_final_20260917/a4_5c/response_reward_predictor.pt"
DEFAULT_PROGRESS = "outputs/uamcts_cc4/progress/progress_ensemble_train_only.pt"
DEFAULT_NORMALIZERS = "outputs/uamcts_cc4/calibration/uamcts_uncertainty_normalizers_frozen.pt"
DEFAULT_PROTOTYPES = "outputs/priorrl_cc4/prototypes/frozen_prototypes.json"
DEFAULT_PRIOR_ENTROPY = "outputs/uamcts_cc4/calibration/validation_prior_entropy.json"


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _validate_prior_chain(*, prototype_path: Path, prior_entropy_path: Path) -> tuple[dict, list[str]]:
    """Validate prototype -> validation entropy without reading any test truth."""
    errors: list[str] = []
    artifacts: dict[str, str] = {}
    try:
        prototype = UAMCTSPrototypeRetriever(prototype_path)
        artifacts["prototype_prior_sha256"] = prototype.file_sha256
    except Exception as exc:
        return artifacts, [f"prototype prior artifact: {exc}"]
    try:
        entropy = json.loads(prior_entropy_path.read_text(encoding="utf-8"))
        artifacts["prior_entropy_sha256"] = sha256_file(prior_entropy_path)
    except Exception as exc:
        return artifacts, [f"validation prior entropy artifact: {exc}"]

    expected = {
        "schema": "uamcts_validation_prior_entropy_v1",
        "split": "validation",
        "allowed_seeds": sorted(VALIDATION_SEEDS),
        "provider_calls": 0,
        "prototype_artifact_sha256": prototype.file_sha256,
        "prototype_sha256": prototype.payload.get("prototype_sha256"),
    }
    for key, value in expected.items():
        if entropy.get(key) != value:
            errors.append(f"prior entropy {key} provenance mismatch")
    aligned = entropy.get("aligned_unique_records")
    hits, misses = entropy.get("hits"), entropy.get("misses")
    if not isinstance(aligned, int) or aligned <= 0 or hits != aligned or misses != 0:
        errors.append("prior entropy must have complete frozen validation coverage")
    values = entropy.get("entropy")
    if not isinstance(values, list) or len(values) != aligned or any(value is None for value in values):
        errors.append("prior entropy values are incomplete or misaligned")
    return artifacts, errors


def run_preflight(
    *,
    world_model=DEFAULT_WORLD_MODEL,
    reward_model=DEFAULT_REWARD_MODEL,
    progress=DEFAULT_PROGRESS,
    normalizers=DEFAULT_NORMALIZERS,
    prototype_artifact=DEFAULT_PROTOTYPES,
    prior_entropy_artifact=DEFAULT_PRIOR_ENTROPY,
    offline_prior_artifact=None,
):
    errors: list[str] = []
    artifacts: dict[str, str] = {}
    if offline_prior_artifact is not None:
        prior_entropy_artifact = offline_prior_artifact
    paths = {
        "world_model": resolve(world_model), "reward_model": resolve(reward_model),
        "progress": resolve(progress), "normalizers": resolve(normalizers),
        "prototype": resolve(prototype_artifact), "prior_entropy": resolve(prior_entropy_artifact),
    }
    for key in ("world_model", "reward_model"):
        path = paths[key]
        if not path.is_file(): errors.append(f"missing {key}: {path}")
        else: artifacts[key + "_sha256"] = sha256_file(path)
    try:
        progress_model = FrozenProgressEnsemble(paths["progress"])
        artifacts["progress_sha256"] = progress_model.sha256
    except Exception as exc:
        errors.append(f"progress artifact: {exc}")
    try:
        normalizer = FrozenUncertaintyNormalizers(paths["normalizers"])
        artifacts["normalizers_sha256"] = normalizer.sha256
    except Exception as exc:
        errors.append(f"normalizer artifact: {exc}")
    prior_artifacts, prior_errors = _validate_prior_chain(
        prototype_path=paths["prototype"], prior_entropy_path=paths["prior_entropy"])
    artifacts.update(prior_artifacts); errors.extend(prior_errors)
    if "normalizer" in locals():
        expected = {"progress_sha256": artifacts.get("progress_sha256"),
                    "world_model_sha256": artifacts.get("world_model_sha256"),
                    "prior_artifact_sha256": artifacts.get("prior_entropy_sha256")}
        for key, value in expected.items():
            if normalizer.payload.get(key) != value:
                errors.append(f"normalizer {key} provenance mismatch")
        aligned = normalizer.payload.get("validation_aligned_total")
        if (normalizer.payload.get("record_count") != aligned
                or normalizer.payload.get("prior_coverage_hits") != aligned
                or normalizer.payload.get("prior_coverage_misses") != 0):
            errors.append("normalizer prior coverage is not complete")
    return {"schema": "uamcts_real_gate_v2", "eligible": not errors,
            "formal_result_eligible": False, "artifacts": artifacts, "errors": errors}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-model", default=DEFAULT_WORLD_MODEL)
    parser.add_argument("--reward-model", default=DEFAULT_REWARD_MODEL)
    parser.add_argument("--progress", default=DEFAULT_PROGRESS)
    parser.add_argument("--normalizers", default=DEFAULT_NORMALIZERS)
    parser.add_argument("--prototype-artifact", default=DEFAULT_PROTOTYPES)
    parser.add_argument("--prior-entropy-artifact", default=DEFAULT_PRIOR_ENTROPY)
    parser.add_argument("--offline-prior-artifact", help=argparse.SUPPRESS)
    parser.add_argument("--report", default="outputs/uamcts_cc4/preflight.json")
    args = parser.parse_args(argv)
    result = run_preflight(world_model=args.world_model, reward_model=args.reward_model,
        progress=args.progress, normalizers=args.normalizers,
        prototype_artifact=args.prototype_artifact, prior_entropy_artifact=args.prior_entropy_artifact,
        offline_prior_artifact=args.offline_prior_artifact)
    out = resolve(args.report); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["eligible"] else 2


if __name__ == "__main__": raise SystemExit(main())
