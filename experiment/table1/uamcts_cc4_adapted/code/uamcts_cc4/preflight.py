"""Strict, read-only UAMCTS artifact gate used by every real runtime."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import (
    DEFAULT_NORMALIZER_SIDECAR,
    DEFAULT_PRIOR_ENTROPY_SIDECAR,
    DEFAULT_PROGRESS_SIDECAR,
    DEFAULT_TRAIN_REPLAY,
    DEFAULT_VALIDATION_REPLAY,
    FrozenProgressEnsemble,
    FrozenUncertaintyNormalizers,
    UAMCTSPrototypeRetriever,
    validate_normalizer_sidecar,
    validate_prior_entropy_sidecar,
    validate_progress_sidecar,
    validate_prototype_chain,
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
DEFAULT_PROTOTYPE_COVERAGE = "outputs/priorrl_cc4/prototypes/frozen_prototype_coverage.json"
DEFAULT_PROTOTYPE_PROVENANCE = "outputs/priorrl_cc4/prototypes/frozen_prototype_provenance.json"


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _validate_prior_chain(*, prototype_path: Path, coverage_path: Path,
                          provenance_path: Path, prior_entropy_path: Path,
                          prior_entropy_sidecar_path: Path,
                          validation_replay_path: Path,
                          require_sidecars: bool) -> tuple[dict, list[str]]:
    """Validate prototype -> validation entropy without reading any test truth."""
    errors: list[str] = []
    artifacts: dict[str, str] = {}
    if require_sidecars:
        chain_artifacts, chain_errors = validate_prototype_chain(
            prototype_path=prototype_path, coverage_path=coverage_path,
            provenance_path=provenance_path,
        )
        artifacts.update({"prototype_prior_sha256": chain_artifacts.get("prototype_sha256", ""),
                          **{key: value for key, value in chain_artifacts.items()
                             if key != "prototype_sha256"}})
        errors.extend(chain_errors)
    else:
        try:
            prototype = UAMCTSPrototypeRetriever(prototype_path)
            artifacts["prototype_prior_sha256"] = prototype.file_sha256
        except Exception as exc:
            return artifacts, [f"prototype prior artifact: {exc}"]
    try:
        prototype = UAMCTSPrototypeRetriever(prototype_path)
    except Exception as exc:
        errors.append(f"prototype prior artifact: {exc}")
        return artifacts, errors
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
    if require_sidecars:
        entropy_sidecar, sidecar_errors = validate_prior_entropy_sidecar(
            prior_entropy_sidecar_path,
            entropy_sha256=artifacts.get("prior_entropy_sha256", ""),
            prototype_artifact_sha256=prototype.file_sha256,
            prototype_sha256=prototype.payload.get("prototype_sha256"),
            validation_replay_sha256=sha256_file(validation_replay_path)
            if validation_replay_path.is_file() else None,
        )
        errors.extend(sidecar_errors)
        if entropy_sidecar is not None:
            artifacts["prior_entropy_sidecar_sha256"] = sha256_file(prior_entropy_sidecar_path)
    return artifacts, errors


def run_preflight(
    *,
    world_model=DEFAULT_WORLD_MODEL,
    reward_model=DEFAULT_REWARD_MODEL,
    progress=DEFAULT_PROGRESS,
    normalizers=DEFAULT_NORMALIZERS,
    prototype_artifact=DEFAULT_PROTOTYPES,
    prototype_coverage_artifact=DEFAULT_PROTOTYPE_COVERAGE,
    prototype_provenance_artifact=DEFAULT_PROTOTYPE_PROVENANCE,
    prior_entropy_artifact=DEFAULT_PRIOR_ENTROPY,
    progress_sidecar=DEFAULT_PROGRESS_SIDECAR,
    prior_entropy_sidecar=DEFAULT_PRIOR_ENTROPY_SIDECAR,
    normalizer_sidecar=DEFAULT_NORMALIZER_SIDECAR,
    train_replay=DEFAULT_TRAIN_REPLAY,
    validation_replay=DEFAULT_VALIDATION_REPLAY,
    require_sidecars=False,
    offline_prior_artifact=None,
):
    errors: list[str] = []
    artifacts: dict[str, str] = {}
    if offline_prior_artifact is not None:
        prior_entropy_artifact = offline_prior_artifact
    paths = {
        "world_model": resolve(world_model), "reward_model": resolve(reward_model),
        "progress": resolve(progress), "normalizers": resolve(normalizers),
        "progress_sidecar": resolve(progress_sidecar),
        "normalizer_sidecar": resolve(normalizer_sidecar),
        "prototype": resolve(prototype_artifact),
        "prototype_coverage": resolve(prototype_coverage_artifact),
        "prototype_provenance": resolve(prototype_provenance_artifact),
        "prior_entropy": resolve(prior_entropy_artifact),
        "prior_entropy_sidecar": resolve(prior_entropy_sidecar),
        "train_replay": resolve(train_replay), "validation_replay": resolve(validation_replay),
    }
    for key in ("world_model", "reward_model"):
        path = paths[key]
        if not path.is_file(): errors.append(f"missing {key}: {path}")
        else: artifacts[key + "_sha256"] = sha256_file(path)
    for key in (
        "progress", "normalizers", "prototype", "prototype_coverage",
        "prototype_provenance", "prior_entropy", "progress_sidecar",
        "normalizer_sidecar", "prior_entropy_sidecar",
    ):
        path = paths[key]
        if path.is_file():
            artifacts[key + "_sha256"] = sha256_file(path)
        elif require_sidecars or key not in {"progress_sidecar", "normalizer_sidecar", "prior_entropy_sidecar"}:
            errors.append(f"missing {key}: {path}")
    try:
        progress_model = FrozenProgressEnsemble(
            paths["progress"],
            sidecar_path=paths["progress_sidecar"] if require_sidecars else None,
            train_replay_path=paths["train_replay"] if require_sidecars else None,
            validation_replay_path=paths["validation_replay"] if require_sidecars else None,
            require_sidecar=require_sidecars,
        )
        artifacts["progress_sha256"] = progress_model.sha256
        if getattr(progress_model, "sidecar_sha256", None):
            artifacts["progress_sidecar_sha256"] = progress_model.sidecar_sha256
    except Exception as exc:
        errors.append(f"progress artifact: {exc}")
    try:
        normalizer = FrozenUncertaintyNormalizers(paths["normalizers"])
        artifacts["normalizers_sha256"] = normalizer.sha256
    except Exception as exc:
        errors.append(f"normalizer artifact: {exc}")
    prior_artifacts, prior_errors = _validate_prior_chain(
        prototype_path=paths["prototype"], coverage_path=paths["prototype_coverage"],
        provenance_path=paths["prototype_provenance"], prior_entropy_path=paths["prior_entropy"],
        prior_entropy_sidecar_path=paths["prior_entropy_sidecar"],
        validation_replay_path=paths["validation_replay"], require_sidecars=require_sidecars,
    )
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
        if require_sidecars:
            _, normalizer_sidecar_errors = validate_normalizer_sidecar(
                paths["normalizer_sidecar"],
                normalizer_sha256=normalizer.sha256,
                world_model_sha256=artifacts.get("world_model_sha256", ""),
                reward_model_sha256=artifacts.get("reward_model_sha256", ""),
                progress_sha256=artifacts.get("progress_sha256", ""),
                prior_entropy_sha256=artifacts.get("prior_entropy_sha256", ""),
                validation_replay_sha256=sha256_file(paths["validation_replay"])
                if paths["validation_replay"].is_file() else None,
            )
            errors.extend(normalizer_sidecar_errors)
            if paths["normalizer_sidecar"].is_file():
                artifacts["normalizer_sidecar_sha256"] = sha256_file(paths["normalizer_sidecar"])
    artifacts["frozen_artifact_sha256"] = dict(artifacts)
    return {"schema": "uamcts_real_gate_v2", "eligible": not errors,
            "formal_result_eligible": False, "artifacts": artifacts, "errors": errors}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-model", default=DEFAULT_WORLD_MODEL)
    parser.add_argument("--reward-model", default=DEFAULT_REWARD_MODEL)
    parser.add_argument("--progress", default=DEFAULT_PROGRESS)
    parser.add_argument("--normalizers", default=DEFAULT_NORMALIZERS)
    parser.add_argument("--prototype-artifact", default=DEFAULT_PROTOTYPES)
    parser.add_argument("--prototype-coverage-artifact", default=DEFAULT_PROTOTYPE_COVERAGE)
    parser.add_argument("--prototype-provenance-artifact", default=DEFAULT_PROTOTYPE_PROVENANCE)
    parser.add_argument("--prior-entropy-artifact", default=DEFAULT_PRIOR_ENTROPY)
    parser.add_argument("--progress-sidecar", default=DEFAULT_PROGRESS_SIDECAR)
    parser.add_argument("--prior-entropy-sidecar", default=DEFAULT_PRIOR_ENTROPY_SIDECAR)
    parser.add_argument("--normalizer-sidecar", default=DEFAULT_NORMALIZER_SIDECAR)
    parser.add_argument("--offline-prior-artifact", help=argparse.SUPPRESS)
    parser.add_argument("--report", default="outputs/uamcts_cc4/preflight.json")
    args = parser.parse_args(argv)
    result = run_preflight(world_model=args.world_model, reward_model=args.reward_model,
        progress=args.progress, normalizers=args.normalizers,
        prototype_artifact=args.prototype_artifact,
        prototype_coverage_artifact=args.prototype_coverage_artifact,
        prototype_provenance_artifact=args.prototype_provenance_artifact,
        prior_entropy_artifact=args.prior_entropy_artifact,
        progress_sidecar=args.progress_sidecar,
        prior_entropy_sidecar=args.prior_entropy_sidecar,
        normalizer_sidecar=args.normalizer_sidecar,
        offline_prior_artifact=args.offline_prior_artifact, require_sidecars=True)
    out = resolve(args.report); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["eligible"] else 2


if __name__ == "__main__": raise SystemExit(main())
