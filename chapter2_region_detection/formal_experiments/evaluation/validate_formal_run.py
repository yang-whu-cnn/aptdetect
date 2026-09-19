"""Validation gate for one formal v3 repeat directory."""

from __future__ import annotations

import json
import hashlib
from math import isclose
from pathlib import Path
from typing import Any

from formal_experiments.common.run_manifest import (
    FINAL_TEST_SEEDS,
    load_jsonl,
    validate_episode_schema,
    validate_manifest,
)
from formal_experiments.evaluation.metrics_v3 import aggregate_repeat


VALIDATOR_VERSION = "formal_integrity_v2"
ELIGIBILITY_SCHEMA = "cc4_v3_eligibility_report_v1"
VALIDATOR_IDENTITY = "formal_experiments.evaluation.validate_formal_run"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_eligibility(root: Path, *, passed: bool, formal: bool,
                       errors: list[str], warnings: list[str],
                       input_hashes: dict[str, str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": ELIGIBILITY_SCHEMA,
        "schema_version": 1,
        "validator": VALIDATOR_IDENTITY,
        "validator_version": VALIDATOR_VERSION,
        "status": "PASS" if passed else "FAIL",
        "eligibility": "PASS" if passed else "FAIL",
        "formal": bool(formal),
        "passed": bool(passed),
        "errors": list(errors),
        "warnings": list(warnings),
        "input_sha256": dict(sorted(input_hashes.items())),
        "validated_input_sha256": dict(sorted(input_hashes.items())),
    }
    path = root / "eligibility_report.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def validate_run_directory(path: str | Path, *, formal: bool = True) -> dict[str, Any]:
    root = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = root / "manifest.json"
    episodes_path = root / "episodes.jsonl"
    if not manifest_path.is_file():
        errors.append("missing manifest.json")
    if not episodes_path.is_file():
        errors.append("missing episodes.jsonl")
    metrics_path = root / "metrics.json"
    if formal:
        for filename in ("config.resolved.yaml", "stdout.log"):
            if not (root / filename).is_file():
                errors.append(f"missing {filename}")
        if not ((root / "decisions.jsonl").is_file() or (root / "decisions.jsonl.zst").is_file()):
            errors.append("missing decisions.jsonl or decisions.jsonl.zst")
        if not metrics_path.is_file():
            errors.append("missing metrics.json")
        if not ((root / "checkpoint.pt").is_file() or (root / "policy_spec.json").is_file()):
            errors.append("missing checkpoint.pt or policy_spec.json")
    input_hashes: dict[str, str] = {}
    if errors:
        _write_eligibility(root, passed=False, formal=formal, errors=errors,
                           warnings=warnings, input_hashes=input_hashes)
        return {"passed": False, "errors": errors, "warnings": warnings}

    input_hashes["manifest.json"] = _sha256_file(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid manifest.json: {exc}")
        _write_eligibility(root, passed=False, formal=formal, errors=errors,
                           warnings=warnings, input_hashes=input_hashes)
        return {"passed": False, "errors": errors, "warnings": warnings, "metrics": None}
    if not isinstance(manifest, dict):
        errors.append("manifest.json must contain an object")
        _write_eligibility(root, passed=False, formal=formal, errors=errors,
                           warnings=warnings, input_hashes=input_hashes)
        return {"passed": False, "errors": errors, "warnings": warnings, "metrics": None}
    errors.extend(validate_manifest(manifest, formal=formal))
    artifacts = manifest.get("artifact_sha256")
    if isinstance(artifacts, dict):
        root_resolved = root.resolve()
        for relative, expected_digest in artifacts.items():
            if not isinstance(relative, str) or not relative.strip():
                continue
            relative_path = Path(relative)
            if relative_path.is_absolute():
                errors.append(f"artifact path must be relative: {relative!r}")
                continue
            candidate = (root_resolved / relative_path).resolve()
            if root_resolved not in candidate.parents:
                errors.append(f"artifact path escapes run directory: {relative!r}")
                continue
            if not candidate.is_file():
                errors.append(f"bound artifact is missing: {relative}")
                continue
            actual_digest = _sha256_file(candidate)
            input_hashes[relative] = actual_digest
            if actual_digest != expected_digest:
                errors.append(f"artifact SHA256 mismatch: {relative}")
    try:
        episodes = load_jsonl(episodes_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"invalid episodes.jsonl: {exc}")
        episodes = None
    raw_expected_ticks = manifest.get("episode_ticks")
    expected_ticks = raw_expected_ticks if isinstance(raw_expected_ticks, int) and not isinstance(raw_expected_ticks, bool) and raw_expected_ticks > 0 else 500
    expected_count = len(FINAL_TEST_SEEDS) if formal else len(manifest.get("test_episode_seeds", []))
    if episodes is not None:
        if len(episodes) != expected_count:
            errors.append(f"expected {expected_count} episodes, found {len(episodes)}")
        for index, episode in enumerate(episodes):
            errors.extend(f"episode[{index}]: {error}" for error in validate_episode_schema(episode, expected_ticks=expected_ticks))
        seeds = [episode.get("episode_seed") for episode in episodes]
        if seeds != list(manifest.get("test_episode_seeds", [])):
            errors.append("episode seeds/order do not match manifest")
        if len(set(seeds)) != len(seeds):
            errors.append("episode seeds are not unique")
    metrics = None
    if episodes is not None:
        try:
            metrics = aggregate_repeat(episodes, expected_ticks=expected_ticks).to_dict()
        except (TypeError, ValueError) as exc:
            errors.append(f"metric recomputation failed: {exc}")
    if formal and metrics is not None and metrics["recovery_precision"] is None:
        errors.append("recovery precision denominator is zero; formal review required")
    elif not formal and metrics is not None and metrics["recovery_precision"] is None:
        warnings.append(
            "development run has no completed recovery interventions; recovery_precision is NA"
        )
    if metrics is not None and metrics_path.is_file():
        try:
            recorded = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid metrics.json: {exc}")
            recorded = None
        if not isinstance(recorded, dict):
            errors.append("metrics.json must contain an object")
        else:
            for name, expected in metrics.items():
                actual = recorded.get(name)
                if expected is None:
                    equal = actual is None
                elif isinstance(expected, float):
                    equal = isinstance(actual, (int, float)) and isclose(
                        float(actual), expected, rel_tol=0.0, abs_tol=1e-9
                    )
                else:
                    equal = actual == expected
                if not equal:
                    errors.append(f"metrics.json mismatch for {name}")
    passed = not errors
    _write_eligibility(root, passed=passed, formal=formal, errors=errors,
                       warnings=warnings, input_hashes=input_hashes)
    return {"passed": passed, "errors": errors, "warnings": warnings, "metrics": metrics}
