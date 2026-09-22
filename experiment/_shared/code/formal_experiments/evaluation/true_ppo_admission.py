"""Read-only admission bridge for completed true-PPO formal repeats.

The true-PPO runner intentionally owns a stronger, runner-specific manifest and
eligibility schema than the older generic formal validator.  This module does
not rewrite either source file and does not invent the generic manifest fields
that the runner never recorded.  It writes a small derived pointer under an
``admission_derived`` namespace and revalidates the source on every read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from math import isclose, isfinite
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
from uuid import uuid4

from formal_experiments.common.run_manifest import (
    FINAL_TEST_SEEDS,
    POLICY_SEEDS,
    validate_episode_schema,
)
from formal_experiments.evaluation.metrics_v3 import aggregate_repeat


SOURCE_MANIFEST_SCHEMA = "cc4_v3_true_ppo_manifest_v1"
SOURCE_ELIGIBILITY_SCHEMA = "cc4_v3_true_ppo_eligibility_v1"
SOURCE_AGGREGATE_SCHEMA = "cc4_v3_true_ppo_aggregate_v1"
SOURCE_PROTOCOL = "cc4_v3_true_on_policy_clipped_ppo_v1"
BRIDGE_SCHEMA = "cc4_v3_true_ppo_admission_bridge_v1"
BRIDGE_REPORT_SCHEMA = "cc4_v3_true_ppo_admission_report_v1"
BRIDGE_STATUS = "PROVISIONAL"

CORE_SOURCE_FILES = (
    "manifest.json",
    "eligibility_report.json",
    "aggregate.json",
    "episodes.jsonl",
    "decisions.jsonl",
    "metrics.json",
)
ELIGIBILITY_FILE_BINDINGS = (
    "training_curve.jsonl",
    "episodes.jsonl",
    "decisions.jsonl",
    "metrics.json",
    "normalizer.json",
    "calibration_manifest.json",
    "train_summary.json",
    "validation_records.json",
    "journal_index.json",
    "prepare_test_authorization.json",
)
METRIC_FIELDS = (
    "cc4_official_reward",
    "operation_failure_penalty",
    "recovery_precision",
    "recovery_time_censored_mean",
    "recovery_time_completed_only_mean",
    "recovery_unrecovered_rate",
)
ROW_LABELS = {
    "rl_only": "RL-Only",
    "llm_rl": "LLM-RL",
    "wm_rl": "WM-RL",
    "lwm_rl": "LWM-RL",
    "delay_only": "Delay-Only",
    "fail_only": "Fail-Only",
}


class AdmissionError(ValueError):
    """A fail-closed true-PPO admission error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmissionError(message)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"invalid JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{path} must contain an object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                _require(isinstance(value, dict), f"{path}:{number} must contain an object")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"invalid JSONL {path}: {exc}") from exc
    return rows


def _bound_path(root: Path, relative: str) -> Path:
    _require(isinstance(relative, str) and relative.strip(), "artifact path must be non-empty")
    rel = Path(relative)
    _require(not rel.is_absolute() and ".." not in rel.parts, f"artifact path escapes source: {relative}")
    candidate = (root / rel).resolve()
    _require(root.resolve() in candidate.parents, f"artifact path escapes source: {relative}")
    _require(candidate.is_file(), f"bound source artifact is missing: {relative}")
    return candidate


def _completion_guard(source: Path) -> None:
    lock = source / ".true_ppo_writer.lock.json"
    _require(lock.is_file(), "source has no true-PPO writer lock/completion record")
    lock_payload = _json(lock)
    _require(lock_payload.get("status") != "ACTIVE", "source writer is ACTIVE; admission is forbidden")
    _require((source / "aggregate.json").is_file(), "source aggregate.json completion marker is missing")
    torn = [item for item in source.rglob("*") if item.is_file() and item.name.endswith((".tmp", ".resume_tmp"))]
    _require(not torn, f"source contains torn temporary artifact: {torn[0] if torn else ''}")


def _normalised_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    policy_seed = manifest.get("policy_seed")
    _require(type(policy_seed) is int and policy_seed in POLICY_SEEDS, "policy_seed is not one of 51001..51005")
    repeat_index = POLICY_SEEDS.index(policy_seed) + 1
    table_id = str(manifest.get("table_id", "")).lower()
    row_id = str(manifest.get("row_id", "")).lower()
    variant = str(manifest.get("variant", "")).lower()
    reward_mode = str(manifest.get("reward_mode", "")).lower().replace("-", "_")
    _require(table_id in {"table2", "table3"}, "true-PPO bridge only admits physical Table 2/3 rows")
    if table_id == "table2":
        _require(row_id in {"rl_only", "llm_rl", "wm_rl", "lwm_rl"}, "Table 2 row is outside the frozen whitelist")
        _require(variant == row_id, "Table 2 row_id/variant mismatch")
        _require(reward_mode == "full_reward", "Table 2 true-PPO source must use Full-Reward")
        component = row_id
        semantics = "cc4_v3_full_reward"
    else:
        _require(row_id in {"delay_only", "fail_only"}, "Table 3 Full-Reward is alias-only; physical row rejected")
        _require(variant == "lwm_rl", "Table 3 physical reward rows require variant=lwm_rl")
        _require(reward_mode == row_id, "Table 3 row_id/reward_mode mismatch")
        component = "lwm_rl"
        semantics = f"cc4_v3_{row_id}"
    figure = manifest.get("figure_artifacts", {}).get("training_curve")
    _require(isinstance(figure, Mapping), "source manifest has no training-curve declaration")
    relative = figure.get("relative_path", figure.get("path"))
    normalised_figure = dict(figure)
    normalised_figure["path"] = relative
    result = {
        "schema": "cc4_v3_true_ppo_admission_normalized_v1",
        "method": ROW_LABELS[row_id],
        "method_slug": row_id,
        "repeat_index": repeat_index,
        "training_seed": policy_seed,
        "run_mode": "formal",
        "formal_result_eligible": True,
        "table_id": table_id,
        "row_id": row_id,
        "reward_mode": reward_mode,
        "component_variant": component,
        "reward_semantics": semantics,
        "figure_artifacts": {"training_curve": normalised_figure},
    }
    if (table_id, row_id) == ("table2", "lwm_rl"):
        # This is a named downstream alias identity, not claimed source provenance.
        result["canonical_run_id"] = "lwm_full"
    if (table_id, row_id) == ("table3", "fail_only"):
        for field in ("fail_only_training_contract", "ofox_cache_audit"):
            _require(isinstance(manifest.get(field), Mapping), f"source manifest lacks {field}; cannot derive it")
            result[field] = dict(manifest[field])
    return result


def audit_true_ppo_source(source_dir: str | Path) -> dict[str, Any]:
    """Audit a completed source directory without writing to it."""
    source = Path(source_dir).resolve()
    _require(source.is_dir(), f"source directory is missing: {source}")
    _completion_guard(source)
    for name in CORE_SOURCE_FILES:
        _require((source / name).is_file(), f"source is missing {name}")

    manifest = _json(source / "manifest.json")
    eligibility = _json(source / "eligibility_report.json")
    aggregate = _json(source / "aggregate.json")
    _require(manifest.get("schema") == SOURCE_MANIFEST_SCHEMA, "source manifest schema mismatch")
    _require(manifest.get("protocol") == SOURCE_PROTOCOL, "source protocol mismatch")
    _require(manifest.get("formal_result_eligible") is True, "source is not formal-result eligible")
    _require(manifest.get("git_dirty") is False, "source manifest records a dirty Git snapshot")
    _require(eligibility.get("schema") == SOURCE_ELIGIBILITY_SCHEMA, "source eligibility schema mismatch")
    _require(eligibility.get("passed") is True, "source true-PPO eligibility is not PASS")
    _require(aggregate.get("schema") == SOURCE_AGGREGATE_SCHEMA and aggregate.get("status") == "PASS",
             "source aggregate completion gate is not PASS")

    source_sha = {name: sha256_file(source / name) for name in CORE_SOURCE_FILES}
    _require(aggregate.get("source_manifest_sha256") == source_sha["manifest.json"],
             "source aggregate manifest binding mismatch")
    _require(aggregate.get("source_eligibility_sha256") == source_sha["eligibility_report.json"],
             "source aggregate eligibility binding mismatch")
    _require(manifest.get("eligibility_report_sha256") == source_sha["eligibility_report.json"],
             "source manifest eligibility binding mismatch")
    _require(manifest.get("metrics_sha256") == source_sha["metrics.json"],
             "source manifest metrics binding mismatch")

    artifact_bindings = manifest.get("artifact_sha256")
    _require(isinstance(artifact_bindings, Mapping), "source manifest artifact_sha256 is missing")
    for name, expected in artifact_bindings.items():
        target = _bound_path(source, name)
        _require(sha256_file(target) == expected, f"source manifest artifact SHA mismatch: {name}")
    for name in ("episodes.jsonl", "decisions.jsonl", "metrics.json", "eligibility_report.json", "training_curve.jsonl"):
        _require(artifact_bindings.get(name) == sha256_file(source / name), f"source manifest does not bind {name}")

    input_bindings = eligibility.get("input_sha256")
    _require(isinstance(input_bindings, Mapping), "source eligibility input_sha256 is missing")
    for name in ELIGIBILITY_FILE_BINDINGS:
        target = _bound_path(source, name)
        _require(input_bindings.get(name) == sha256_file(target), f"source eligibility binding mismatch: {name}")
    frozen_before = eligibility.get("frozen_artifact_sha256_before")
    _require(isinstance(frozen_before, Mapping), "source eligibility frozen-artifact snapshot is missing")
    _require(frozen_before == eligibility.get("frozen_artifact_sha256_after"), "source frozen artifacts changed during test")
    _require(input_bindings.get("frozen_artifact_bundle") == canonical_sha256(frozen_before),
             "source frozen-artifact bundle binding mismatch")

    expected_splits = {
        "train_seeds": list(range(1000, 1032)),
        "calibration_seeds": list(range(3000, 3008)),
        "validation_seeds": list(range(2000, 2008)),
        "test_seeds": list(FINAL_TEST_SEEDS),
    }
    _require(all(eligibility.get(key) == value for key, value in expected_splits.items()),
             "source eligibility seed splits are not the frozen formal splits")
    _require(manifest.get("test_seeds") == list(FINAL_TEST_SEEDS), "source manifest test seeds mismatch")
    _require(manifest.get("episode_ticks") == 500, "source manifest episode_ticks must be 500")
    _require(eligibility.get("episode_count") == 100 and eligibility.get("expected_episode_count") == 100,
             "source eligibility episode count mismatch")
    _require(eligibility.get("expected_ticks") == 500, "source eligibility tick budget mismatch")
    _require(eligibility.get("provider_calls") == eligibility.get("cache_misses") == eligibility.get("test_time_updates") == 0,
             "source eligibility records forbidden test-time calls, misses, or updates")
    _require(eligibility.get("metrics_recomputed") is True, "source eligibility did not independently recompute metrics")

    episodes = _jsonl(source / "episodes.jsonl")
    _require(len(episodes) == 100, "source episodes.jsonl must contain exactly 100 episodes")
    for index, episode in enumerate(episodes):
        errors = validate_episode_schema(episode, expected_ticks=500)
        _require(not errors, f"source episode[{index}] is invalid: {'; '.join(errors)}")
    _require([row.get("episode_seed") for row in episodes] == list(FINAL_TEST_SEEDS),
             "source episode seeds/order mismatch")

    decisions = _jsonl(source / "decisions.jsonl")
    _require(decisions, "source decisions.jsonl is empty")
    expected_decisions = sum(int(row.get("decision_count", 0)) for row in episodes)
    _require(expected_decisions == len(decisions), "source episode/decision counts disagree")
    _require(all(type(row.get("episode_seed")) is int and row["episode_seed"] in FINAL_TEST_SEEDS for row in decisions),
             "source decision audit contains an invalid episode seed")

    recomputed = aggregate_repeat(episodes, expected_ticks=500).to_dict()
    stored_metrics = _json(source / "metrics.json")
    _require(stored_metrics.get("schema") == "cc4_v3_true_ppo_metrics_v1", "source metrics schema mismatch")
    recorded_mean = stored_metrics.get("mean")
    _require(isinstance(recorded_mean, Mapping), "source metrics mean is missing")
    for name, expected in recomputed.items():
        actual = recorded_mean.get(name)
        equal = actual is None if expected is None else isinstance(actual, (int, float)) and isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9
        )
        _require(equal, f"source metrics recomputation mismatch: {name}")
    _require(all(recomputed.get(name) is not None and isfinite(float(recomputed[name])) for name in METRIC_FIELDS),
             "source has NA/non-finite paper metric")

    curve = _jsonl(source / "training_curve.jsonl")
    declaration = eligibility.get("training_curve")
    _require(isinstance(declaration, Mapping), "source eligibility training-curve declaration is missing")
    _require(declaration.get("schema") == "cc4_v3_training_curve_v1" and declaration.get("split") == "train",
             "source training-curve schema/split mismatch")
    _require(declaration.get("sha256") == sha256_file(source / "training_curve.jsonl"),
             "source training-curve SHA mismatch")
    _require(declaration.get("record_count") == len(curve) and len(curve) >= 2,
             "source training-curve record count is invalid")
    steps = [row.get("environment_steps") for row in curve]
    _require(all(type(step) is int and step >= 0 for step in steps) and all(a < b for a, b in zip(steps, steps[1:])),
             "source training-curve step grid is invalid")
    _require(all(isinstance(row.get("training_objective_reward"), (int, float)) for row in curve),
             "source training curve lacks numeric objective rewards")

    normalized = _normalised_identity(manifest)
    # Detect a concurrent mutation after the audit without writing any source file.
    _require(source_sha == {name: sha256_file(source / name) for name in CORE_SOURCE_FILES},
             "source changed while admission audit was running")
    return {
        "source_dir": source,
        "source_sha256": source_sha,
        "source_manifest": manifest,
        "source_eligibility": eligibility,
        "normalized_manifest": normalized,
        "episodes": episodes,
        "metrics": recomputed,
        "training_curve_path": source / "training_curve.jsonl",
    }


def _bridge_payload(audit: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(audit["source_dir"])
    normalized = dict(audit["normalized_manifest"])
    return {
        "schema": BRIDGE_SCHEMA,
        "status": BRIDGE_STATUS,
        "source_kind": SOURCE_MANIFEST_SCHEMA,
        "source_run_dir": str(source),
        "source_sha256": dict(sorted(audit["source_sha256"].items())),
        "source_identity": {
            "protocol": audit["source_manifest"].get("protocol"),
            "code_commit": audit["source_manifest"].get("code_commit"),
            "git_dirty": audit["source_manifest"].get("git_dirty"),
            "policy_seed": audit["source_manifest"].get("policy_seed"),
            "variant": audit["source_manifest"].get("variant"),
            "reward_mode": audit["source_manifest"].get("reward_mode"),
            "table_id": audit["source_manifest"].get("table_id"),
            "row_id": audit["source_manifest"].get("row_id"),
        },
        "normalized_manifest": normalized,
        "derived_fields": {
            "repeat_index": "index of source policy_seed in frozen 51001..51005",
            "method_and_figure_identity": "frozen table/row mapping; not claimed as source-recorded provenance",
            "canonical_run_id": "table2/lwm_rl read-only alias identity only",
        },
        "formal_repeat_eligible": True,
        "paper_row_eligible": False,
        "paper_row_reason": "PROVISIONAL: one admitted repeat; paper row requires repeats 1..5 exactly once",
    }


def create_admission_bridge(source_dir: str | Path, destination_dir: str | Path) -> dict[str, Any]:
    """Create a non-overwriting derived bridge for one completed repeat."""
    destination = Path(destination_dir).resolve()
    _require("admission_derived" in {part.lower() for part in destination.parts},
             "destination must be inside an admission_derived namespace")
    _require(not destination.exists(), f"destination already exists; refusing overwrite: {destination}")
    audit = audit_true_ppo_source(source_dir)
    source = Path(audit["source_dir"])
    _require(source != destination and source not in destination.parents and destination not in source.parents,
             "source and derived destination must be independent directories")
    payload = _bridge_payload(audit)
    staging = destination.with_name(f".{destination.name}.tmp-{uuid4().hex}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.mkdir()
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        report = {
            "schema": BRIDGE_REPORT_SCHEMA,
            "status": BRIDGE_STATUS,
            "passed": True,
            "formal_repeat_eligible": True,
            "paper_row_eligible": False,
            "source_sha256": payload["source_sha256"],
            "bridge_manifest_sha256": sha256_file(manifest_path),
            "warnings": [payload["paper_row_reason"]],
            "errors": [],
        }
        (staging / "admission_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        os.rename(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return validate_admission_directory(destination)


def validate_admission_directory(path: str | Path) -> dict[str, Any]:
    """Validate a derived bridge and its still-immutable source, without writes."""
    root = Path(path).resolve()
    manifest_path = root / "manifest.json"
    report_path = root / "admission_report.json"
    bridge = _json(manifest_path)
    report = _json(report_path)
    _require(bridge.get("schema") == BRIDGE_SCHEMA and bridge.get("status") == BRIDGE_STATUS,
             "derived admission manifest schema/status mismatch")
    _require(report.get("schema") == BRIDGE_REPORT_SCHEMA and report.get("passed") is True,
             "derived admission report is not valid")
    _require(report.get("bridge_manifest_sha256") == sha256_file(manifest_path),
             "derived admission report does not bind manifest.json")
    audit = audit_true_ppo_source(bridge.get("source_run_dir", ""))
    _require(bridge.get("source_sha256") == audit["source_sha256"], "derived source SHA bindings drifted")
    _require(report.get("source_sha256") == audit["source_sha256"], "admission report source SHA bindings drifted")
    _require(bridge.get("normalized_manifest") == audit["normalized_manifest"], "derived normalized identity drifted")
    return {
        "passed": True,
        "errors": [],
        "warnings": list(report.get("warnings", [])),
        "metrics": audit["metrics"],
        "episodes": audit["episodes"],
        "training_curve_path": audit["training_curve_path"],
        "normalized_manifest": audit["normalized_manifest"],
        "source_dir": audit["source_dir"],
        "source_sha256": audit["source_sha256"],
        "bridge_manifest_path": manifest_path,
        "admission_report_path": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a read-only true-PPO formal admission bridge")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    try:
        result = audit_true_ppo_source(args.source) if args.audit_only else create_admission_bridge(args.source, args.destination)
        print(json.dumps({"status": "PROVISIONAL", "passed": True,
                          "source": str(args.source), "destination": None if args.audit_only else str(args.destination),
                          "identity": result.get("normalized_manifest")}, indent=2, default=str))
    except AdmissionError as exc:
        print(json.dumps({"status": "BLOCKED", "passed": False, "error": str(exc)}, indent=2))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
