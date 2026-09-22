"""Build train prototypes and freeze radius from formal validation replay only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from baselines.priorrl_cc4.build_prototype_artifact import (
    DEFAULT_CACHE, DEFAULT_REGISTRY, DEFAULT_TRAIN, audit_and_collect, path,
)
from baselines.priorrl_cc4.build_supplemental_cache import load_records as load_supplement_records
from baselines.priorrl_cc4.prototype_retrieval import (
    FrozenPrototypeRetriever, _distance_rows, _sha, coverage_report,
    fit_train_prototypes, freeze_validation_radius, validate_payload,
)
from formal_experiments.evaluation.build_final_prior_cache import (
    EXACT_MODEL_ID, GENERATION_CONFIG, MODEL_ALIAS, PROMPT_VERSION, PROVIDER,
    load_train_state_bank,
)
from formal_experiments.ours.prior_cache import make_identity
from shared.formal_state import BLUE_AGENTS

DEFAULT_VALIDATION = "outputs/formal_replay_final_20260917/validation.jsonl"
DEFAULT_ARTIFACT = "outputs/priorrl_cc4/prototypes/frozen_prototypes.json"
DEFAULT_COVERAGE = "outputs/priorrl_cc4/prototypes/frozen_prototype_coverage.json"
DEFAULT_PROVENANCE = "outputs/priorrl_cc4/prototypes/frozen_prototype_provenance.json"
DEFAULT_THRESHOLD_AUDIT = "outputs/priorrl_cc4/prototypes/full_train_coverage_audit.json"
PRE_SUPPLEMENT_RADIUS = 2.560749706662155
PRE_SUPPLEMENT_THRESHOLD_AUDIT_SHA256 = "004ecb1764dc3072a6c3219d2c21f3daa84e72941bdcb1af483d70b4f2097228"
PRE_SUPPLEMENT_BASE_FILE_SHA256 = "11d7b3948e2f15d04d80e16e33ba5e2ab55308156c0de2fe7b059747410d9739"
PRE_SUPPLEMENT_BASE_PROTOTYPE_SHA256 = "05ff321ca9f5f38a4ee2b0a09c634bea4c9fb283585f5bc7290ccc0c6a1a67e0"
EXPECTED_TRAIN_UNIQUE_COUNT = 41079
EXPECTED_TRAIN_RAW_COUNT = 43297
EXPECTED_VALIDATION_UNIQUE_COUNT = 10515
SUPERSEDED_1909_ARTIFACT_SHA256 = "dc9b1142ceb4bd5f76821f14dc4e70300abac2e9795a747d7ddbefcc6c3f8c61"
SUPERSEDED_1909_COVERAGE_SHA256 = "ffd35f1879cedde16d2a8ad4d4c9c2cda493106756f71df856018633f53b9737"


def file_sha(path_: Path) -> str:
    return hashlib.sha256(path_.read_bytes()).hexdigest()


def _canonical_sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _payload_file_sha(payload: dict, *, newline: str = "\n") -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + newline
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_verified_supplement(manifest: str | Path | None, *, train: str | Path):
    if manifest is None:
        return [], None
    manifest_path = path(manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (payload.get("schema") != "formal_prior_cache_train_supplement_v1"
            or payload.get("status") != "PASS" or payload.get("mode") != "live"
            or payload.get("split") != "train"):
        raise ValueError("supplement manifest must be a completed train-only PASS")
    inputs = payload.get("inputs", {})
    records, provenance, verified_inputs = load_supplement_records(
        inputs.get("selection_path", ""), inputs.get("online_probe_path", ""),
        inputs.get("train_replay_path", ""),
    )
    if len(records) != 6 or len(provenance) != 6:
        raise ValueError("supplement manifest must bind exactly 6 records")
    train_path = path(train)
    if verified_inputs["train_replay_sha256"] != file_sha(train_path):
        raise ValueError("supplement manifest does not bind the selected train replay")
    if payload.get("selection_sha256") != _canonical_sha(provenance):
        raise ValueError("supplement manifest selection SHA mismatch")
    if int(payload.get("target_size", -1)) != 6:
        raise ValueError("supplement manifest target size mismatch")
    if (int(payload.get("provider_calls", -1)) != 6
            or int(payload.get("generated", -1)) != 6
            or int(payload.get("verified_entries", -1)) != 6
            or int(payload.get("failed_entries", -1)) != 0):
        raise ValueError("supplement manifest must bind exactly six verified provider calls")
    for field in ("entry_sha256_set_sha256", "logical_manifest_sha256"):
        value = str(payload.get(field, ""))
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"supplement manifest has invalid {field}")
    logical_payload = {
        key: value for key, value in payload.items()
        if key not in {"completed_at_utc", "logical_manifest_sha256"}
    }
    if payload["logical_manifest_sha256"] != _canonical_sha(logical_payload):
        raise ValueError("supplement manifest logical SHA mismatch")
    expected_keys = [make_identity(
        split="train", model_alias=MODEL_ALIAS, provider=PROVIDER,
        exact_model_id=EXACT_MODEL_ID, registry_version=1,
        registry_sha256=str(payload.get("registry_sha256", "")),
        prompt_version=PROMPT_VERSION, agent_name=row.agent_name,
        generation_config=GENERATION_CONFIG, state=row.state,
    ).cache_key for row in records]
    if len(expected_keys) != 6 or list(payload.get("cache_keys", [])) != expected_keys:
        raise ValueError("supplement manifest cache-key binding mismatch")
    return records, {
        "manifest": str(manifest_path), "manifest_file_sha256": file_sha(manifest_path),
        "logical_manifest_sha256": payload.get("logical_manifest_sha256"),
        "entry_sha256_set_sha256": payload.get("entry_sha256_set_sha256"),
        "target_size": len(records), "selection_sha256": payload["selection_sha256"],
        "cache_keys": list(payload.get("cache_keys", [])),
        "provider_calls": int(payload["provider_calls"]),
        "supplement_train_only": True,
        "online_allowed": False,
        "runtime_mode": "offline_read_only",
    }


def _json_bytes(payload: dict, *, compact: bool) -> bytes:
    text = (json.dumps(payload, sort_keys=True, separators=(",", ":"))
            if compact else json.dumps(payload, indent=2, sort_keys=True))
    # Preserve the byte-level Windows serialization used by the reviewed
    # frozen artifacts; atomic staging must not silently rotate trust anchors.
    return (text.replace("\n", "\r\n") + "\r\n").encode("utf-8")


def _stage_json(final_path: Path, payload: dict, *, compact: bool) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", dir=final_path.parent, prefix=f".{final_path.name}.",
        suffix=".tmp", delete=False,
    )
    staged = Path(handle.name)
    try:
        with handle:
            handle.write(_json_bytes(payload, compact=compact))
            handle.flush(); os.fsync(handle.fileno())
        if json.loads(staged.read_text(encoding="utf-8")) != payload:
            raise RuntimeError(f"temporary JSON verification failed: {final_path}")
        return staged
    except Exception:
        staged.unlink(missing_ok=True)
        raise


def _atomic_commit_json_set(items, *, verifier=None) -> None:
    """Stage and cross-check the set, then restore all old files on errors."""
    staged = []
    originals = {}
    committed = []
    try:
        for final_path, payload, compact in items:
            originals[final_path] = final_path.read_bytes() if final_path.is_file() else None
            staged.append((final_path, _stage_json(final_path, payload, compact=compact)))
        staged_map = dict(staged)
        if verifier is not None:
            verifier(staged_map)
        for final_path, staged_path in staged:
            os.replace(staged_path, final_path)
            committed.append(final_path)
    except Exception:
        for final_path in reversed(committed):
            old = originals[final_path]
            if old is None:
                final_path.unlink(missing_ok=True)
                continue
            restore = tempfile.NamedTemporaryFile(
                mode="wb", dir=final_path.parent, prefix=f".{final_path.name}.",
                suffix=".restore", delete=False,
            )
            restore_path = Path(restore.name)
            with restore:
                restore.write(old); restore.flush(); os.fsync(restore.fileno())
            os.replace(restore_path, final_path)
        raise
    finally:
        for _, staged_path in staged:
            staged_path.unlink(missing_ok=True)


def load_pre_supplement_threshold(audit: str | Path, *, validation: str | Path) -> dict:
    """Load the validation-only geometry frozen before train-only supplementation."""
    audit_path = path(audit)
    audit_sha = file_sha(audit_path)
    if audit_sha != PRE_SUPPLEMENT_THRESHOLD_AUDIT_SHA256:
        raise ValueError("pre-supplement threshold audit SHA mismatch")
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    protocol = payload.get("distance_protocol", {})
    validation_info = payload.get("validation_replay", {})
    base_info = payload.get("base_prototypes", {})
    if (payload.get("format_version") != "priorrl_full_train_coverage_audit_v1"
            or protocol.get("radius_selection_split") != "validation"
            or float(protocol.get("radius", -1)) != PRE_SUPPLEMENT_RADIUS
            or base_info.get("file_sha256") != PRE_SUPPLEMENT_BASE_FILE_SHA256
            or base_info.get("internal_prototype_sha256") != PRE_SUPPLEMENT_BASE_PROTOTYPE_SHA256
            or validation_info.get("file_sha256") != file_sha(path(validation))):
        raise ValueError("pre-supplement threshold audit contract mismatch")
    scalers = protocol.get("scaler_by_agent", {})
    if set(scalers) != set(BLUE_AGENTS):
        raise ValueError("pre-supplement threshold audit agent scalers mismatch")
    return {
        "path": str(audit_path), "sha256": audit_sha,
        "radius": PRE_SUPPLEMENT_RADIUS, "weights": protocol["weights"],
        "scaler_by_agent": scalers,
        "threshold_source": "validation_pre_supplement",
        "validation_source_sha256": validation_info["file_sha256"],
        "base_artifact_file_sha256": base_info["file_sha256"],
        "base_prototype_sha256": base_info["internal_prototype_sha256"],
    }


def regenerate_verified_base(*, train, validation, cache, registry, target_size: int):
    """Rebuild the exact original 2,000-prototype base from verified cache entries."""
    prototypes, report = audit_and_collect(
        train=train, cache_root=cache, registry=registry, target_size=target_size,
    )
    if not report["complete"] or len(prototypes) != target_size:
        raise RuntimeError("original base prototype cache is incomplete")
    train_payload = fit_train_prototypes(
        prototypes, source_sha256=report["train_source_sha256"]
    )
    validation_path = path(validation)
    validation_records = load_train_state_bank(validation_path)
    base_frozen = freeze_validation_radius(
        train_payload,
        [(x.agent_name, x.state) for x in validation_records],
        validation_source_sha256=file_sha(validation_path),
        quantile=1.0,
    )
    # The original base was serialized as v1 (A4 marginal only).  Reconstruct
    # that semantic projection from the same verified cache entries and prove
    # exact identity before admitting the v2 K6/H4 format migration.
    v1_projection = json.loads(json.dumps(base_frozen))
    v1_projection.pop("prototype_sha256", None)
    v1_projection["format_version"] = "priorrl_agent_local_prototypes_v1"
    for agent in BLUE_AGENTS:
        for item in v1_projection["agents"][agent]["prototypes"]:
            for field in ("plans", "raw_prior_scores", "prior_preferences", "sources"):
                item.pop(field, None)
    v1_projection["prototype_sha256"] = _sha(v1_projection)
    v1_file_sha = _payload_file_sha(v1_projection, newline="\r\n")
    binding = {
        "target_size": target_size,
        "artifact_file_sha256": _payload_file_sha(base_frozen),
        "prototype_sha256": base_frozen["prototype_sha256"],
        "migration": "v1_to_v2_recomputed",
        "source_v1_recomputed_file_sha256": v1_file_sha,
        "source_v1_recomputed_prototype_sha256": v1_projection["prototype_sha256"],
        "semantic_entry_count": len(prototypes),
        "cache_hits": report["cache_hits"],
        "cache_misses": report["cache_misses"],
        "semantic_missing": 0,
        "semantic_extra": 0,
        "train_source_sha256": report["train_source_sha256"],
        "validation_source_sha256": file_sha(validation_path),
        "radius": base_frozen["radius"],
    }
    if (binding["source_v1_recomputed_file_sha256"] != PRE_SUPPLEMENT_BASE_FILE_SHA256
            or binding["source_v1_recomputed_prototype_sha256"] != PRE_SUPPLEMENT_BASE_PROTOTYPE_SHA256
            or binding["semantic_entry_count"] != 2000
            or binding["cache_hits"] != 2000 or binding["cache_misses"] != 0
            or binding["radius"] != PRE_SUPPLEMENT_RADIUS):
        raise RuntimeError("regenerated original base semantic identity mismatch")
    return base_frozen, binding


def freeze_pre_supplement_validation_threshold(
    train_payload: dict, validation_states, *, threshold: dict, supplement: dict,
    base_frozen: dict, base_binding: dict, supplement_entries: list[dict],
) -> dict:
    """Add train-only centers without refitting the frozen validation geometry."""
    payload = json.loads(json.dumps(train_payload)); payload.pop("prototype_sha256", None)
    payload["weights"] = list(base_frozen["weights"])
    for agent in BLUE_AGENTS:
        payload["agents"][agent]["scaler_mean"] = list(base_frozen["agents"][agent]["scaler_mean"])
        payload["agents"][agent]["scaler_scale"] = list(base_frozen["agents"][agent]["scaler_scale"])
    distances = _distance_rows(payload, validation_states)
    if max(distances) > float(threshold["radius"]):
        raise RuntimeError("supplemented prototypes violate the pre-supplement validation threshold")
    payload.update({
        "radius_selection_split": "validation",
        "radius": float(threshold["radius"]),
        "radius_quantile": 1.0,
        "validation_source_sha256": threshold["validation_source_sha256"],
        "validation_distance_count": len(distances),
        "validation_distance_min": min(distances),
        "validation_distance_max": max(distances),
        "threshold_source": threshold["threshold_source"],
        "threshold_audit_sha256": threshold["sha256"],
        "base_artifact_file_sha256": base_binding["artifact_file_sha256"],
        "base_prototype_sha256": base_binding["prototype_sha256"],
        "base_migration": base_binding["migration"],
        "base_source_v1_recomputed_file_sha256": base_binding["source_v1_recomputed_file_sha256"],
        "base_source_v1_recomputed_prototype_sha256": base_binding["source_v1_recomputed_prototype_sha256"],
        "supplement_train_only": bool(supplement["supplement_train_only"]),
        "provider_calls": int(supplement["provider_calls"]),
        "supplement_manifest_sha256": supplement["manifest_file_sha256"],
        "supplement_logical_manifest_sha256": supplement["logical_manifest_sha256"],
        "supplement_entry_sha256_set_sha256": supplement["entry_sha256_set_sha256"],
        "supplement_entries": supplement_entries,
        "online_allowed": bool(supplement["online_allowed"]),
        "runtime_mode": supplement["runtime_mode"],
    })
    payload["prototype_sha256"] = _sha(payload)
    validate_payload(payload, require_frozen=True)
    return payload


def freeze(*, train=DEFAULT_TRAIN, validation=DEFAULT_VALIDATION, cache=DEFAULT_CACHE,
           registry=DEFAULT_REGISTRY, artifact=DEFAULT_ARTIFACT,
           coverage=DEFAULT_COVERAGE, target_size=2000, radius_quantile=1.0,
           supplement_manifest=None, threshold_audit=DEFAULT_THRESHOLD_AUDIT,
           provenance=DEFAULT_PROVENANCE) -> dict:
    supplement_records, supplement = load_verified_supplement(
        supplement_manifest, train=train
    )
    prototypes, cache_report = audit_and_collect(
        train=train, cache_root=cache, registry=registry, target_size=target_size,
        extra_train_records=supplement_records,
    )
    if not cache_report["complete"]:
        raise RuntimeError("formal train cache is incomplete; freeze fails closed")
    prototype_source_sha256 = _canonical_sha({
        "train_source_sha256": cache_report["train_source_sha256"],
        "supplement_manifest_file_sha256": (
            supplement["manifest_file_sha256"] if supplement else None
        ),
    })
    train_payload = fit_train_prototypes(prototypes, source_sha256=prototype_source_sha256)
    validation_path = path(validation)
    validation_records = load_train_state_bank(validation_path)
    validation_states = [(x.agent_name, x.state) for x in validation_records]
    base_binding = None; supplement_entries = []
    if supplement is None:
        frozen = freeze_validation_radius(
            train_payload, validation_states,
            validation_source_sha256=file_sha(validation_path), quantile=radius_quantile,
        )
        threshold = None
    else:
        if int(target_size) != 2000:
            raise ValueError("formal supplement must extend the exact 2,000-prototype base")
        if float(radius_quantile) != 1.0:
            raise ValueError("supplemented formal artifact requires the frozen q=1.0 threshold")
        threshold = load_pre_supplement_threshold(threshold_audit, validation=validation_path)
        base_frozen, base_binding = regenerate_verified_base(
            train=train, validation=validation, cache=cache, registry=registry,
            target_size=target_size,
        )
        if (base_binding["source_v1_recomputed_file_sha256"] != threshold["base_artifact_file_sha256"]
                or base_binding["source_v1_recomputed_prototype_sha256"] != threshold["base_prototype_sha256"]):
            raise RuntimeError("threshold audit is not bound to the regenerated base artifact")
        supplement_keys = set(supplement["cache_keys"])
        supplement_entries = sorted((
            {"cache_key": item.cache_key, "entry_sha256": item.cache_entry_sha256,
             "agent_name": item.agent_name}
            for item in prototypes if item.cache_key in supplement_keys
        ), key=lambda item: item["cache_key"])
        if (len(supplement_entries) != len(supplement_keys)
                or _canonical_sha(sorted(x["entry_sha256"] for x in supplement_entries))
                != supplement["entry_sha256_set_sha256"]):
            raise RuntimeError("six OFOX supplement entry hashes do not match the live manifest")
        frozen = freeze_pre_supplement_validation_threshold(
            train_payload, validation_states, threshold=threshold, supplement=supplement,
            base_frozen=base_frozen, base_binding=base_binding,
            supplement_entries=supplement_entries,
        )
    artifact_path = path(artifact)
    artifact_file_sha256 = hashlib.sha256(_json_bytes(frozen, compact=True)).hexdigest()
    artifact_probe = _stage_json(artifact_path, frozen, compact=True)
    try:
        retriever = FrozenPrototypeRetriever.load(artifact_probe)
    finally:
        artifact_probe.unlink(missing_ok=True)
    train_records = load_train_state_bank(path(train))
    report = coverage_report(retriever, {
        "train_prototypes": [(x.agent_name, x.state) for x in prototypes],
        "full_train_unique": [(x.agent_name, x.state) for x in train_records],
        "validation": validation_states,
    })
    full_train = report["splits"]["full_train_unique"]
    if supplement and (full_train["misses"] != 0 or full_train["coverage"] != 1.0):
        raise RuntimeError("formal prototype artifact does not cover every train state")
    with path(train).open("r", encoding="utf-8") as handle:
        full_train["raw_row_count"] = sum(1 for line in handle if line.strip())
    if supplement and (full_train["total"] != EXPECTED_TRAIN_UNIQUE_COUNT
            or full_train["raw_row_count"] != EXPECTED_TRAIN_RAW_COUNT
            or report["splits"]["validation"]["total"] != EXPECTED_VALIDATION_UNIQUE_COUNT
            or report["splits"]["validation"]["misses"] != 0):
        raise RuntimeError("formal prototype full-coverage count contract mismatch")
    report.update({
        "artifact": str(artifact_path), "artifact_file_sha256": artifact_file_sha256,
        "registry_sha256": cache_report["registry_sha256"],
        "train_source_sha256": cache_report["train_source_sha256"],
        "prototype_source_sha256": prototype_source_sha256,
        "supplement": supplement,
        "validation_source_sha256": file_sha(validation_path),
        "radius_selection_split": "validation", "test_mode": "read_only_no_radius_updates",
        "threshold_source": "validation_pre_supplement" if threshold else "validation",
        "threshold_audit_sha256": threshold["sha256"] if threshold else None,
        "supplement_train_only": bool(supplement),
        "provider_calls": int(supplement["provider_calls"]) if supplement else 0,
        "provider_calls_during_freeze": 0,
        "base_artifact": base_binding,
        "ofox_entries": supplement_entries,
        "entry_sha256_set_sha256": (
            supplement["entry_sha256_set_sha256"] if supplement else None
        ),
        "online_allowed": False,
        "runtime_mode": "offline_read_only",
        "provenance_manifest": str(path(provenance)) if supplement else None,
    })
    coverage_path = path(coverage)
    commit_items = [(artifact_path, frozen, True), (coverage_path, report, False)]
    provenance_path = None
    if supplement:
        coverage_file_sha256 = hashlib.sha256(_json_bytes(report, compact=False)).hexdigest()
        provenance_payload = {
            "schema": "priorrl_frozen_prototype_provenance_v1",
            "status": "PASS",
            "artifact": {"path": str(artifact_path), "file_sha256": artifact_file_sha256,
                         "prototype_sha256": frozen["prototype_sha256"]},
            "coverage": {"path": str(coverage_path), "file_sha256": coverage_file_sha256,
                         "full_train_unique_hits": full_train["hits"],
                         "full_train_unique_misses": full_train["misses"],
                         "full_train_raw_count": full_train["raw_row_count"],
                         "validation_hits": report["splits"]["validation"]["hits"],
                         "validation_misses": report["splits"]["validation"]["misses"]},
            "base_artifact": base_binding,
            "sources": {"train_sha256": cache_report["train_source_sha256"],
                        "validation_sha256": file_sha(validation_path),
                        "threshold_audit_sha256": threshold["sha256"],
                        "supplement_manifest_sha256": supplement["manifest_file_sha256"],
                        "supplement_logical_manifest_sha256": supplement["logical_manifest_sha256"]},
            "threshold_source": "validation_pre_supplement",
            "radius": PRE_SUPPLEMENT_RADIUS,
            "supplement_train_only": True,
            "provider_calls": supplement["provider_calls"],
            "online_allowed": False,
            "runtime_mode": "offline_read_only",
            "entry_sha256_set_sha256": supplement["entry_sha256_set_sha256"],
            "ofox_entries": supplement_entries,
            "superseded_artifacts": [
                {"kind": "post_supplement_refit_artifact", "sha256": SUPERSEDED_1909_ARTIFACT_SHA256,
                 "status": "SUPERSEDED_FAIL_CLOSED", "reason": "radius was incorrectly refit to 1.909"},
                {"kind": "post_supplement_refit_coverage", "sha256": SUPERSEDED_1909_COVERAGE_SHA256,
                 "status": "SUPERSEDED_FAIL_CLOSED", "reason": "full train coverage was not demonstrated"},
                {"kind": "pre_supplement_coverage_audit", "sha256": threshold["sha256"],
                 "status": "SUPERSEDED_FAIL_CLOSED", "reason": "audit predates the six verified OFOX entries"},
            ],
        }
        provenance_path = path(provenance)
        commit_items.append((provenance_path, provenance_payload, False))

    def verify_staged(staged: dict[Path, Path]) -> None:
        FrozenPrototypeRetriever.load(staged[artifact_path])
        staged_artifact_sha = file_sha(staged[artifact_path])
        staged_coverage_sha = file_sha(staged[coverage_path])
        staged_coverage = json.loads(staged[coverage_path].read_text(encoding="utf-8"))
        if staged_coverage.get("artifact_file_sha256") != staged_artifact_sha:
            raise RuntimeError("staged coverage does not bind the staged artifact")
        if provenance_path is not None:
            staged_provenance = json.loads(
                staged[provenance_path].read_text(encoding="utf-8")
            )
            if (staged_provenance.get("artifact", {}).get("file_sha256") != staged_artifact_sha
                    or staged_provenance.get("coverage", {}).get("file_sha256")
                    != staged_coverage_sha):
                raise RuntimeError("staged provenance does not bind artifact/coverage")

    _atomic_commit_json_set(commit_items, verifier=verify_staged)
    if provenance_path is not None:
        report["provenance_manifest_sha256"] = file_sha(provenance_path)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default=DEFAULT_TRAIN); p.add_argument("--validation", default=DEFAULT_VALIDATION)
    p.add_argument("--cache", default=DEFAULT_CACHE); p.add_argument("--registry", default=DEFAULT_REGISTRY)
    p.add_argument("--artifact", default=DEFAULT_ARTIFACT); p.add_argument("--coverage", default=DEFAULT_COVERAGE)
    p.add_argument("--target-size", type=int, default=2000); p.add_argument("--radius-quantile", type=float, default=1.0)
    p.add_argument("--supplement-manifest", default=None)
    p.add_argument("--threshold-audit", default=DEFAULT_THRESHOLD_AUDIT)
    p.add_argument("--provenance", default=DEFAULT_PROVENANCE)
    args = p.parse_args(argv)
    report = freeze(**vars(args))
    print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
