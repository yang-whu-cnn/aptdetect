"""Single fail-closed gate for the frozen offline PriorRL/OFOX artifact chain."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


EXPECTED_TRAIN_RAW = 43297
EXPECTED_TRAIN_UNIQUE = 41079
EXPECTED_VALIDATION_UNIQUE = 10515
EXPECTED_RADIUS = 2.560749706662155
FROZEN_PROTOTYPE_PROVENANCE_SHA256 = (
    "464b306274a4123acf39384d289eca632e02960f3dfb9a381e9cf7613d471363"
)
OFOX_ENTRY_SET_SHA256 = "7941f6f11d47989265955bdc6caff950de97a9f5b9562afa8f7cb5e0209a7e10"
OFOX_SUPPLEMENT_MANIFEST_SHA256 = (
    "ece590f5bc01ea6b6cf06458cd86096914e85d886d50686b2c7c9d365e909d44"
)
OFOX_LOGICAL_MANIFEST_SHA256 = (
    "5ddc621817a902d3c8f70d76353f80ba2ab0c37e1947d8e855a5f4518443e49f"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _internal_prototype_sha(payload: dict[str, Any]) -> str:
    material = dict(payload)
    declared = material.pop("prototype_sha256", None)
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    actual = hashlib.sha256(encoded).hexdigest()
    if declared != actual:
        raise RuntimeError("BLOCKED: frozen prototype internal SHA256 mismatch")
    return actual


def validate_frozen_prior_provenance(
    artifact_path: Path,
    *,
    expected_provenance_sha256: str = FROZEN_PROTOTYPE_PROVENANCE_SHA256,
    expected_entry_set_sha256: str = OFOX_ENTRY_SET_SHA256,
    expected_supplement_manifest_sha256: str = OFOX_SUPPLEMENT_MANIFEST_SHA256,
    expected_logical_manifest_sha256: str = OFOX_LOGICAL_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Validate artifact, coverage, provenance and the approved 6/6/0 live manifest."""
    artifact_path = Path(artifact_path)
    coverage_path = artifact_path.parent / "frozen_prototype_coverage.json"
    provenance_path = artifact_path.parent / "frozen_prototype_provenance.json"
    missing = [str(path) for path in (artifact_path, coverage_path, provenance_path)
               if not path.is_file()]
    if missing:
        raise RuntimeError(f"BLOCKED: missing frozen prior provenance artifacts: {missing}")
    if sha256_file(provenance_path) != expected_provenance_sha256:
        raise RuntimeError("BLOCKED: frozen prototype provenance SHA256 mismatch")

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    coverage_payload = json.loads(coverage_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not all(isinstance(value, dict) for value in (artifact, coverage_payload, provenance)):
        raise RuntimeError("BLOCKED: frozen prior provenance artifacts must be JSON objects")
    artifact_sha = sha256_file(artifact_path)
    coverage_sha = sha256_file(coverage_path)
    prototype_sha = _internal_prototype_sha(artifact)
    train = coverage_payload.get("splits", {}).get("full_train_unique", {})
    validation = coverage_payload.get("splits", {}).get("validation", {})
    supplement = coverage_payload.get("supplement", {})
    entries = provenance.get("ofox_entries")
    entry_keys = [item.get("cache_key") for item in entries] if isinstance(entries, list) else []
    entry_hashes = [item.get("entry_sha256") for item in entries] if isinstance(entries, list) else []
    base = provenance.get("base_artifact", {})
    pcoverage = provenance.get("coverage", {})
    sources = provenance.get("sources", {})
    checks = {
        "schema/status": (provenance.get("schema") == "priorrl_frozen_prototype_provenance_v1"
                          and provenance.get("status") == "PASS"),
        "artifact identity": (coverage_payload.get("artifact_file_sha256") == artifact_sha
                              and coverage_payload.get("prototype_sha256") == prototype_sha
                              and provenance.get("artifact", {}).get("file_sha256") == artifact_sha
                              and provenance.get("artifact", {}).get("prototype_sha256") == prototype_sha),
        "coverage identity": pcoverage.get("file_sha256") == coverage_sha,
        "coverage counts": (
            train.get("raw_row_count"), train.get("total"), train.get("hits"),
            train.get("misses"), validation.get("total"), validation.get("hits"),
            validation.get("misses"),
        ) == (EXPECTED_TRAIN_RAW, EXPECTED_TRAIN_UNIQUE, EXPECTED_TRAIN_UNIQUE,
              0, EXPECTED_VALIDATION_UNIQUE, EXPECTED_VALIDATION_UNIQUE, 0),
        "coverage ratios": train.get("coverage") == 1.0 and validation.get("coverage") == 1.0,
        "provenance counts": (
            pcoverage.get("full_train_raw_count"), pcoverage.get("full_train_unique_hits"),
            pcoverage.get("full_train_unique_misses"), pcoverage.get("validation_hits"),
            pcoverage.get("validation_misses"),
        ) == (EXPECTED_TRAIN_RAW, EXPECTED_TRAIN_UNIQUE, 0, EXPECTED_VALIDATION_UNIQUE, 0),
        "base semantic coverage": (base.get("migration") == "v1_to_v2_recomputed"
                                   and base.get("semantic_entry_count") == 2000
                                   and base.get("semantic_missing") == 0
                                   and base.get("semantic_extra") == 0),
        "radius/threshold": (artifact.get("radius") == EXPECTED_RADIUS
                             and artifact.get("threshold_source") == "validation_pre_supplement"
                             and coverage_payload.get("radius") == EXPECTED_RADIUS
                             and coverage_payload.get("radius_selection_split") == "validation"
                             and coverage_payload.get("threshold_source") == "validation_pre_supplement"
                             and provenance.get("radius") == EXPECTED_RADIUS
                             and provenance.get("threshold_source") == "validation_pre_supplement"),
        "exactly six": (provenance.get("provider_calls") == 6
                        and coverage_payload.get("provider_calls") == 6
                        and coverage_payload.get("provider_calls_during_freeze") == 0
                        and supplement.get("target_size") == 6
                        and supplement.get("provider_calls") == 6
                        and len(entry_keys) == len(set(entry_keys)) == 6
                        and len(entry_hashes) == len(set(entry_hashes)) == 6
                        and all(SHA256_RE.fullmatch(str(value))
                                for value in entry_keys + entry_hashes)),
        "entry set SHA256": (provenance.get("entry_sha256_set_sha256")
                             == coverage_payload.get("entry_sha256_set_sha256")
                             == supplement.get("entry_sha256_set_sha256")
                             == expected_entry_set_sha256),
        "offline runtime": (provenance.get("online_allowed") is False
                            and provenance.get("runtime_mode") == "offline_read_only"
                            and provenance.get("supplement_train_only") is True
                            and coverage_payload.get("online_allowed") is False
                            and coverage_payload.get("runtime_mode") == "offline_read_only"
                            and coverage_payload.get("supplement_train_only") is True
                            and coverage_payload.get("test_mode") == "read_only_no_radius_updates"
                            and supplement.get("online_allowed") is False
                            and supplement.get("runtime_mode") == "offline_read_only"
                            and supplement.get("supplement_train_only") is True),
        "supplement sources": (
            sources.get("supplement_manifest_sha256")
            == supplement.get("manifest_file_sha256")
            == expected_supplement_manifest_sha256
            and sources.get("supplement_logical_manifest_sha256")
            == supplement.get("logical_manifest_sha256")
            == expected_logical_manifest_sha256),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("BLOCKED: frozen OFOX provenance contract FAIL: " + ", ".join(failed))

    local_candidates = (
        artifact_path.parent / "supplement_manifest.json",
        artifact_path.parents[2] / "lwm_rl_final_20260917" / "prior_cache_final" /
        "supplement_manifest.json",
    )
    supplement_manifest = next(
        (candidate for candidate in local_candidates if candidate.is_file()),
        Path(str(supplement.get("manifest", ""))),
    )
    if not supplement_manifest.is_file() or sha256_file(supplement_manifest) != expected_supplement_manifest_sha256:
        raise RuntimeError("BLOCKED: approved OFOX supplement manifest file binding mismatch")
    live = json.loads(supplement_manifest.read_text(encoding="utf-8"))
    logical_payload = {
        key: value for key, value in live.items()
        if key not in {"completed_at_utc", "logical_manifest_sha256"}
    }
    if (live.get("schema") != "formal_prior_cache_train_supplement_v1"
            or live.get("status") != "PASS" or live.get("mode") != "live"
            or live.get("split") != "train" or live.get("provider_calls") != 6
            or live.get("generated") != 6 or live.get("verified_entries") != 6
            or live.get("failed_entries") != 0
            or live.get("entry_sha256_set_sha256") != expected_entry_set_sha256
            or live.get("logical_manifest_sha256") != expected_logical_manifest_sha256
            or canonical_sha256(logical_payload) != expected_logical_manifest_sha256):
        raise RuntimeError("BLOCKED: approved OFOX supplement is not exactly 6/6/0 train-only")

    return {
        "provider": "ofox", "provider_calls": 6, "generated": 6,
        "verified_entries": 6, "failed": 0, "d27_exact_count": 6,
        "split": "train_only", "coverage_pass": True,
        "online_calls_allowed": False,
        "entry_set_sha256": expected_entry_set_sha256,
        "logical_manifest_sha256": expected_logical_manifest_sha256,
        "prototype_sha256": prototype_sha,
        "prototype_file_sha256": artifact_sha,
        "prototype_coverage_sha256": coverage_sha,
        "prototype_provenance_sha256": sha256_file(provenance_path),
        "supplement_manifest_sha256": expected_supplement_manifest_sha256,
    }
