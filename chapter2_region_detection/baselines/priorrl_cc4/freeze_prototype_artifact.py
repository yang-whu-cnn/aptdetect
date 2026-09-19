"""Build train prototypes and freeze radius from formal validation replay only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from baselines.priorrl_cc4.build_prototype_artifact import (
    DEFAULT_CACHE, DEFAULT_REGISTRY, DEFAULT_TRAIN, audit_and_collect, path,
)
from baselines.priorrl_cc4.build_supplemental_cache import load_records as load_supplement_records
from baselines.priorrl_cc4.prototype_retrieval import (
    FrozenPrototypeRetriever, coverage_report, fit_train_prototypes,
    freeze_validation_radius,
)
from formal_experiments.evaluation.build_final_prior_cache import (
    EXACT_MODEL_ID, GENERATION_CONFIG, MODEL_ALIAS, PROMPT_VERSION, PROVIDER,
    load_train_state_bank,
)
from formal_experiments.ours.prior_cache import make_identity

DEFAULT_VALIDATION = "outputs/formal_replay_final_20260917/validation.jsonl"
DEFAULT_ARTIFACT = "outputs/priorrl_cc4/prototypes/frozen_prototypes.json"
DEFAULT_COVERAGE = "outputs/priorrl_cc4/prototypes/frozen_prototype_coverage.json"


def file_sha(path_: Path) -> str:
    return hashlib.sha256(path_.read_bytes()).hexdigest()


def _canonical_sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_verified_supplement(manifest: str | Path | None, *, train: str | Path):
    if manifest is None:
        return [], None
    manifest_path = path(manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (payload.get("schema") != "formal_prior_cache_train_supplement_v1"
            or payload.get("status") != "PASS" or payload.get("split") != "train"):
        raise ValueError("supplement manifest must be a completed train-only PASS")
    inputs = payload.get("inputs", {})
    records, provenance, verified_inputs = load_supplement_records(
        inputs.get("selection_path", ""), inputs.get("online_probe_path", ""),
        inputs.get("train_replay_path", ""),
    )
    train_path = path(train)
    if verified_inputs["train_replay_sha256"] != file_sha(train_path):
        raise ValueError("supplement manifest does not bind the selected train replay")
    if payload.get("selection_sha256") != _canonical_sha(provenance):
        raise ValueError("supplement manifest selection SHA mismatch")
    if int(payload.get("target_size", -1)) != len(records):
        raise ValueError("supplement manifest target size mismatch")
    expected_keys = [make_identity(
        split="train", model_alias=MODEL_ALIAS, provider=PROVIDER,
        exact_model_id=EXACT_MODEL_ID, registry_version=1,
        registry_sha256=str(payload.get("registry_sha256", "")),
        prompt_version=PROMPT_VERSION, agent_name=row.agent_name,
        generation_config=GENERATION_CONFIG, state=row.state,
    ).cache_key for row in records]
    if list(payload.get("cache_keys", [])) != expected_keys:
        raise ValueError("supplement manifest cache-key binding mismatch")
    return records, {
        "manifest": str(manifest_path), "manifest_file_sha256": file_sha(manifest_path),
        "logical_manifest_sha256": payload.get("logical_manifest_sha256"),
        "target_size": len(records), "selection_sha256": payload["selection_sha256"],
        "cache_keys": list(payload.get("cache_keys", [])),
    }


def freeze(*, train=DEFAULT_TRAIN, validation=DEFAULT_VALIDATION, cache=DEFAULT_CACHE,
           registry=DEFAULT_REGISTRY, artifact=DEFAULT_ARTIFACT,
           coverage=DEFAULT_COVERAGE, target_size=2000, radius_quantile=1.0,
           supplement_manifest=None) -> dict:
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
    frozen = freeze_validation_radius(
        train_payload, validation_states,
        validation_source_sha256=file_sha(validation_path), quantile=radius_quantile,
    )
    artifact_path = path(artifact); artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(frozen, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    retriever = FrozenPrototypeRetriever.load(artifact_path)
    report = coverage_report(retriever, {
        "train_prototypes": [(x.agent_name, x.state) for x in prototypes],
        "validation": validation_states,
    })
    report.update({
        "artifact": str(artifact_path), "artifact_file_sha256": file_sha(artifact_path),
        "registry_sha256": cache_report["registry_sha256"],
        "train_source_sha256": cache_report["train_source_sha256"],
        "prototype_source_sha256": prototype_source_sha256,
        "supplement": supplement,
        "validation_source_sha256": file_sha(validation_path),
        "radius_selection_split": "validation", "test_mode": "read_only_no_radius_updates",
        "provider_calls": 0,
    })
    coverage_path = path(coverage); coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default=DEFAULT_TRAIN); p.add_argument("--validation", default=DEFAULT_VALIDATION)
    p.add_argument("--cache", default=DEFAULT_CACHE); p.add_argument("--registry", default=DEFAULT_REGISTRY)
    p.add_argument("--artifact", default=DEFAULT_ARTIFACT); p.add_argument("--coverage", default=DEFAULT_COVERAGE)
    p.add_argument("--target-size", type=int, default=2000); p.add_argument("--radius-quantile", type=float, default=1.0)
    p.add_argument("--supplement-manifest", default=None)
    args = p.parse_args(argv)
    report = freeze(**vars(args))
    print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
