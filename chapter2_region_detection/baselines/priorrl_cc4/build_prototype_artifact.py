"""Offline-only builder/auditor for the frozen PriorRL prototype artifact."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from baselines.priorrl_cc4.policy import action_prior_from_plans
from baselines.priorrl_cc4.prototype_retrieval import PrototypeInput, fit_train_prototypes
from formal_experiments.evaluation.build_final_prior_cache import (
    EXACT_MODEL_ID, GENERATION_CONFIG, MODEL_ALIAS, PROMPT_VERSION, PROVIDER,
    load_frozen_registry, load_train_state_bank, select_representative_states,
)
from formal_experiments.ours.prior_cache import PriorCache, PriorCacheIdentity, make_identity

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = "outputs/formal_replay_final_20260917/train.jsonl"
DEFAULT_CACHE = "outputs/lwm_rl_final_20260917/prior_cache_final"
DEFAULT_REGISTRY = "configs/llm_model_registry_v1.yaml"
DEFAULT_OUT = "outputs/priorrl_cc4/prototypes/train_prototypes_unfrozen_radius.json"
DEFAULT_REPORT = "outputs/priorrl_cc4/prototypes/train_prototype_coverage.json"


def path(value):
    value = Path(value); return value if value.is_absolute() else ROOT / value


def file_sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit_and_collect(*, train, cache_root, registry, target_size=2000,
                      extra_train_records=()):
    train_path = path(train)
    records = select_representative_states(load_train_state_bank(train_path), target_size)
    base_keys = {(row.agent_name, row.state_sha256) for row in records}
    extras = list(extra_train_records)
    extra_keys = [(row.agent_name, row.state_sha256) for row in extras]
    if len(extra_keys) != len(set(extra_keys)):
        raise ValueError("supplement contains duplicate agent-local states")
    overlap = base_keys.intersection(extra_keys)
    if overlap:
        raise ValueError("supplement must contain only states absent from the base prototype selection")
    records.extend(extras)
    _, registry_sha = load_frozen_registry(path(registry))
    cache = PriorCache(path(cache_root)); prototypes = []; misses = []; per_agent = Counter()
    for row in records:
        identity = make_identity(
            split="train", model_alias=MODEL_ALIAS, provider=PROVIDER,
            exact_model_id=EXACT_MODEL_ID, registry_version=1, registry_sha256=registry_sha,
            prompt_version=PROMPT_VERSION, agent_name=row.agent_name,
            generation_config=GENERATION_CONFIG, state=row.state,
        )
        cached = cache.load(identity)
        if cached is None:
            misses.append({"agent_name": row.agent_name, "state_sha256": row.state_sha256,
                           "cache_key": identity.cache_key})
            continue
        raw = json.loads(cached.path.read_text(encoding="utf-8"))
        action_prior = action_prior_from_plans(cached.prior.plans, cached.prior.prior_preferences)
        prototypes.append(PrototypeInput(
            row.state, row.agent_name, action_prior.numpy(), identity.cache_key,
            str(raw["entry_sha256"]), cached.prior.plans,
            cached.prior.raw_prior_scores, cached.prior.prior_preferences,
            cached.prior.sources,
        )); per_agent[row.agent_name] += 1
    stale_hashes = Counter(); current_entries = 0; current_verified = 0; current_invalid = 0
    for cache_file in path(cache_root).rglob("*.json"):
        if cache_file.name == "manifest.json":
            continue
        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            identity_raw = raw.get("identity", {})
            old_hash = identity_raw.get("registry_sha256")
        except Exception:
            continue
        if old_hash and old_hash != registry_sha:
            stale_hashes[str(old_hash)] += 1
        elif old_hash == registry_sha:
            current_entries += 1
            try:
                identity = PriorCacheIdentity(**{
                    key: identity_raw[key] for key in (
                        "split", "model_alias", "provider", "exact_model_id",
                        "registry_version", "registry_sha256", "prompt_version",
                        "agent_name", "n_actions", "k_candidates", "horizon",
                        "generation_config", "state_dtype", "state_shape", "state_sha256",
                    )
                })
                cache._decode_payload(cache_file, identity, raw)
                current_verified += 1
            except Exception:
                current_invalid += 1
    report = {
        "schema": "priorrl_train_prototype_coverage_v1", "provider_calls": 0,
        "fit_split": "train", "target_size": len(records),
        "base_target_size": target_size, "supplement_target_size": len(extras),
        "train_source": str(train_path), "train_source_sha256": file_sha(train_path),
        "registry_sha256": registry_sha, "cache_root": str(path(cache_root)),
        "cache_hits": len(prototypes), "cache_misses": len(misses),
        "coverage": len(prototypes) / len(records), "per_agent_hits": dict(sorted(per_agent.items())),
        "complete": len(misses) == 0, "missing_examples": misses[:20],
        "stale_registry_entries": dict(sorted(stale_hashes.items())),
        "stale_registry_entries_ignored": True,
        "current_registry_entries": current_entries,
        "current_registry_verified_entries": current_verified,
        "current_registry_invalid_entries": current_invalid,
    }
    return prototypes, report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default=DEFAULT_TRAIN); p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--registry", default=DEFAULT_REGISTRY); p.add_argument("--target-size", type=int, default=2000)
    p.add_argument("--out", default=DEFAULT_OUT); p.add_argument("--report", default=DEFAULT_REPORT)
    p.add_argument("--audit-only", action="store_true")
    args = p.parse_args(argv)
    prototypes, report = audit_and_collect(train=args.train, cache_root=args.cache,
                                            registry=args.registry, target_size=args.target_size)
    report_path = path(args.report); report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.audit_only:
        print(json.dumps(report, indent=2, sort_keys=True)); return 0
    if not report["complete"]:
        raise RuntimeError("train prior cache incomplete; prototype build fails closed")
    payload = fit_train_prototypes(prototypes, source_sha256=report["train_source_sha256"])
    out = path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "out": str(out), "prototype_sha256": payload["prototype_sha256"]}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
