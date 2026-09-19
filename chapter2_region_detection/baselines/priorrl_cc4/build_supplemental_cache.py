"""Build a small audited train-only supplement for the frozen prior cache.

The default is read-only. ``--execute-live`` is required for paid provider
calls. Inputs may contain only exact train-replay states plus a fail-closed
on-policy state produced by a train seed; validation/test states are rejected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from formal_experiments.evaluation.build_final_prior_cache import (
    DEFAULT_OUTPUT, DEFAULT_PRIMARY_SELECTION, DEFAULT_REGISTRY, EXACT_MODEL_ID,
    GENERATION_CONFIG, MODEL_ALIAS, PROMPT_VERSION, PROVIDER, StateBankRecord,
    _live_generator, build_cache, load_frozen_registry, load_primary_model_selection,
    resolve_path, sha256_file, validate_probe_generation,
)
from formal_experiments.ours.prior_cache import (
    PriorCache, exact_state_float32, exact_state_sha256, make_identity,
)
from shared.formal_state import BLUE_AGENTS

DEFAULT_SELECTION = "outputs/priorrl_cc4/prototypes/supplement_train_selection.json"
DEFAULT_ONLINE_PROBE = "outputs/priorrl_cc4/prototypes/online_coverage_probe.json"
DEFAULT_MANIFEST = "outputs/lwm_rl_final_20260917/prior_cache_final/supplement_manifest.json"


def _canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_records(selection: str | Path, online_probe: str | Path,
                 train_replay: str | Path) -> tuple[list[StateBankRecord], list[dict], dict]:
    selection_path = resolve_path(selection); probe_path = resolve_path(online_probe)
    replay_path = resolve_path(train_replay)
    selected = json.loads(selection_path.read_text(encoding="utf-8"))
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if selected.get("selection_count") != 5 or len(selected.get("selections", [])) != 5:
        raise ValueError("train supplement must contain the audited five-state selection")
    if selected.get("train_replay", {}).get("file_sha256") != sha256_file(replay_path):
        raise ValueError("supplement selection train replay SHA mismatch")
    replay_lines = replay_path.read_text(encoding="utf-8").splitlines()
    records, provenance = [], []
    for item in selected["selections"]:
        index = int(item["source_index"])
        row = json.loads(replay_lines[index])
        state = exact_state_float32(item["state"])
        digest = exact_state_sha256(state)
        if (digest != item["state_sha256"] or row.get("agent_name") != item["agent_name"]
                or int(row.get("episode_seed")) != int(item["episode_seed"])
                or exact_state_sha256(row["state"]) != digest):
            raise ValueError("audited train selection no longer matches source replay")
        action_id = int(row.get("requested_action_id", row.get("executed_index", -1)))
        records.append(StateBankRecord(index, int(item["episode_seed"]), item["agent_name"],
                                       action_id, int(float(state[17])), state, digest))
        provenance.append({"origin": "train_replay_coverage", "agent_name": item["agent_name"],
                           "episode_seed": int(item["episode_seed"]), "source_index": index,
                           "state_sha256": digest})
    if probe.get("schema") != "priorrl_online_prototype_coverage_probe_v1" or probe.get("status") != "BLOCKED":
        raise ValueError("online supplement input must be a fail-closed coverage probe")
    episode_seed = int(probe["episode_seed"]); agent = str(probe["agent_name"])
    if episode_seed not in range(1000, 1032) or agent not in BLUE_AGENTS:
        raise ValueError("online supplement state must come from a train seed and known Blue agent")
    state = exact_state_float32(probe["state"]); digest = exact_state_sha256(state)
    if digest != probe.get("state_sha256"):
        raise ValueError("online coverage state SHA mismatch")
    records.append(StateBankRecord(-1, episode_seed, agent, -1, int(float(state[17])), state, digest))
    provenance.append({"origin": "fail_closed_on_policy_train_probe", "agent_name": agent,
                       "episode_seed": episode_seed, "policy_seed": int(probe["policy_seed"]),
                       "state_sha256": digest, "probe_report_sha256": probe["report_sha256"]})
    keys = [(x.agent_name, x.state_sha256) for x in records]
    if len(keys) != len(set(keys)):
        raise ValueError("supplement contains duplicate agent-local states")
    inputs = {"selection_path": str(selection_path), "selection_file_sha256": sha256_file(selection_path),
              "online_probe_path": str(probe_path), "online_probe_file_sha256": sha256_file(probe_path),
              "train_replay_path": str(replay_path), "train_replay_sha256": sha256_file(replay_path)}
    return records, provenance, inputs


def run(*, selection=DEFAULT_SELECTION, online_probe=DEFAULT_ONLINE_PROBE,
        train_replay="outputs/formal_replay_final_20260917/train.jsonl",
        output=DEFAULT_OUTPUT, registry=DEFAULT_REGISTRY,
        primary_selection=DEFAULT_PRIMARY_SELECTION, manifest=DEFAULT_MANIFEST,
        execute_live=False, workers=6) -> dict:
    _, registry_sha = load_frozen_registry(registry)
    _, primary_sha = load_primary_model_selection(primary_selection, registry_sha256=registry_sha)
    records, provenance, inputs = load_records(selection, online_probe, train_replay)
    cache = PriorCache(resolve_path(output))
    identities = [make_identity(
        split="train", model_alias=MODEL_ALIAS, provider=PROVIDER,
        exact_model_id=EXACT_MODEL_ID, registry_version=1, registry_sha256=registry_sha,
        prompt_version=PROMPT_VERSION, agent_name=row.agent_name,
        generation_config=GENERATION_CONFIG, state=row.state,
    ) for row in records]
    hits = sum(cache.load(identity) is not None for identity in identities)
    base = {
        "schema": "formal_prior_cache_train_supplement_v1",
        "status": "DRY_RUN_BLOCKED_MISSING_ENTRIES" if hits < len(records) else "DRY_RUN_PASS",
        "mode": "live" if execute_live else "dry_run",
        "split": "train", "target_size": len(records), "verified_existing": hits,
        "provider_calls": 0, "registry_sha256": registry_sha,
        "primary_model_selection_sha256": primary_sha, "inputs": inputs,
        "selection_sha256": _canonical_sha(provenance), "provenance": provenance,
        "cache_keys": [identity.cache_key for identity in identities],
    }
    if not execute_live:
        return base
    result = build_cache(
        selected=records, output_root=output, registry_sha256=registry_sha,
        generator_factory=_live_generator, workers=int(workers), max_attempts=3,
        retry_backoff_seconds=2.0,
        progress_path=resolve_path(output) / "supplement_progress.jsonl",
        result_validator=validate_probe_generation,
    )
    if result["verified_entries"] != len(records) or result["failed_entries"]:
        raise RuntimeError("supplement cache build did not verify every selected state")
    for identity in identities:
        if cache.load(identity) is None:
            raise RuntimeError("supplement entry failed independent cache reload")
    report = {**base, **result, "status": "PASS", "provider_calls": result["generated"],
              "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    report["logical_manifest_sha256"] = _canonical_sha({
        k: v for k, v in report.items() if k != "completed_at_utc"
    })
    manifest_path = resolve_path(manifest); manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", default=DEFAULT_SELECTION)
    parser.add_argument("--online-probe", default=DEFAULT_ONLINE_PROBE)
    parser.add_argument("--train-replay", default="outputs/formal_replay_final_20260917/train.jsonl")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--primary-selection", default=DEFAULT_PRIMARY_SELECTION)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--execute-live", action="store_true")
    args = parser.parse_args(argv)
    report = run(**vars(args)); print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"PASS", "DRY_RUN_PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
