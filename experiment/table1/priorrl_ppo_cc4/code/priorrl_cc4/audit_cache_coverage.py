"""Read-only audit of exact-state PriorRL cache coverage. Never calls a provider."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from formal_experiments.evaluation.build_final_prior_cache import (
    GENERATION_CONFIG, MODEL_ALIAS, PROVIDER, EXACT_MODEL_ID, PROMPT_VERSION,
    load_frozen_registry, load_train_state_bank, select_representative_states,
)
from formal_experiments.ours.prior_cache import (
    PriorCache, exact_state_float32, exact_state_sha256, make_identity,
)


ROOT = Path(__file__).resolve().parents[2]


def _path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def _key(agent: str, state) -> tuple[str, str]:
    return str(agent), exact_state_sha256(exact_state_float32(state))


def _rate(covered: int, total: int) -> dict:
    return {"covered": covered, "missed": total - covered, "total": total,
            "coverage_rate": covered / total if total else None,
            "miss_rate": (total - covered) / total if total else None}


def _audit_replay(path: Path, selected: set[tuple[str, str]]) -> dict:
    rows = []; unique = set(); strata = defaultdict(lambda: [0, 0])
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip(): continue
            item = json.loads(line); state = item["state"]; agent = item["agent_name"]
            key = _key(agent, state); hit = key in selected
            action = int(item.get("requested_action_id", -1)); evidence = int(float(state[17]))
            rows.append(hit); unique.add(key)
            cell = strata[(agent, action, evidence)]; cell[1] += 1; cell[0] += int(hit)
    return {
        "row_level": _rate(sum(rows), len(rows)),
        "unique_agent_state_level": _rate(len(unique & selected), len(unique)),
        "per_agent_action_evidence_rows": {
            f"{a}:{b}:{c}": _rate(v[0], v[1]) for (a, b, c), v in sorted(strata.items())
        },
    }


def _audit_validation_bank(path: Path, selected: set[tuple[str, str]]) -> dict:
    cells = defaultdict(lambda: [0, 0]); keys = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip(): continue
            item = json.loads(line); key = _key(item["agent_name"], item["state"])
            hit = key in selected; keys.add(key)
            cell = cells[(item["agent_name"], int(float(item["state"][17])))]
            cell[1] += 1; cell[0] += int(hit)
    return {"semantic_agent_state_overlap_with_train_selection": _rate(len(keys & selected), len(keys)),
            "per_agent_evidence": {f"{a}:{e}": _rate(v[0], v[1])
                                   for (a, e), v in sorted(cells.items())}}


def _compatible_entries(records, *, split: str, cache: PriorCache, registry_sha: str) -> int:
    hits = 0
    for item in records:
        identity = make_identity(
            split=split, model_alias=MODEL_ALIAS, provider=PROVIDER,
            exact_model_id=EXACT_MODEL_ID, registry_version=1,
            registry_sha256=registry_sha, prompt_version=PROMPT_VERSION,
            agent_name=item.agent_name, generation_config=GENERATION_CONFIG, state=item.state,
        )
        hits += int(cache.load(identity) is not None)
    return hits


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="outputs/formal_replay_final_20260917/train.jsonl")
    p.add_argument("--validation-replay", default="outputs/formal_replay_final_20260917/validation.jsonl")
    p.add_argument("--validation-bank", default="outputs/lwm_rl_v2/b2/state_bank.jsonl")
    p.add_argument("--cache", default="outputs/lwm_rl_final_20260917/prior_cache_final")
    p.add_argument("--registry", default="configs/llm_model_registry_v1.yaml")
    p.add_argument("--target-size", type=int, default=2000)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    train_records = load_train_state_bank(_path(args.train))
    selected_records = select_representative_states(train_records, args.target_size)
    selected = {(x.agent_name, x.state_sha256) for x in selected_records}
    _, registry_sha = load_frozen_registry(_path(args.registry))
    cache = PriorCache(_path(args.cache))
    validation_records = load_train_state_bank(_path(args.validation_replay))
    report = {
        "schema": "priorrl_exact_cache_coverage_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider_calls": 0, "target_size": args.target_size,
        "identity_semantics": "exact_float32_D27_plus_agent_plus_split_and_provenance",
        "train": _audit_replay(_path(args.train), selected),
        "validation_replay": _audit_replay(_path(args.validation_replay), selected),
        "validation_bank": _audit_validation_bank(_path(args.validation_bank), selected),
        "compatible_cache_entries": {
            "selected_train": _rate(_compatible_entries(selected_records, split="train", cache=cache,
                                                          registry_sha=registry_sha), len(selected_records)),
            "final_validation_replay_unique": _rate(
                _compatible_entries(validation_records, split="validation", cache=cache,
                                    registry_sha=registry_sha), len(validation_records)),
        },
        "conclusion": "BLOCKED_EXACT_CACHE_CANNOT_SUPPORT_ONLINE_PPO" ,
    }
    out = _path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(out), "conclusion": report["conclusion"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
