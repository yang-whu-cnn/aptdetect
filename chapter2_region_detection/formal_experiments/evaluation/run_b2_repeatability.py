from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

from formal_experiments.evaluation.build_llm_validation_state_bank import state_sha256
from formal_experiments.evaluation.run_b2_multi_model_preflight import make_live_client
from formal_experiments.evaluation.run_b2_prior_quality import (
    EXPECTED_BANK_RECORDS,
    DEFAULT_REGISTRY,
    DEFAULT_REWARD_MODEL,
    DEFAULT_STATE_BANK,
    DEFAULT_STATE_BANK_SUMMARY,
    DEFAULT_WORLD_MODEL,
    build_evaluator,
    evaluate_prior,
    file_sha256,
    load_frozen_state_bank,
    request_prior_once_logical_transaction,
    resolve_project_path,
)
from formal_experiments.ours.model_registry import LLMModelSpec, load_model_registry
from formal_experiments.ours.prior_cache import PriorCache, make_identity
from shared.formal_state import BLUE_AGENTS, FORMAL_STATE_DIM


REPEATABILITY_FORMAT_VERSION = 1
REPEATABILITY_STATE_COUNT = 30
REPLICATES = 3
EXPERIMENT_TAG = "b2_repeatability_v1"
DEFAULT_CACHE_ROOT = "outputs/lwm_rl_v2/prior_cache_repeatability"
DEFAULT_OUT_ROOT = "outputs/lwm_rl_v2/b2/repeatability"


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json_lines(rows: list[dict]) -> str:
    payload = "\n".join(_canonical_json(row) for row in rows) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_repeatability_subset(rows: list[dict]) -> tuple[list[dict], dict]:
    """Select 30 deterministic validation states, balanced by agent and feature17."""
    if len(rows) != EXPECTED_BANK_RECORDS:
        raise ValueError("repeatability selection requires the frozen 240-state bank")

    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(int(row["episode_seed"]), str(row["agent_name"]))].append(row)
    for cell_rows in groups.values():
        cell_rows.sort(
            key=lambda row: (
                int(row["decision_index"]),
                int(row["global_tick_start"]),
                str(row["state_sha256"]),
            )
        )

    seeds = tuple(sorted({int(row["episode_seed"]) for row in rows}))
    agents = tuple(BLUE_AGENTS)
    if len(seeds) != 8 or len(agents) != 5:
        raise ValueError("repeatability protocol requires 8 validation seeds and 5 Blue agents")

    selected: list[dict] = []
    seen_cells: set[tuple[int, str]] = set()
    seen_hashes: set[str] = set()

    for position in range(REPEATABILITY_STATE_COUNT):
        agent = agents[position % len(agents)]
        seed = seeds[position % len(seeds)]
        desired_feature17 = position % 2
        cell = (seed, agent)
        if cell in seen_cells:
            raise RuntimeError("repeatability selection repeated a seed-agent cell")
        seen_cells.add(cell)

        candidates = [
            row
            for row in groups.get(cell, [])
            if int(row["feature17_any_valid_observable_target"]) == desired_feature17
        ]
        if not candidates:
            raise ValueError(
                f"repeatability cell {seed}:{agent} lacks feature17={desired_feature17}"
            )
        chosen = candidates[(len(candidates) - 1) // 2]
        digest = str(chosen["state_sha256"])
        if digest in seen_hashes:
            raise RuntimeError("repeatability subset contains duplicate exact D27 state")
        seen_hashes.add(digest)
        selected.append(dict(chosen))

    selected.sort(
        key=lambda row: (
            int(row["episode_seed"]),
            str(row["agent_name"]),
            int(row["decision_index"]),
            int(row["global_tick_start"]),
        )
    )

    per_agent = Counter(str(row["agent_name"]) for row in selected)
    per_seed = Counter(int(row["episode_seed"]) for row in selected)
    feature17 = Counter(int(row["feature17_any_valid_observable_target"]) for row in selected)

    if len(selected) != REPEATABILITY_STATE_COUNT:
        raise RuntimeError("repeatability subset count mismatch")
    if any(per_agent[agent] != 6 for agent in agents):
        raise RuntimeError("repeatability subset must contain exactly 6 states per agent")
    if sorted(per_seed.values()) != [3, 3, 4, 4, 4, 4, 4, 4]:
        raise RuntimeError("repeatability seed coverage must be six seeds x4 plus two seeds x3")
    if feature17 != Counter({0: 15, 1: 15}):
        raise RuntimeError("repeatability subset must be feature17-balanced 15/15")

    summary = {
        "format_version": REPEATABILITY_FORMAT_VERSION,
        "selection_algorithm": "position_mod_agent_seed_feature17_v1",
        "record_count": len(selected),
        "unique_exact_state_count": len(seen_hashes),
        "per_agent": {agent: int(per_agent[agent]) for agent in agents},
        "per_seed": {str(seed): int(per_seed[seed]) for seed in seeds},
        "feature17_counts": {"0": int(feature17[0]), "1": int(feature17[1])},
        "subset_sha256": _sha256_json_lines(selected),
        "pass": True,
    }
    return selected, summary


def make_repeatability_identity(
    *,
    spec: LLMModelSpec,
    registry,
    registry_sha256: str,
    state: np.ndarray,
    replicate: int,
):
    if replicate not in (0, 1, 2):
        raise ValueError("repeatability replicate must be 0,1,2")
    return make_identity(
        split="validation",
        model_alias=spec.experiment_alias,
        provider=spec.provider,
        exact_model_id=spec.exact_model_id,
        registry_version=registry.version,
        registry_sha256=registry_sha256,
        prompt_version=registry.prompt_version,
        generation_config={
            "temperature": float(spec.actual_temperature),
            "structured_output": str(spec.structured_output_requested),
            "experiment": EXPERIMENT_TAG,
            "replicate": int(replicate),
        },
        state=state,
    )


def _plan_set(record: dict) -> set[tuple[int, ...]]:
    return {tuple(int(x) for x in plan) for plan in record["quality"]["plans"]}


def _top_plan(record: dict) -> tuple[int, ...]:
    quality = record["quality"]
    index = int(quality["top_prior_index"])
    return tuple(int(x) for x in quality["plans"][index])


def analyze_state_replicates(records: list[dict]) -> dict:
    if len(records) != REPLICATES or any(row.get("status") != "ok" for row in records):
        raise ValueError("state repeatability analysis requires exactly 3 successful replicates")
    records = sorted(records, key=lambda row: int(row["replicate"]))
    if [int(row["replicate"]) for row in records] != [0, 1, 2]:
        raise ValueError("repeatability replicate IDs must be 0,1,2")

    sets = [_plan_set(row) for row in records]
    pairs = ((0, 1), (0, 2), (1, 2))
    jaccards: list[float] = []
    overlaps: list[int] = []
    top_agreement: list[int] = []
    for left, right in pairs:
        intersection = sets[left] & sets[right]
        union = sets[left] | sets[right]
        jaccards.append(float(len(intersection) / len(union)))
        overlaps.append(int(len(intersection)))
        top_agreement.append(int(_top_plan(records[left]) == _top_plan(records[right])))

    top_plans = [_top_plan(row) for row in records]
    top_prefs = [
        float(row["quality"]["prior_preferences"][int(row["quality"]["top_prior_index"])])
        for row in records
    ]

    per_plan_prefs: dict[tuple[int, ...], list[float]] = defaultdict(list)
    for row in records:
        for plan, pref in zip(row["quality"]["plans"], row["quality"]["prior_preferences"]):
            per_plan_prefs[tuple(int(x) for x in plan)].append(float(pref))
    matched_variances = [
        float(np.var(np.asarray(values, dtype=np.float64), ddof=0))
        for values in per_plan_prefs.values()
        if len(values) >= 2
    ]

    best_values = np.asarray(
        [float(row["quality"]["best_of_k_value"]) for row in records],
        dtype=np.float64,
    )

    return {
        "pairwise_candidate_set_jaccard": jaccards,
        "candidate_set_jaccard_mean": float(np.mean(jaccards)),
        "pairwise_candidate_overlap_count": overlaps,
        "candidate_overlap_count_mean": float(np.mean(overlaps)),
        "top_prior_pair_agreement": top_agreement,
        "top_prior_pair_agreement_mean": float(np.mean(top_agreement)),
        "top_prior_all_three_agree": bool(top_plans[0] == top_plans[1] == top_plans[2]),
        "top_prior_preference_variance": float(np.var(np.asarray(top_prefs), ddof=0)),
        "matched_plan_preference_variance_mean": (
            None if not matched_variances else float(np.mean(matched_variances))
        ),
        "matched_plan_count": int(len(matched_variances)),
        "best_of_k_values": best_values.tolist(),
        "best_of_k_value_variance": float(np.var(best_values, ddof=0)),
        "best_of_k_value_range": float(best_values.max() - best_values.min()),
    }


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_repeatability(
    state_results: list[dict],
    generation_records: list[dict],
    *,
    expected_states: int,
) -> dict:
    successful = [row for row in generation_records if row.get("status") == "ok"]
    failed = [row for row in generation_records if row.get("status") != "ok"]
    analyses = [row["repeatability"] for row in state_results if row.get("status") == "ok"]
    metadata = [row["metadata"] for row in successful]
    expected_generations = expected_states * REPLICATES
    cache_hits = sum(bool(row.get("cache_hit")) for row in successful)
    live_records = [row for row in successful if not bool(row.get("cache_hit"))]

    matched = [
        float(item["matched_plan_preference_variance_mean"])
        for item in analyses
        if item["matched_plan_preference_variance_mean"] is not None
    ]
    artifact_costs = [float(item["cost_usd"]) for item in metadata if item.get("cost_usd") is not None]
    live_costs = [
        float(row["metadata"]["cost_usd"])
        for row in live_records
        if row["metadata"].get("cost_usd") is not None
    ]
    latencies = [float(item["latency_ms"]) for item in metadata]

    return {
        "expected_states": int(expected_states),
        "completed_states": len(analyses),
        "expected_generations": int(expected_generations),
        "successful_generations": len(successful),
        "failed_generations": len(failed),
        "schema_valid_generation_rate": float(
            sum(bool(item.get("schema_valid")) for item in metadata) / max(1, expected_generations)
        ),
        "semantic_valid_candidate_rate": float(
            sum(int(item.get("semantic_valid_candidate_count", 0)) for item in metadata)
            / max(1, expected_generations * 6)
        ),
        "fallback_candidate_rate": float(
            sum(int(item.get("fallback_count", 0)) for item in metadata)
            / max(1, expected_generations * 6)
        ),
        "duplicate_candidate_rate": float(
            sum(int(item.get("duplicate_candidate_count", 0)) for item in metadata)
            / max(1, expected_generations * 6)
        ),
        "retry_generation_rate": float(
            sum(int(item.get("retry_count", 0)) > 0 for item in metadata)
            / max(1, expected_generations)
        ),
        "cache_hits_this_run": int(cache_hits),
        "cache_misses_this_run": int(len(successful) - cache_hits),
        "live_api_calls_this_run": int(len(live_records)),
        "candidate_set_jaccard_mean": _mean(
            [float(item["candidate_set_jaccard_mean"]) for item in analyses]
        ),
        "candidate_overlap_count_mean": _mean(
            [float(item["candidate_overlap_count_mean"]) for item in analyses]
        ),
        "top_prior_pair_agreement_mean": _mean(
            [float(item["top_prior_pair_agreement_mean"]) for item in analyses]
        ),
        "top_prior_all_three_agreement_rate": float(
            sum(bool(item["top_prior_all_three_agree"]) for item in analyses)
            / max(1, expected_states)
        ),
        "top_prior_preference_variance_mean": _mean(
            [float(item["top_prior_preference_variance"]) for item in analyses]
        ),
        "matched_plan_preference_variance_mean": _mean(matched),
        "matched_plan_preference_variance_defined_states": len(matched),
        "best_of_k_value_variance_mean": _mean(
            [float(item["best_of_k_value_variance"]) for item in analyses]
        ),
        "best_of_k_value_range_mean": _mean(
            [float(item["best_of_k_value_range"]) for item in analyses]
        ),
        "latency_ms_p50": _percentile(latencies, 50),
        "latency_ms_p95": _percentile(latencies, 95),
        "artifact_estimated_cost_usd_total": (
            None if len(artifact_costs) != len(metadata) else float(sum(artifact_costs))
        ),
        "live_api_cost_usd_this_run": (
            None if len(live_costs) != len(live_records) else float(sum(live_costs))
        ),
        "pass": bool(
            len(analyses) == expected_states
            and len(successful) == expected_generations
            and not failed
        ),
    }


def run_model_repeatability(
    *,
    spec: LLMModelSpec,
    registry,
    registry_sha256: str,
    rows: list[dict],
    evaluator,
    cache: PriorCache,
    client_factory: Callable[[], object],
) -> dict:
    client_holder: dict[str, object] = {}
    generation_records: list[dict] = []
    state_results: list[dict] = []

    for state_index, row in enumerate(rows, start=1):
        state = np.asarray(row["state"], dtype=np.float32)
        if state.shape != (FORMAL_STATE_DIM,) or state_sha256(state) != str(row["state_sha256"]):
            raise RuntimeError("repeatability state integrity failure")

        per_state: list[dict] = []
        for replicate in range(REPLICATES):
            identity = make_repeatability_identity(
                spec=spec,
                registry=registry,
                registry_sha256=registry_sha256,
                state=state,
                replicate=replicate,
            )

            def generate():
                if "client" not in client_holder:
                    client_holder["client"] = client_factory()
                return request_prior_once_logical_transaction(
                    spec=spec,
                    registry=registry,
                    state=state,
                    agent_name=str(row["agent_name"]),
                    client=client_holder["client"],
                )

            try:
                lookup = cache.get_or_generate(identity, generate)
                cached = lookup.cached_prior
                record = {
                    "status": "ok",
                    "replicate": int(replicate),
                    "state_sha256": str(row["state_sha256"]),
                    "episode_seed": int(row["episode_seed"]),
                    "agent_name": str(row["agent_name"]),
                    "decision_index": int(row["decision_index"]),
                    "cache_hit": bool(lookup.cache_hit),
                    "cache_key": str(cached.cache_key),
                    "metadata": dict(cached.metadata),
                    "quality": evaluate_prior(state, cached.prior, evaluator),
                }
            except Exception as exc:
                record = {
                    "status": "failed",
                    "replicate": int(replicate),
                    "state_sha256": str(row["state_sha256"]),
                    "episode_seed": int(row["episode_seed"]),
                    "agent_name": str(row["agent_name"]),
                    "decision_index": int(row["decision_index"]),
                    "cache_hit": False,
                    "error_type": type(exc).__name__,
                }
            generation_records.append(record)
            per_state.append(record)

        if all(item["status"] == "ok" for item in per_state):
            state_results.append(
                {
                    "status": "ok",
                    "state_sha256": str(row["state_sha256"]),
                    "episode_seed": int(row["episode_seed"]),
                    "agent_name": str(row["agent_name"]),
                    "decision_index": int(row["decision_index"]),
                    "repeatability": analyze_state_replicates(per_state),
                    "replicates": per_state,
                }
            )
        else:
            state_results.append(
                {
                    "status": "failed",
                    "state_sha256": str(row["state_sha256"]),
                    "episode_seed": int(row["episode_seed"]),
                    "agent_name": str(row["agent_name"]),
                    "decision_index": int(row["decision_index"]),
                    "replicates": per_state,
                }
            )

        live_so_far = sum(
            item.get("status") == "ok" and not bool(item.get("cache_hit"))
            for item in generation_records
        )
        print(
            f"[REPEATABILITY] {spec.experiment_alias} "
            f"state={state_index:02d}/{len(rows)} live_calls_this_run={live_so_far}"
        )

    summary = summarize_repeatability(
        state_results,
        generation_records,
        expected_states=len(rows),
    )
    return {
        "format_version": REPEATABILITY_FORMAT_VERSION,
        "tier": spec.registry_tier,
        "experiment_alias": spec.experiment_alias,
        "provider": spec.provider,
        "exact_model_id": spec.exact_model_id,
        "replicates_per_state": REPLICATES,
        "state_count": len(rows),
        "primary_model_selected": False,
        "summary": summary,
        "states": state_results,
        "pass": bool(summary["pass"]),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_repeatability(
    *,
    registry_path=DEFAULT_REGISTRY,
    bank_path=DEFAULT_STATE_BANK,
    bank_summary_path=DEFAULT_STATE_BANK_SUMMARY,
    world_model_path=DEFAULT_WORLD_MODEL,
    reward_model_path=DEFAULT_REWARD_MODEL,
    cache_root=DEFAULT_CACHE_ROOT,
    out_root=DEFAULT_OUT_ROOT,
    device="cpu",
    client_factory: Callable[[], object] = make_live_client,
) -> dict:
    registry_path = resolve_project_path(registry_path)
    registry = load_model_registry(registry_path)
    if registry.primary_model_selected:
        raise RuntimeError("primary LLM must remain unselected during repeatability")
    for spec in registry.models.values():
        if not spec.structured_output_verified or spec.actual_temperature is None:
            raise RuntimeError(f"model capability is not frozen: {spec.experiment_alias}")

    rows, bank_summary = load_frozen_state_bank(bank_path, bank_summary_path)
    subset, subset_summary = select_repeatability_subset(rows)
    out_root = resolve_project_path(out_root)
    _write_jsonl(out_root / "repeatability_state_subset.jsonl", subset)
    subset_summary = {**subset_summary, "source_bank_sha256": str(bank_summary["bank_sha256"])}
    _write_json(out_root / "repeatability_state_subset_summary.json", subset_summary)

    evaluator = build_evaluator(
        world_model_path=world_model_path,
        reward_model_path=reward_model_path,
        device=str(device),
    )
    cache = PriorCache(resolve_project_path(cache_root))
    registry_hash = file_sha256(registry_path)

    model_reports = []
    for tier in ("tier_h", "tier_m", "tier_l"):
        spec = registry.get(tier)
        report = run_model_repeatability(
            spec=spec,
            registry=registry,
            registry_sha256=registry_hash,
            rows=subset,
            evaluator=evaluator,
            cache=cache,
            client_factory=client_factory,
        )
        _write_json(out_root / f"{spec.experiment_alias}.json", report)
        model_reports.append(report)

    summary = {
        "phase": "gate_b2_repeatability",
        "format_version": REPEATABILITY_FORMAT_VERSION,
        "registry_sha256": registry_hash,
        "source_bank_sha256": str(bank_summary["bank_sha256"]),
        "subset_sha256": str(subset_summary["subset_sha256"]),
        "state_count": len(subset),
        "replicates_per_state": REPLICATES,
        "primary_model_selected": False,
        "models": [
            {
                "tier": report["tier"],
                "experiment_alias": report["experiment_alias"],
                "exact_model_id": report["exact_model_id"],
                "summary": report["summary"],
                "pass": report["pass"],
            }
            for report in model_reports
        ],
        "pass": bool(all(report["pass"] for report in model_reports)),
    }
    _write_json(out_root / "repeatability_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate B2 multi-LLM 30-state x3 repeatability study")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--bank", default=DEFAULT_STATE_BANK)
    parser.add_argument("--bank-summary", default=DEFAULT_STATE_BANK_SUMMARY)
    parser.add_argument("--world-model", default=DEFAULT_WORLD_MODEL)
    parser.add_argument("--reward-model", default=DEFAULT_REWARD_MODEL)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    summary = run_repeatability(
        registry_path=args.registry,
        bank_path=args.bank,
        bank_summary_path=args.bank_summary,
        world_model_path=args.world_model,
        reward_model_path=args.reward_model,
        cache_root=args.cache_root,
        out_root=args.out_root,
        device=args.device,
    )

    print("=" * 80)
    print("[GATE B2 REPEATABILITY]")
    print("state_count:", summary["state_count"])
    print("replicates_per_state:", summary["replicates_per_state"])
    print("source_bank_sha256:", summary["source_bank_sha256"])
    print("subset_sha256:", summary["subset_sha256"])
    for item in summary["models"]:
        s = item["summary"]
        print(
            f"{item['tier']} {item['exact_model_id']}: "
            f"states={s['completed_states']}/30 generations={s['successful_generations']}/90 "
            f"jaccard={s['candidate_set_jaccard_mean']:.6f} "
            f"top_agree={s['top_prior_pair_agreement_mean']:.6f} "
            f"all3={s['top_prior_all_three_agreement_rate']:.6f} "
            f"best_var={s['best_of_k_value_variance_mean']:.6f} "
            f"calls={s['live_api_calls_this_run']} "
            f"cost_this_run={s['live_api_cost_usd_this_run']} pass={item['pass']}"
        )
    print("primary_model_selected:", summary["primary_model_selected"])
    print("pass:", summary["pass"])
    print("[OK] summary:", resolve_project_path(DEFAULT_OUT_ROOT) / "repeatability_summary.json")
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
