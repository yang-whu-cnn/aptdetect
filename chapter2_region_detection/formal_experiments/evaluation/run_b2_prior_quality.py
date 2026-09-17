from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch

from formal_experiments.evaluation.build_llm_validation_state_bank import (
    DEFAULT_OUT as DEFAULT_STATE_BANK,
    DEFAULT_SUMMARY as DEFAULT_STATE_BANK_SUMMARY,
    state_sha256,
)
from formal_experiments.evaluation.run_b2_multi_model_preflight import (
    RETRY_SLEEP_SECONDS,
    classify_preflight_exception,
    make_live_client,
    validate_raw_prior_contract,
)
from formal_experiments.ours.llm_prior_v2 import (
    FORMAL_HORIZON,
    FORMAL_K_CANDIDATES,
    PriorBatch,
    build_formal_prompt,
    deterministic_fallback_plans,
    normalize_prior_scores,
    parse_prior_response,
)
from formal_experiments.ours.model_registry import (
    DEFAULT_REGISTRY,
    LLMModelSpec,
    load_model_registry,
)
from formal_experiments.ours.ofox_prior_client import response_format_json_schema, sha256_text
from formal_experiments.ours.prior_cache import PriorCache, make_identity
from formal_experiments.training.bootstrap_world_model import BootstrapProbabilisticWorldModel
from formal_experiments.training.response_reward_predictor import ResponseRewardPredictor
from shared.action_contract import N_ACTIONS
from shared.formal_state import FORMAL_STATE_DIM
from shared.rollout_evaluator import SharedRolloutConfig, SharedRolloutEvaluator


THIS = Path(__file__).resolve()
PROJ = THIS.parents[2]
DEFAULT_WORLD_MODEL = "outputs/world_model_v2/a4_5b/world_model_absolute.pt"
DEFAULT_REWARD_MODEL = "outputs/world_model_v2/a4_5c/response_reward_predictor.pt"
DEFAULT_CACHE_ROOT = "outputs/lwm_rl_v2/prior_cache"
DEFAULT_OUT_ROOT = "outputs/lwm_rl_v2/b2"
EXPECTED_BANK_RECORDS = 240


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJ / path).resolve()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(resolve_project_path(path).read_bytes()).hexdigest()


def _canonical_bank_hash(rows: list[dict]) -> str:
    lines = [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def load_frozen_state_bank(
    bank_path: str | Path = DEFAULT_STATE_BANK,
    summary_path: str | Path = DEFAULT_STATE_BANK_SUMMARY,
) -> tuple[list[dict], dict]:
    bank_path = resolve_project_path(bank_path)
    summary_path = resolve_project_path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not bool(summary.get("pass")):
        raise RuntimeError("validation state bank summary is not PASS")
    if int(summary.get("record_count", -1)) != EXPECTED_BANK_RECORDS:
        raise RuntimeError("validation state bank must contain exactly 240 records")
    if int(summary.get("unique_exact_state_count", -1)) != EXPECTED_BANK_RECORDS:
        raise RuntimeError("validation state bank must contain 240 unique exact states")

    rows = [
        json.loads(line)
        for line in bank_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(
        key=lambda row: (
            int(row["episode_seed"]),
            str(row["agent_name"]),
            int(row["decision_index"]),
            int(row["global_tick_start"]),
        )
    )
    if len(rows) != EXPECTED_BANK_RECORDS:
        raise RuntimeError("state bank file count mismatch")
    if len({str(row["state_sha256"]) for row in rows}) != EXPECTED_BANK_RECORDS:
        raise RuntimeError("state bank contains duplicate state hashes")
    if _canonical_bank_hash(rows) != str(summary.get("bank_sha256", "")):
        raise RuntimeError("state bank hash does not match frozen summary")

    expected_config_hash = str(summary.get("selection_config_sha256", ""))
    for row in rows:
        if str(row.get("split")) != "validation":
            raise RuntimeError("prior-quality state bank must be validation-only")
        if str(row.get("selection_config_sha256")) != expected_config_hash:
            raise RuntimeError("state bank selection-config hash mismatch")
        state = np.asarray(row["state"], dtype=np.float32)
        if state.shape != (FORMAL_STATE_DIM,) or not np.isfinite(state).all():
            raise RuntimeError("state bank contains invalid D27 state")
        if state_sha256(state) != str(row["state_sha256"]):
            raise RuntimeError("state bank record SHA256 mismatch")
    return rows, summary


def _usage_dict(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

    def read(name: str):
        value = getattr(usage, name, None)
        return None if value is None else int(value)

    return {
        "prompt_tokens": read("prompt_tokens"),
        "completion_tokens": read("completion_tokens"),
        "total_tokens": read("total_tokens"),
    }


def estimate_cost_usd(spec: LLMModelSpec, usage: Mapping[str, object]) -> float | None:
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if prompt is None or completion is None:
        return None
    return (
        float(prompt) * float(spec.pricing_snapshot.input)
        + float(completion) * float(spec.pricing_snapshot.output)
    ) / 1_000_000.0


def request_prior_once_logical_transaction(
    *,
    spec: LLMModelSpec,
    registry,
    state: np.ndarray,
    agent_name: str,
    client,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[PriorBatch, dict]:
    cfg = spec.to_prior_config(
        prompt_version=registry.prompt_version,
        k_candidates=registry.k_candidates,
        horizon=registry.horizon,
    )
    messages = build_formal_prompt(state, agent_name=agent_name, config=cfg)
    prompt_material = "\n".join(str(item["content"]) for item in messages)

    attempts = 0
    response = None
    last_error: Exception | None = None
    started = time.perf_counter()
    while attempts < spec.retry_policy.max_attempts:
        attempts += 1
        try:
            response = client.chat.completions.create(
                model=spec.exact_model_id,
                messages=messages,
                temperature=float(cfg.temperature),
                response_format=response_format_json_schema(),
            )
            break
        except Exception as exc:
            last_error = exc
            failure_class = classify_preflight_exception(exc)
            if (
                failure_class not in set(spec.retry_policy.retryable_classes)
                or attempts >= spec.retry_policy.max_attempts
            ):
                raise RuntimeError(
                    f"provider transaction failed: class={failure_class}; model={spec.exact_model_id}"
                ) from exc
            sleep_fn(RETRY_SLEEP_SECONDS)

    if response is None:
        raise RuntimeError("provider transaction produced no response") from last_error

    latency_ms = (time.perf_counter() - started) * 1000.0
    try:
        text = str(response.choices[0].message.content or "").strip()
    except Exception as exc:
        raise RuntimeError("provider returned invalid Chat Completions shape") from exc

    raw_contract = validate_raw_prior_contract(
        text,
        k=registry.k_candidates,
        h=registry.horizon,
    )
    prior = parse_prior_response(text, config=cfg)
    fallback_count = sum(source != "llm" for source in prior.sources)
    usage = _usage_dict(response)
    cost_usd = estimate_cost_usd(spec, usage)

    metadata = {
        "api_success": True,
        "fallback_count": int(fallback_count),
        "prompt_sha256": sha256_text(prompt_material),
        "response_sha256": sha256_text(text),
        "latency_ms": float(latency_ms),
        "retry_count": max(0, attempts - 1),
        "attempts": int(attempts),
        "response_model": str(getattr(response, "model", "") or ""),
        "json_valid": bool(raw_contract["json_valid"]),
        "schema_valid": bool(raw_contract["schema_valid"]),
        "semantic_valid_candidate_count": int(raw_contract["semantic_valid_candidate_count"]),
        "duplicate_candidate_count": int(raw_contract["duplicate_candidate_count"]),
        "usage": usage,
        "cost_usd": None if cost_usd is None else float(cost_usd),
    }
    return prior, metadata


def build_evaluator(*, world_model_path, reward_model_path, device: str) -> SharedRolloutEvaluator:
    world_model = BootstrapProbabilisticWorldModel.load_checkpoint(
        resolve_project_path(world_model_path),
        device=device,
    )
    reward_predictor = ResponseRewardPredictor.load_checkpoint(
        resolve_project_path(reward_model_path),
        device=device,
    )
    if int(world_model.config.state_dim) != FORMAL_STATE_DIM:
        raise RuntimeError("world model is not D27")
    if int(world_model.config.n_actions) != N_ACTIONS:
        raise RuntimeError("world model is not A4")
    if int(world_model.config.ensemble_size) != 5:
        raise RuntimeError("world model is not M5")
    if str(world_model.config.target_mode) != "absolute":
        raise RuntimeError("world model must be the frozen absolute checkpoint")
    return SharedRolloutEvaluator(
        world_model=world_model,
        reward_predictor=reward_predictor,
        config=SharedRolloutConfig(horizon=FORMAL_HORIZON, gamma_tick=0.99),
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average
        start = end
    return ranks


def spearman(values_a, values_b) -> float | None:
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        raise ValueError("Spearman inputs must be equal 1D arrays with n>=2")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Spearman inputs must be finite")
    ra = _average_ranks(a)
    rb = _average_ranks(b)
    if float(np.std(ra)) == 0.0 or float(np.std(rb)) == 0.0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    p = counts[counts > 0.0] / total
    return float(-(p * np.log(p)).sum())


def evaluate_prior(state, prior: PriorBatch, evaluator: SharedRolloutEvaluator) -> dict:
    rollout = evaluator.evaluate(state, prior.plans)
    values = rollout.expected_return.detach().cpu().numpy().astype(np.float64)
    members = rollout.member_returns.detach().cpu().numpy().astype(np.float64)
    uncertainty = members.std(axis=1, ddof=0)
    prefs = np.asarray(prior.prior_preferences, dtype=np.float64)
    plans = np.asarray(prior.plans, dtype=np.int64)

    top_index = int(np.argmax(prefs))
    best_index = int(np.argmax(values))
    strictly_better = int(np.sum(values > values[top_index]))
    correlation = spearman(prefs, values)
    unique_plans = len({tuple(int(x) for x in row) for row in plans.tolist()})

    return {
        "top_prior_index": top_index,
        "top_prior_value": float(values[top_index]),
        "top_prior_rank_by_value": int(1 + strictly_better),
        "best_value_index": best_index,
        "best_of_k_value": float(values[best_index]),
        "mean_candidate_value": float(values.mean()),
        "candidate_value_spread": float(values.max() - values.min()),
        "mean_uncertainty": float(uncertainty.mean()),
        "max_uncertainty": float(uncertainty.max()),
        "prior_value_spearman": correlation,
        "unique_plan_ratio": float(unique_plans / FORMAL_K_CANDIDATES),
        "all_no_op_count": int(np.sum(np.all(plans == 0, axis=1))),
        "all_restore_count": int(np.sum(np.all(plans == 3, axis=1))),
        "action_ids_present": sorted(int(x) for x in np.unique(plans)),
        "plans": plans.tolist(),
        "prior_preferences": prefs.tolist(),
        "predicted_values": values.tolist(),
        "predictive_uncertainty": uncertainty.tolist(),
    }


def uniform_prior() -> PriorBatch:
    plans = np.asarray(
        deterministic_fallback_plans(existing=(), needed=FORMAL_K_CANDIDATES, seed=20260916),
        dtype=np.int64,
    )
    scores = np.ones(FORMAL_K_CANDIDATES, dtype=np.float32)
    return PriorBatch(
        plans=plans,
        raw_prior_scores=scores,
        prior_preferences=normalize_prior_scores(scores),
        sources=("uniform_baseline",) * FORMAL_K_CANDIDATES,
    )


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_model_records(records: list[dict], *, expected_states: int) -> dict:
    successes = [row for row in records if row.get("status") == "ok"]
    failures = [row for row in records if row.get("status") != "ok"]
    metadata = [row["metadata"] for row in successes]
    quality = [row["quality"] for row in successes]

    semantic_valid = sum(int(item.get("semantic_valid_candidate_count", 0)) for item in metadata)
    duplicates = sum(int(item.get("duplicate_candidate_count", 0)) for item in metadata)
    fallback = sum(int(item.get("fallback_count", 0)) for item in metadata)
    cache_hits = sum(bool(row.get("cache_hit")) for row in successes)
    cache_misses = len(successes) - cache_hits

    all_plans = [
        action
        for row in quality
        for plan in row["plans"]
        for action in plan
    ]
    position_counts = []
    for position in range(FORMAL_HORIZON):
        counts = np.zeros(N_ACTIONS, dtype=np.int64)
        for row in quality:
            for plan in row["plans"]:
                counts[int(plan[position])] += 1
        entropy = _entropy_from_counts(counts)
        position_counts.append(
            {
                "position": position,
                "counts": counts.tolist(),
                "entropy_nats": entropy,
                "entropy_normalized": float(entropy / math.log(N_ACTIONS)),
            }
        )

    correlations = [
        float(row["prior_value_spearman"])
        for row in quality
        if row["prior_value_spearman"] is not None
    ]
    costs = [float(item["cost_usd"]) for item in metadata if item.get("cost_usd") is not None]
    latencies = [float(item["latency_ms"]) for item in metadata]
    prompt_tokens = [int(item["usage"]["prompt_tokens"]) for item in metadata if item.get("usage", {}).get("prompt_tokens") is not None]
    completion_tokens = [int(item["usage"]["completion_tokens"]) for item in metadata if item.get("usage", {}).get("completion_tokens") is not None]

    return {
        "expected_states": int(expected_states),
        "completed_states": len(successes),
        "failure_count": len(failures),
        "api_success_rate": float(len(successes) / expected_states),
        "schema_valid_response_rate": float(
            sum(bool(item.get("schema_valid")) for item in metadata) / expected_states
        ),
        "semantic_valid_candidate_rate": float(
            semantic_valid / (expected_states * FORMAL_K_CANDIDATES)
        ),
        "fallback_candidate_rate": float(
            fallback / (expected_states * FORMAL_K_CANDIDATES)
        ),
        "duplicate_candidate_rate": float(
            duplicates / (expected_states * FORMAL_K_CANDIDATES)
        ),
        "retry_state_rate": float(
            sum(int(item.get("retry_count", 0)) > 0 for item in metadata) / expected_states
        ),
        "cache_hits_this_run": int(cache_hits),
        "cache_misses_this_run": int(cache_misses),
        "live_api_calls_this_run": int(cache_misses),
        "unique_plan_ratio_mean": _mean([float(row["unique_plan_ratio"]) for row in quality]),
        "per_position_action_entropy": position_counts,
        "global_action_coverage": sorted(set(int(x) for x in all_plans)),
        "all_no_op_plan_rate": float(
            sum(int(row["all_no_op_count"]) for row in quality)
            / (max(1, len(quality)) * FORMAL_K_CANDIDATES)
        ),
        "all_restore_plan_rate": float(
            sum(int(row["all_restore_count"]) for row in quality)
            / (max(1, len(quality)) * FORMAL_K_CANDIDATES)
        ),
        "top_prior_value_mean": _mean([float(row["top_prior_value"]) for row in quality]),
        "best_of_k_value_mean": _mean([float(row["best_of_k_value"]) for row in quality]),
        "mean_candidate_value_mean": _mean([float(row["mean_candidate_value"]) for row in quality]),
        "candidate_value_spread_mean": _mean([float(row["candidate_value_spread"]) for row in quality]),
        "mean_uncertainty_mean": _mean([float(row["mean_uncertainty"]) for row in quality]),
        "max_uncertainty_mean": _mean([float(row["max_uncertainty"]) for row in quality]),
        "prior_value_spearman_mean_finite": _mean(correlations),
        "prior_value_spearman_defined_states": len(correlations),
        "top_prior_rank_mean": _mean([float(row["top_prior_rank_by_value"]) for row in quality]),
        "latency_ms_p50": _percentile(latencies, 50),
        "latency_ms_p95": _percentile(latencies, 95),
        "prompt_tokens_total": None if len(prompt_tokens) != len(metadata) else int(sum(prompt_tokens)),
        "completion_tokens_total": None if len(completion_tokens) != len(metadata) else int(sum(completion_tokens)),
        "estimated_cost_usd_total": None if len(costs) != len(metadata) else float(sum(costs)),
        "pass": bool(len(successes) == expected_states and not failures),
    }


def evaluate_uniform_baseline(rows: list[dict], evaluator: SharedRolloutEvaluator) -> dict:
    prior = uniform_prior()
    records = []
    for row in rows:
        state = np.asarray(row["state"], dtype=np.float32)
        records.append(
            {
                "status": "ok",
                "state_sha256": str(row["state_sha256"]),
                "episode_seed": int(row["episode_seed"]),
                "agent_name": str(row["agent_name"]),
                "quality": evaluate_prior(state, prior, evaluator),
            }
        )
    quality = [row["quality"] for row in records]
    position_counts = []
    for position in range(FORMAL_HORIZON):
        counts = np.zeros(N_ACTIONS, dtype=np.int64)
        for plan in prior.plans.tolist():
            counts[int(plan[position])] += 1
        entropy = _entropy_from_counts(counts)
        position_counts.append(
            {
                "position": position,
                "counts": counts.tolist(),
                "entropy_nats": entropy,
                "entropy_normalized": float(entropy / math.log(N_ACTIONS)),
            }
        )
    summary = {
        "expected_states": len(rows),
        "completed_states": len(records),
        "failure_count": 0,
        "api_success_rate": None,
        "schema_valid_response_rate": None,
        "semantic_valid_candidate_rate": None,
        "fallback_candidate_rate": None,
        "duplicate_candidate_rate": None,
        "retry_state_rate": None,
        "cache_hits_this_run": None,
        "cache_misses_this_run": None,
        "live_api_calls_this_run": 0,
        "unique_plan_ratio_mean": 1.0,
        "per_position_action_entropy": position_counts,
        "global_action_coverage": sorted(int(x) for x in np.unique(prior.plans)),
        "all_no_op_plan_rate": float(np.sum(np.all(prior.plans == 0, axis=1)) / FORMAL_K_CANDIDATES),
        "all_restore_plan_rate": float(np.sum(np.all(prior.plans == 3, axis=1)) / FORMAL_K_CANDIDATES),
        "top_prior_value_mean": _mean([float(row["top_prior_value"]) for row in quality]),
        "best_of_k_value_mean": _mean([float(row["best_of_k_value"]) for row in quality]),
        "mean_candidate_value_mean": _mean([float(row["mean_candidate_value"]) for row in quality]),
        "candidate_value_spread_mean": _mean([float(row["candidate_value_spread"]) for row in quality]),
        "mean_uncertainty_mean": _mean([float(row["mean_uncertainty"]) for row in quality]),
        "max_uncertainty_mean": _mean([float(row["max_uncertainty"]) for row in quality]),
        "prior_value_spearman_mean_finite": None,
        "prior_value_spearman_defined_states": 0,
        "top_prior_rank_mean": _mean([float(row["top_prior_rank_by_value"]) for row in quality]),
        "latency_ms_p50": None,
        "latency_ms_p95": None,
        "prompt_tokens_total": None,
        "completion_tokens_total": None,
        "estimated_cost_usd_total": 0.0,
        "pass": True,
    }
    return {
        "experiment_alias": "uniform_non_llm",
        "provider": "none",
        "prior_tied": True,
        "fixed_plans": prior.plans.tolist(),
        "summary": summary,
        "states": records,
        "pass": True,
    }


def run_model(
    *,
    spec: LLMModelSpec,
    registry,
    registry_sha256: str,
    rows: list[dict],
    evaluator: SharedRolloutEvaluator,
    cache: PriorCache,
    client_factory: Callable[[], object],
) -> dict:
    client_holder: dict[str, object] = {}
    records: list[dict] = []
    generation_config = {
        "temperature": float(spec.actual_temperature),
        "structured_output": str(spec.structured_output_requested),
    }

    for row in rows:
        state = np.asarray(row["state"], dtype=np.float32)
        identity = make_identity(
            split="validation",
            model_alias=spec.experiment_alias,
            provider=spec.provider,
            exact_model_id=spec.exact_model_id,
            registry_version=registry.version,
            registry_sha256=registry_sha256,
            prompt_version=registry.prompt_version,
            generation_config=generation_config,
            state=state,
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
            quality = evaluate_prior(state, cached.prior, evaluator)
            records.append(
                {
                    "status": "ok",
                    "state_sha256": str(row["state_sha256"]),
                    "episode_seed": int(row["episode_seed"]),
                    "agent_name": str(row["agent_name"]),
                    "decision_index": int(row["decision_index"]),
                    "cache_hit": bool(lookup.cache_hit),
                    "cache_key": str(cached.cache_key),
                    "metadata": dict(cached.metadata),
                    "quality": quality,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "status": "failed",
                    "state_sha256": str(row["state_sha256"]),
                    "episode_seed": int(row["episode_seed"]),
                    "agent_name": str(row["agent_name"]),
                    "decision_index": int(row["decision_index"]),
                    "cache_hit": False,
                    "failure_class": classify_preflight_exception(exc),
                    "error": str(exc),
                }
            )

    summary = summarize_model_records(records, expected_states=len(rows))
    return {
        "tier": spec.registry_tier,
        "experiment_alias": spec.experiment_alias,
        "provider": spec.provider,
        "exact_model_id": spec.exact_model_id,
        "requested_temperature": spec.requested_temperature,
        "actual_temperature": spec.actual_temperature,
        "structured_output_verified_preflight": bool(spec.structured_output_verified),
        "pricing_snapshot": {
            "currency": spec.pricing_snapshot.currency,
            "unit": spec.pricing_snapshot.unit,
            "input": spec.pricing_snapshot.input,
            "output": spec.pricing_snapshot.output,
            "source": spec.pricing_snapshot.source,
            "snapshot_date": spec.pricing_snapshot.snapshot_date,
        },
        "summary": summary,
        "states": records,
        "pass": bool(summary["pass"]),
    }


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def run_prior_quality(
    *,
    registry_path=DEFAULT_REGISTRY,
    bank_path=DEFAULT_STATE_BANK,
    summary_path=DEFAULT_STATE_BANK_SUMMARY,
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
        raise RuntimeError("primary LLM must remain unselected during prior-quality study")
    for spec in registry.models.values():
        if not spec.structured_output_verified or spec.actual_temperature is None:
            raise RuntimeError(f"model capability preflight not frozen: {spec.experiment_alias}")

    rows, bank_summary = load_frozen_state_bank(bank_path, summary_path)
    evaluator = build_evaluator(
        world_model_path=world_model_path,
        reward_model_path=reward_model_path,
        device=str(device),
    )
    cache = PriorCache(resolve_project_path(cache_root))
    registry_hash = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    out_root = resolve_project_path(out_root)

    model_reports = []
    for tier in ("tier_h", "tier_m", "tier_l"):
        spec = registry.get(tier)
        report = run_model(
            spec=spec,
            registry=registry,
            registry_sha256=registry_hash,
            rows=rows,
            evaluator=evaluator,
            cache=cache,
            client_factory=client_factory,
        )
        write_report(out_root / spec.experiment_alias / "prior_quality.json", report)
        model_reports.append(report)

    uniform = evaluate_uniform_baseline(rows, evaluator)
    write_report(out_root / "uniform_non_llm" / "prior_quality.json", uniform)

    overall_pass = all(report["pass"] for report in model_reports) and uniform["pass"]
    summary = {
        "phase": "gate_b2_prior_quality",
        "registry_sha256": registry_hash,
        "bank_sha256": str(bank_summary["bank_sha256"]),
        "state_count": len(rows),
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
        "uniform": uniform["summary"],
        "pass": bool(overall_pass),
    }
    write_report(out_root / "prior_quality_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate B2 cached multi-LLM prior-quality evaluation")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--bank", default=DEFAULT_STATE_BANK)
    parser.add_argument("--bank-summary", default=DEFAULT_STATE_BANK_SUMMARY)
    parser.add_argument("--world-model", default=DEFAULT_WORLD_MODEL)
    parser.add_argument("--reward-model", default=DEFAULT_REWARD_MODEL)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    summary = run_prior_quality(
        registry_path=args.registry,
        bank_path=args.bank,
        summary_path=args.bank_summary,
        world_model_path=args.world_model,
        reward_model_path=args.reward_model,
        cache_root=args.cache_root,
        out_root=args.out_root,
        device=args.device,
    )

    print("=" * 80)
    print("[GATE B2 PRIOR QUALITY]")
    print("state_count:", summary["state_count"])
    print("bank_sha256:", summary["bank_sha256"])
    for item in summary["models"]:
        s = item["summary"]
        print(
            f"{item['tier']} {item['exact_model_id']}: "
            f"completed={s['completed_states']}/240 "
            f"schema={s['schema_valid_response_rate']:.4f} "
            f"fallback={s['fallback_candidate_rate']:.4f} "
            f"best_value={s['best_of_k_value_mean']:.6f} "
            f"top_value={s['top_prior_value_mean']:.6f} "
            f"spearman={s['prior_value_spearman_mean_finite']} "
            f"calls={s['live_api_calls_this_run']} "
            f"cost={s['estimated_cost_usd_total']} "
            f"pass={item['pass']}"
        )
    u = summary["uniform"]
    print(
        "uniform_non_llm:",
        f"completed={u['completed_states']}/240",
        f"best_value={u['best_of_k_value_mean']:.6f}",
        f"pass={u['pass']}",
    )
    print("primary_model_selected:", summary["primary_model_selected"])
    print("pass:", summary["pass"])
    print("[OK] summary:", resolve_project_path(DEFAULT_OUT_ROOT) / "prior_quality_summary.json")
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
