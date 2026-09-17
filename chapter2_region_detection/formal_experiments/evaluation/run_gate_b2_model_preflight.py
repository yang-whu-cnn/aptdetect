from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from math import isfinite
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Mapping

import numpy as np

from formal_experiments.evaluation.build_llm_validation_state_bank import (
    DEFAULT_OUT as DEFAULT_STATE_BANK,
    DEFAULT_SUMMARY as DEFAULT_STATE_BANK_SUMMARY,
    state_sha256,
)
from formal_experiments.ours.llm_prior_v2 import (
    FORMAL_ACTION_NAMES,
    FORMAL_HORIZON,
    FORMAL_K_CANDIDATES,
    PriorBatch,
    build_formal_prompt,
    parse_prior_response,
)
from formal_experiments.ours.model_registry import (
    DEFAULT_REGISTRY,
    EXPECTED_TIERS,
    LLMModelRegistry,
    LLMModelSpec,
    load_model_registry,
)
from formal_experiments.ours.ofox_prior_client import (
    OFOX_API_KEY_ENV,
    build_openai_http_client,
    classify_live_exception,
    ofox_network_mode,
    response_format_json_schema,
    sha256_text,
)
from shared.formal_state import FORMAL_STATE_DIM


DEFAULT_OUT = "outputs/lwm_rl_v2/b2/preflight/model_capability_preflight.json"
EXPECTED_BANK_COUNT = 240


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except Exception as exc:
                raise ValueError(f"invalid state-bank JSON at line {line_no}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"state-bank line {line_no} must be an object")
            rows.append(item)
    return rows


def load_frozen_state_bank(
    bank_path: str | Path = DEFAULT_STATE_BANK,
    summary_path: str | Path = DEFAULT_STATE_BANK_SUMMARY,
) -> tuple[list[dict], dict]:
    bank_path = resolve_path(bank_path)
    summary_path = resolve_path(summary_path)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    if not isinstance(summary, dict) or not bool(summary.get("pass")):
        raise ValueError("state-bank summary must be a PASS object")
    if int(summary.get("record_count", -1)) != EXPECTED_BANK_COUNT:
        raise ValueError("state-bank summary record_count must be 240")
    if int(summary.get("unique_exact_state_count", -1)) != EXPECTED_BANK_COUNT:
        raise ValueError("state-bank summary unique_exact_state_count must be 240")
    if str(summary.get("split", "validation")) != "validation":
        raise ValueError("state-bank split must be validation")

    rows = _read_jsonl(bank_path)
    if len(rows) != EXPECTED_BANK_COUNT:
        raise ValueError(f"state-bank must contain 240 records; got {len(rows)}")

    hashes: list[str] = []
    selection_hashes: set[str] = set()
    for index, row in enumerate(rows):
        if str(row.get("split")) != "validation":
            raise ValueError(f"state-bank row {index} is not validation")
        state = np.asarray(row.get("state"), dtype=np.float32)
        if state.shape != (FORMAL_STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError(f"state-bank row {index} has invalid D27 state")
        actual_hash = state_sha256(state)
        if str(row.get("state_sha256")) != actual_hash:
            raise ValueError(f"state-bank row {index} state hash mismatch")
        hashes.append(actual_hash)
        selection_hashes.add(str(row.get("selection_config_sha256", "")))

    if len(set(hashes)) != EXPECTED_BANK_COUNT:
        raise ValueError("state-bank exact D27 states are not unique")
    if len(selection_hashes) != 1 or "" in selection_hashes:
        raise ValueError("state-bank selection_config_sha256 is inconsistent")
    if str(summary.get("selection_config_sha256")) not in selection_hashes:
        raise ValueError("state-bank summary selection hash mismatch")

    canonical_lines = [_canonical_json(row) for row in rows]
    bank_hash = hashlib.sha256(("\n".join(canonical_lines) + "\n").encode("utf-8")).hexdigest()
    if str(summary.get("bank_sha256")) != bank_hash:
        raise ValueError("state-bank summary bank_sha256 mismatch")

    return rows, summary


def select_preflight_state(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("empty state bank")
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["episode_seed"]),
            str(row["agent_name"]),
            int(row["decision_index"]),
            int(row["global_tick_start"]),
            str(row["state_sha256"]),
        ),
    )
    return dict(ordered[0])


def inspect_raw_prior_response(text: str) -> dict:
    report = {
        "json_object_valid": False,
        "schema_valid": False,
        "candidate_count": 0,
        "semantic_valid_candidate_count": 0,
        "unique_valid_plan_count": 0,
        "duplicate_valid_plan_count": 0,
    }
    try:
        obj = json.loads(str(text).strip())
    except Exception:
        return report
    if not isinstance(obj, dict):
        return report
    report["json_object_valid"] = True
    candidates = obj.get("candidates")
    if not isinstance(candidates, list):
        return report
    report["candidate_count"] = len(candidates)

    valid_plans: list[tuple[str, ...]] = []
    all_schema_valid = set(obj.keys()) == {"candidates"} and len(candidates) == FORMAL_K_CANDIDATES
    for item in candidates:
        item_valid = isinstance(item, dict) and set(item.keys()) == {"actions", "prior_score", "reason"}
        if not item_valid:
            all_schema_valid = False
            continue
        actions = item.get("actions")
        score = item.get("prior_score")
        reason = item.get("reason")
        actions_valid = (
            isinstance(actions, list)
            and len(actions) == FORMAL_HORIZON
            and all(isinstance(x, str) and x in FORMAL_ACTION_NAMES for x in actions)
        )
        score_valid = (
            not isinstance(score, bool)
            and isinstance(score, (int, float))
            and isfinite(float(score))
            and 0.0 <= float(score) <= 1.0
        )
        reason_valid = isinstance(reason, str)
        semantic_valid = actions_valid and score_valid
        if semantic_valid:
            report["semantic_valid_candidate_count"] += 1
            valid_plans.append(tuple(str(x) for x in actions))
        if not (actions_valid and score_valid and reason_valid):
            all_schema_valid = False

    unique_count = len(set(valid_plans))
    report["unique_valid_plan_count"] = unique_count
    report["duplicate_valid_plan_count"] = len(valid_plans) - unique_count
    report["schema_valid"] = bool(all_schema_valid)
    return report


def _usage_int(usage, *names: str) -> int | None:
    if usage is None:
        return None
    for name in names:
        if isinstance(usage, Mapping) and name in usage:
            value = usage[name]
        else:
            value = getattr(usage, name, None)
        if value is None:
            continue
        try:
            number = int(value)
        except Exception:
            continue
        if number >= 0:
            return number
    return None


def extract_usage(response) -> dict:
    usage = getattr(response, "usage", None)
    input_tokens = _usage_int(usage, "prompt_tokens", "input_tokens")
    output_tokens = _usage_int(usage, "completion_tokens", "output_tokens")
    total_tokens = _usage_int(usage, "total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "available": input_tokens is not None and output_tokens is not None,
    }


def estimate_cost_usd(spec: LLMModelSpec, usage: dict) -> float | None:
    if not bool(usage.get("available")):
        return None
    return float(
        float(usage["input_tokens"]) * spec.pricing_snapshot.input / 1_000_000.0
        + float(usage["output_tokens"]) * spec.pricing_snapshot.output / 1_000_000.0
    )


def _final_prior_valid(prior: PriorBatch) -> bool:
    plans = np.asarray(prior.plans, dtype=np.int64)
    scores = np.asarray(prior.raw_prior_scores, dtype=np.float32)
    prefs = np.asarray(prior.prior_preferences, dtype=np.float32)
    return bool(
        plans.shape == (FORMAL_K_CANDIDATES, FORMAL_HORIZON)
        and scores.shape == (FORMAL_K_CANDIDATES,)
        and prefs.shape == (FORMAL_K_CANDIDATES,)
        and np.all(plans >= 0)
        and np.all(plans < len(FORMAL_ACTION_NAMES))
        and np.isfinite(scores).all()
        and np.isfinite(prefs).all()
        and np.all(prefs >= 0.0)
        and abs(float(prefs.sum()) - 1.0) <= 1e-5
    )


def _default_client_factory(registry: LLMModelRegistry, *, require_env_key: bool):
    if require_env_key:
        key = str(os.environ.get(OFOX_API_KEY_ENV, "")).strip()
        if not key:
            raise RuntimeError("OFOX_API_KEY is required for live model preflight")
    else:
        key = "test-not-used"
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is required; install with python -m pip install -U openai") from exc
    http_client = build_openai_http_client()
    kwargs = {"base_url": registry.base_url, "api_key": key}
    if http_client is not None:
        kwargs["http_client"] = http_client
    return OpenAI(**kwargs)


def run_model_preflight(
    *,
    registry: LLMModelRegistry,
    tier: str,
    state_record: dict,
    client,
) -> dict:
    spec = registry.get(tier)
    config = spec.to_prior_config(
        prompt_version=registry.prompt_version,
        k_candidates=registry.k_candidates,
        horizon=registry.horizon,
    )
    state = np.asarray(state_record["state"], dtype=np.float32)
    messages = build_formal_prompt(state, agent_name=str(state_record["agent_name"]), config=config)
    prompt_material = "\n".join(str(item["content"]) for item in messages)

    base = {
        "tier": str(tier),
        "experiment_alias": spec.experiment_alias,
        "requested_model": spec.exact_model_id,
        "requested_temperature": float(config.temperature),
        "structured_output_requested": spec.structured_output_requested,
        "state_sha256": str(state_record["state_sha256"]),
        "api_success": False,
        "failure_class": None,
        "exception_type": None,
        "response_model": None,
        "finish_reason": None,
        "latency_ms": None,
        "usage": {"input_tokens": None, "output_tokens": None, "total_tokens": None, "available": False},
        "estimated_cost_usd": None,
        "prompt_sha256": sha256_text(prompt_material),
        "response_sha256": None,
        "raw_prompt_recorded": False,
        "raw_response_recorded": False,
        "api_key_value_recorded": False,
        "cache_used": False,
        "raw_inspection": None,
        "fallback_count": None,
        "final_prior_valid": False,
        "pass": False,
    }

    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=spec.exact_model_id,
            messages=messages,
            temperature=float(config.temperature),
            response_format=response_format_json_schema(),
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        text = str(response.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("empty Chat Completions message content")
        inspection = inspect_raw_prior_response(text)
        prior = parse_prior_response(text, config=config)
        usage = extract_usage(response)
        fallback_count = sum(source != "llm" for source in prior.sources)
        final_valid = _final_prior_valid(prior)
        base.update(
            {
                "api_success": True,
                "response_model": None if getattr(response, "model", None) is None else str(response.model),
                "finish_reason": None if getattr(response.choices[0], "finish_reason", None) is None else str(response.choices[0].finish_reason),
                "latency_ms": float(latency_ms),
                "usage": usage,
                "estimated_cost_usd": estimate_cost_usd(spec, usage),
                "response_sha256": sha256_text(text),
                "raw_inspection": inspection,
                "fallback_count": int(fallback_count),
                "final_prior_valid": bool(final_valid),
            }
        )
        base["pass"] = bool(
            inspection["schema_valid"]
            and final_valid
            and bool(usage["available"])
        )
    except Exception as exc:
        base["latency_ms"] = float((time.perf_counter() - start) * 1000.0)
        base["failure_class"] = classify_live_exception(exc)
        base["exception_type"] = type(exc).__name__
    return base


def run_preflight(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    bank_path: str | Path = DEFAULT_STATE_BANK,
    summary_path: str | Path = DEFAULT_STATE_BANK_SUMMARY,
    client_factory: Callable[[LLMModelRegistry], object] | None = None,
    require_env_key: bool = True,
) -> dict:
    registry_path = resolve_path(registry_path)
    registry = load_model_registry(registry_path)
    rows, bank_summary = load_frozen_state_bank(bank_path, summary_path)
    state_record = select_preflight_state(rows)

    if client_factory is None:
        client = _default_client_factory(registry, require_env_key=require_env_key)
    else:
        client = client_factory(registry)

    model_reports = [
        run_model_preflight(
            registry=registry,
            tier=tier,
            state_record=state_record,
            client=client,
        )
        for tier in EXPECTED_TIERS
    ]
    overall_pass = all(bool(item["pass"]) for item in model_reports)
    return {
        "status": "PASS" if overall_pass else "FAIL",
        "purpose": "three-model exact-ID capability preflight only; not model selection",
        "registry_version": int(registry.version),
        "registry_sha256": file_sha256(registry_path),
        "primary_model_selected": bool(registry.primary_model_selected),
        "network_mode": ofox_network_mode(),
        "state_bank_sha256": str(bank_summary["bank_sha256"]),
        "selection_config_sha256": str(bank_summary["selection_config_sha256"]),
        "common_state": {
            "episode_seed": int(state_record["episode_seed"]),
            "agent_name": str(state_record["agent_name"]),
            "decision_index": int(state_record["decision_index"]),
            "global_tick_start": int(state_record["global_tick_start"]),
            "feature17_any_valid_observable_target": int(state_record["feature17_any_valid_observable_target"]),
            "state_sha256": str(state_record["state_sha256"]),
        },
        "live_call_count": len(EXPECTED_TIERS),
        "cache_used": False,
        "raw_prompt_recorded": False,
        "raw_response_recorded": False,
        "api_key_value_recorded": False,
        "models": model_reports,
        "pass": bool(overall_pass),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate B2 three-model OFOX capability preflight")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--state-bank", default=DEFAULT_STATE_BANK)
    parser.add_argument("--state-bank-summary", default=DEFAULT_STATE_BANK_SUMMARY)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    report = run_preflight(
        registry_path=args.registry,
        bank_path=args.state_bank,
        summary_path=args.state_bank_summary,
        require_env_key=True,
    )
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)

    print("=" * 80)
    print("[GATE B2 MODEL CAPABILITY PREFLIGHT]")
    print("state_sha256:", report["common_state"]["state_sha256"])
    print("network_mode:", report["network_mode"])
    for item in report["models"]:
        print(
            f"{item['tier']} {item['experiment_alias']}: "
            f"model={item['requested_model']} api={item['api_success']} "
            f"schema={None if item['raw_inspection'] is None else item['raw_inspection']['schema_valid']} "
            f"usage={item['usage']['available']} fallback={item['fallback_count']} "
            f"latency_ms={item['latency_ms']} pass={item['pass']} "
            f"failure={item['failure_class']}"
        )
    print("pass:", report["pass"])
    print("[OK] report:", out)
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
