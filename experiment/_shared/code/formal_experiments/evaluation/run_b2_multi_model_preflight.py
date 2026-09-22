from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

from formal_experiments.evaluation.build_llm_validation_state_bank import (
    DEFAULT_OUT as DEFAULT_STATE_BANK,
    DEFAULT_SUMMARY as DEFAULT_STATE_BANK_SUMMARY,
    state_sha256,
)
from formal_experiments.ours.llm_prior_v2 import (
    FORMAL_ACTION_NAMES,
    build_formal_prompt,
    parse_prior_response,
)
from formal_experiments.ours.model_registry import DEFAULT_REGISTRY, LLMModelSpec, load_model_registry
from formal_experiments.ours.ofox_prior_client import (
    OFOX_API_KEY_ENV,
    OFOX_BASE_URL,
    build_openai_http_client,
    ofox_network_mode,
    response_format_json_schema,
)
from shared.formal_state import FORMAL_STATE_DIM


DEFAULT_OUT = "outputs/lwm_rl_v2/b2/live_preflight.json"
RETRY_SLEEP_SECONDS = 0.5


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def load_frozen_probe_state(
    bank_path: str | Path = DEFAULT_STATE_BANK,
    summary_path: str | Path = DEFAULT_STATE_BANK_SUMMARY,
) -> dict:
    bank_path = resolve_path(bank_path)
    summary_path = resolve_path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not bool(summary.get("pass")):
        raise RuntimeError("validation state bank summary is not PASS")
    if int(summary.get("record_count", -1)) != 240:
        raise RuntimeError("validation state bank must contain exactly 240 records")
    if int(summary.get("unique_exact_state_count", -1)) != 240:
        raise RuntimeError("validation state bank must contain 240 unique exact states")

    rows = [
        json.loads(line)
        for line in bank_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 240:
        raise RuntimeError("state-bank file count does not match frozen summary")
    rows.sort(
        key=lambda row: (
            int(row["episode_seed"]),
            str(row["agent_name"]),
            int(row["decision_index"]),
            int(row["global_tick_start"]),
        )
    )
    probe = rows[0]
    state = np.asarray(probe["state"], dtype=np.float32)
    if state.shape != (FORMAL_STATE_DIM,) or not np.isfinite(state).all():
        raise RuntimeError("probe state is not finite D27")
    if state_sha256(state) != str(probe["state_sha256"]):
        raise RuntimeError("probe state SHA256 mismatch")
    return probe


def classify_preflight_exception(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "429" in text or "rate limit" in text or "quota" in text or "too_many_requests" in text:
        return "rate_limit"
    if any(code in text for code in ("500", "502", "503", "504")):
        return "server_5xx"
    if any(term in text for term in ("ssl", "connecterror", "connection", "timeout", "proxy")):
        return "network_transient"
    if "401" in text or "403" in text or "unauthorized" in text or "forbidden" in text:
        return "auth"
    if "404" in text or ("model" in text and "not found" in text):
        return "model_not_found"
    if "400" in text or "invalid" in text or "unsupported" in text:
        return "invalid_request"
    return "api_error"


def validate_raw_prior_contract(text: str, *, k: int, h: int) -> dict:
    try:
        obj = json.loads(str(text).strip())
    except Exception:
        return {
            "json_valid": False,
            "schema_valid": False,
            "semantic_valid_candidate_count": 0,
            "duplicate_candidate_count": 0,
        }
    if not isinstance(obj, dict) or set(obj.keys()) != {"candidates"}:
        return {
            "json_valid": True,
            "schema_valid": False,
            "semantic_valid_candidate_count": 0,
            "duplicate_candidate_count": 0,
        }
    candidates = obj.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != int(k):
        return {
            "json_valid": True,
            "schema_valid": False,
            "semantic_valid_candidate_count": 0,
            "duplicate_candidate_count": 0,
        }

    valid_plans: list[tuple[str, ...]] = []
    schema_valid = True
    required_candidate_keys = {"actions", "prior_score", "reason"}
    for item in candidates:
        if not isinstance(item, dict) or set(item.keys()) != required_candidate_keys:
            schema_valid = False
            continue
        actions = item.get("actions")
        score = item.get("prior_score")
        reason = item.get("reason")
        if (
            not isinstance(actions, list)
            or len(actions) != int(h)
            or any(str(action) not in FORMAL_ACTION_NAMES for action in actions)
            or not isinstance(reason, str)
        ):
            schema_valid = False
            continue
        try:
            score_value = float(score)
        except Exception:
            schema_valid = False
            continue
        if not np.isfinite(score_value) or not (0.0 <= score_value <= 1.0):
            schema_valid = False
            continue
        valid_plans.append(tuple(str(action) for action in actions))

    duplicate_count = len(valid_plans) - len(set(valid_plans))
    return {
        "json_valid": True,
        "schema_valid": bool(schema_valid and len(valid_plans) == int(k)),
        "semantic_valid_candidate_count": int(len(valid_plans)),
        "duplicate_candidate_count": int(duplicate_count),
    }


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


def run_model_preflight(
    *,
    spec: LLMModelSpec,
    registry,
    probe: dict,
    client,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict:
    cfg = spec.to_prior_config(
        prompt_version=registry.prompt_version,
        k_candidates=registry.k_candidates,
        horizon=registry.horizon,
    )
    state = np.asarray(probe["state"], dtype=np.float32)
    messages = build_formal_prompt(state, agent_name=str(probe["agent_name"]), config=cfg)

    attempts = 0
    last_error_class = None
    last_error = None
    response = None
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
            last_error_class = classify_preflight_exception(exc)
            last_error = str(exc)
            if (
                last_error_class not in set(spec.retry_policy.retryable_classes)
                or attempts >= spec.retry_policy.max_attempts
            ):
                break
            sleep_fn(RETRY_SLEEP_SECONDS)
    latency_ms = (time.perf_counter() - started) * 1000.0

    base = {
        "tier": spec.registry_tier,
        "alias": spec.experiment_alias,
        "provider": spec.provider,
        "exact_model_id": spec.exact_model_id,
        "requested_temperature": spec.requested_temperature,
        "structured_output_requested": spec.structured_output_requested,
        "attempts": attempts,
        "retry_count": max(0, attempts - 1),
        "latency_ms": float(latency_ms),
        "probe_state_sha256": str(probe["state_sha256"]),
        "probe_seed": int(probe["episode_seed"]),
        "probe_agent": str(probe["agent_name"]),
    }

    if response is None:
        return {
            **base,
            "api_success": False,
            "failure_class": last_error_class or "api_error",
            "error": last_error,
            "json_valid": False,
            "structured_output_verified": False,
            "temperature_parameter_accepted": False,
            "semantic_valid_candidate_count": 0,
            "duplicate_candidate_count": 0,
            "final_prior_valid": False,
            "fallback_count": None,
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            "pass": False,
        }

    try:
        text = str(response.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("empty response content")
        raw = validate_raw_prior_contract(
            text,
            k=registry.k_candidates,
            h=registry.horizon,
        )
        prior = parse_prior_response(text, config=cfg)
        fallback_count = sum(source != "llm" for source in prior.sources)
        final_valid = bool(
            prior.plans.shape == (registry.k_candidates, registry.horizon)
            and np.isfinite(prior.raw_prior_scores).all()
            and np.isfinite(prior.prior_preferences).all()
            and abs(float(prior.prior_preferences.sum()) - 1.0) <= 1e-5
        )
        passed = bool(raw["schema_valid"] and final_valid)
        return {
            **base,
            "api_success": True,
            "failure_class": None if passed else "raw_contract_failure",
            "error": None if passed else "raw response did not satisfy frozen JSON/semantic contract",
            "response_model": str(getattr(response, "model", "") or ""),
            "json_valid": bool(raw["json_valid"]),
            "structured_output_verified": bool(raw["schema_valid"]),
            "temperature_parameter_accepted": True,
            "semantic_valid_candidate_count": int(raw["semantic_valid_candidate_count"]),
            "duplicate_candidate_count": int(raw["duplicate_candidate_count"]),
            "final_prior_valid": final_valid,
            "fallback_count": int(fallback_count),
            "usage": _usage_dict(response),
            "pass": passed,
        }
    except Exception as exc:
        return {
            **base,
            "api_success": True,
            "failure_class": "semantic_or_parse_failure",
            "error": str(exc),
            "json_valid": False,
            "structured_output_verified": False,
            "temperature_parameter_accepted": True,
            "semantic_valid_candidate_count": 0,
            "duplicate_candidate_count": 0,
            "final_prior_valid": False,
            "fallback_count": None,
            "usage": _usage_dict(response),
            "pass": False,
        }


def make_live_client():
    key = str(os.environ.get(OFOX_API_KEY_ENV, "")).strip()
    if not key:
        raise RuntimeError("OFOX_API_KEY is required for the B2 multi-model live preflight")
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is required: python -m pip install -U openai") from exc
    kwargs = {"base_url": OFOX_BASE_URL, "api_key": key}
    http_client = build_openai_http_client()
    if http_client is not None:
        kwargs["http_client"] = http_client
    return OpenAI(**kwargs)


def run_preflight(
    *,
    registry_path=DEFAULT_REGISTRY,
    bank_path=DEFAULT_STATE_BANK,
    summary_path=DEFAULT_STATE_BANK_SUMMARY,
    client=None,
) -> dict:
    registry = load_model_registry(registry_path)
    probe = load_frozen_probe_state(bank_path, summary_path)
    live_client = make_live_client() if client is None else client
    results = [
        run_model_preflight(
            spec=registry.get(tier),
            registry=registry,
            probe=probe,
            client=live_client,
        )
        for tier in ("tier_h", "tier_m", "tier_l")
    ]
    passed = all(item["pass"] for item in results)
    return {
        "phase": "gate_b2_multi_model_live_preflight",
        "network_mode": ofox_network_mode(),
        "base_url": OFOX_BASE_URL,
        "primary_model_selected": False,
        "probe_state_sha256": str(probe["state_sha256"]),
        "models": results,
        "pass": bool(passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="B2 three-model OFOX live capability preflight")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--bank", default=DEFAULT_STATE_BANK)
    parser.add_argument("--bank-summary", default=DEFAULT_STATE_BANK_SUMMARY)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    report = run_preflight(
        registry_path=args.registry,
        bank_path=args.bank,
        summary_path=args.bank_summary,
    )
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("[GATE B2 MULTI-MODEL LIVE PREFLIGHT]")
    print("network_mode:", report["network_mode"])
    print("probe_state_sha256:", report["probe_state_sha256"])
    for item in report["models"]:
        print(
            f"{item['tier']} {item['exact_model_id']}: "
            f"api={item['api_success']} structured={item['structured_output_verified']} "
            f"temp_accepted={item['temperature_parameter_accepted']} "
            f"raw_valid={item['semantic_valid_candidate_count']}/6 "
            f"duplicates={item['duplicate_candidate_count']} "
            f"final={item['final_prior_valid']} fallback={item['fallback_count']} "
            f"attempts={item['attempts']} usage={item['usage']} pass={item['pass']}"
        )
        if item["error"]:
            print("  error:", item["error"])
    print("primary_model_selected:", report["primary_model_selected"])
    print("pass:", report["pass"])
    print("[OK] report:", out)
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
