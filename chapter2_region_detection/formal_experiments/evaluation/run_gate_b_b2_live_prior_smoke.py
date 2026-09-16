from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from formal_experiments.evaluation.calibrate_ug_normalizer import (
    DEFAULT_STATES,
    load_calibration_records,
    resolve_project_path,
)
from formal_experiments.ours.gemini_prior_client import (
    GeminiPriorLiveClient,
    api_key_source,
    classify_live_exception,
    gemini_network_mode,
    masked_detected_proxies,
)
from formal_experiments.ours.llm_prior_v2 import (
    FORMAL_ACTION_NAMES,
    FORMAL_HORIZON,
    FORMAL_K_CANDIDATES,
)


DEFAULT_OUT = "outputs/lwm_rl_v2/gate_b/b2_live_prior_report.json"
SMOKE_AGENT = "blue_agent_0"
SMOKE_SEED = 3000


def select_smoke_state(
    records: list[dict],
    *,
    agent_name: str = SMOKE_AGENT,
    episode_seed: int = SMOKE_SEED,
) -> np.ndarray:
    candidates = [
        item
        for item in records
        if item["warmup"].agent_name == agent_name
        and int(item["warmup"].episode_seed) == int(episode_seed)
    ]
    if not candidates:
        raise ValueError(
            f"no planner-visible state for agent={agent_name} seed={episode_seed}"
        )
    candidates.sort(
        key=lambda item: (item["global_tick_start"], item["decision_index"])
    )
    return np.asarray(candidates[0]["warmup"].state, dtype=np.float32).copy()


def build_report(result, *, agent_name: str, episode_seed: int) -> dict:
    prior = result.prior
    plans = np.asarray(prior.plans, dtype=np.int64)
    scores = np.asarray(prior.raw_prior_scores, dtype=np.float32)
    prefs = np.asarray(prior.prior_preferences, dtype=np.float32)

    valid_shape = plans.shape == (FORMAL_K_CANDIDATES, FORMAL_HORIZON)
    valid_actions = bool(
        valid_shape
        and np.all(plans >= 0)
        and np.all(plans < len(FORMAL_ACTION_NAMES))
    )
    unique_count = len({tuple(int(x) for x in row) for row in plans.tolist()})
    fallback_count = sum(source != "llm" for source in prior.sources)

    passed = bool(
        result.api_call_succeeded
        and valid_shape
        and valid_actions
        and np.isfinite(scores).all()
        and np.isfinite(prefs).all()
        and abs(float(prefs.sum()) - 1.0) <= 1e-5
    )

    return {
        "status": "PASS" if passed else "FAIL",
        "api_call_succeeded": bool(result.api_call_succeeded),
        "key_source": str(result.key_source),
        "network_mode": str(result.network_mode),
        "api_key_value_recorded": False,
        "model": str(result.model),
        "temperature": float(result.temperature),
        "agent_name": str(agent_name),
        "episode_seed": int(episode_seed),
        "state_source": "planner-visible calibration state-only artifact",
        "k_candidates": FORMAL_K_CANDIDATES,
        "horizon": FORMAL_HORIZON,
        "action_names": list(FORMAL_ACTION_NAMES),
        "plans": plans.tolist(),
        "raw_prior_scores": scores.tolist(),
        "prior_preferences": prefs.tolist(),
        "sources": list(prior.sources),
        "unique_plan_count": int(unique_count),
        "fallback_count": int(fallback_count),
        "prompt_sha256": str(result.prompt_sha256),
        "response_sha256": str(result.response_sha256),
        "raw_prompt_recorded": False,
        "raw_response_recorded": False,
        "pass": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate-B B2-live Gemini prior connectivity/contract smoke"
    )
    parser.add_argument("--states", default=DEFAULT_STATES)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--agent", default=SMOKE_AGENT)
    parser.add_argument("--seed", type=int, default=SMOKE_SEED)
    args = parser.parse_args()

    source = api_key_source()
    if source is None:
        raise RuntimeError(
            "No Gemini API key environment variable is configured. "
            "Set GEMINI_API_KEY or GOOGLE_API_KEY before B2-live smoke."
        )

    records = load_calibration_records(args.states)
    state = select_smoke_state(
        records,
        agent_name=str(args.agent),
        episode_seed=int(args.seed),
    )

    client = GeminiPriorLiveClient(require_env_key=True)
    try:
        result = client.generate(state, agent_name=str(args.agent))
    except Exception as exc:
        mode = gemini_network_mode()
        failure_class = classify_live_exception(exc)
        proxies = masked_detected_proxies()

        if failure_class == "quota_blocked":
            guidance = (
                "The request reached Gemini but this project/model has no usable "
                "quota. For the frozen gemini-3.1-pro-preview contract, enable "
                "billing/paid-tier quota for the API key's project."
            )
        elif failure_class == "auth_blocked":
            guidance = (
                "Check that the configured API key belongs to the intended "
                "project and has Gemini API access."
            )
        elif failure_class == "network_blocked":
            guidance = (
                "If this is an SSL/proxy error, use "
                "GEMINI_DISABLE_ENV_PROXY=1 for direct access or a valid "
                "GEMINI_PROXY_URL for a Gemini-only proxy."
            )
        else:
            guidance = "Inspect the underlying Gemini API error."

        raise RuntimeError(
            "B2-live Gemini request did not produce a valid prior. "
            f"failure_class={failure_class}; network_mode={mode}; "
            f"detected_proxies={proxies}. {guidance} "
            "Do not put proxy credentials or API keys in source."
        ) from exc
    report = build_report(
        result,
        agent_name=str(args.agent),
        episode_seed=int(args.seed),
    )

    out = resolve_project_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[GATE B B2-LIVE SUMMARY]")
    print("api_call_succeeded:", report["api_call_succeeded"])
    print("key_source:", report["key_source"])
    print("network_mode:", report["network_mode"])
    print("model:", report["model"])
    print("temperature:", report["temperature"])
    print("plans_shape:", [FORMAL_K_CANDIDATES, FORMAL_HORIZON])
    print("unique_plan_count:", report["unique_plan_count"])
    print("fallback_count:", report["fallback_count"])
    print("prior_sum:", float(sum(report["prior_preferences"])))
    print("pass:", report["pass"])
    print("[OK] report:", out)

    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
