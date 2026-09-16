from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from math import isfinite
import re
from typing import Iterable

import numpy as np

from shared.action_contract import ACTION_CONTRACTS, N_ACTIONS
from shared.formal_state import FORMAL_STATE_DIM, FORMAL_STATE_FEATURE_NAMES


FORMAL_ACTION_NAMES = tuple(item.name for item in ACTION_CONTRACTS)
FORMAL_K_CANDIDATES = 6
FORMAL_HORIZON = 4
FORMAL_LLM_PROVIDER = "google_gemini"
FORMAL_LLM_MODEL_LABEL = "Gemini-3.1"
FORMAL_LLM_API_MODEL = "gemini-3.1-pro-preview"
FORMAL_LLM_TEMPERATURE = 0.2
FORMAL_PROMPT_VERSION = "lwm_rl_gate_b_v2_1"
FALLBACK_SEED = 20260916


@dataclass(frozen=True)
class FormalLLMPriorConfig:
    k_candidates: int = FORMAL_K_CANDIDATES
    horizon: int = FORMAL_HORIZON
    provider: str = FORMAL_LLM_PROVIDER
    model_label: str = FORMAL_LLM_MODEL_LABEL
    api_model: str = FORMAL_LLM_API_MODEL
    temperature: float = FORMAL_LLM_TEMPERATURE
    prompt_version: str = FORMAL_PROMPT_VERSION
    fallback_seed: int = FALLBACK_SEED

    def __post_init__(self) -> None:
        if int(self.k_candidates) != FORMAL_K_CANDIDATES:
            raise ValueError("formal Gate-B K must be 6")
        if int(self.horizon) != FORMAL_HORIZON:
            raise ValueError("formal Gate-B horizon must be 4")
        if not isfinite(float(self.temperature)) or float(self.temperature) < 0.0:
            raise ValueError("temperature must be finite and >=0")


@dataclass(frozen=True)
class CandidatePlan:
    actions: tuple[int, ...]
    raw_prior_score: float
    source: str


@dataclass(frozen=True)
class PriorBatch:
    plans: np.ndarray
    raw_prior_scores: np.ndarray
    prior_preferences: np.ndarray
    sources: tuple[str, ...]


def _strip_fences(text: str) -> str:
    s = str(text).strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"```$", "", s).strip()
    return s


def _plan_name_tuple_to_ids(actions: Iterable[str]) -> tuple[int, ...] | None:
    names = tuple(str(x).strip() for x in actions)
    if len(names) != FORMAL_HORIZON:
        return None
    if any(name not in FORMAL_ACTION_NAMES for name in names):
        return None
    mapping = {name: i for i, name in enumerate(FORMAL_ACTION_NAMES)}
    return tuple(mapping[name] for name in names)


def deterministic_fallback_plans(
    *,
    existing: Iterable[tuple[int, ...]] = (),
    needed: int,
    seed: int = FALLBACK_SEED,
) -> list[tuple[int, ...]]:
    if needed < 0:
        raise ValueError("needed must be >=0")
    existing_set = {tuple(int(x) for x in plan) for plan in existing}
    all_plans = list(itertools.product(range(N_ACTIONS), repeat=FORMAL_HORIZON))
    candidates = [plan for plan in all_plans if plan not in existing_set]
    if needed > len(candidates):
        raise ValueError("not enough remaining formal plans")
    rng = np.random.RandomState(int(seed))
    order = rng.permutation(len(candidates))
    return [candidates[int(i)] for i in order[:needed]]


def normalize_prior_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape != (FORMAL_K_CANDIDATES,):
        raise ValueError("prior score shape mismatch")
    if not np.isfinite(scores).all():
        raise ValueError("prior scores must be finite")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("raw prior scores must lie in [0,1]")
    total = float(scores.sum())
    if total <= 0.0:
        return np.full(FORMAL_K_CANDIDATES, 1.0 / FORMAL_K_CANDIDATES, dtype=np.float32)
    return (scores / total).astype(np.float32)


def parse_prior_response(
    text: str,
    *,
    config: FormalLLMPriorConfig | None = None,
) -> PriorBatch:
    cfg = config if config is not None else FormalLLMPriorConfig()
    obj = None
    try:
        obj = json.loads(_strip_fences(text))
    except Exception:
        obj = None

    by_plan: dict[tuple[int, ...], CandidatePlan] = {}
    if isinstance(obj, dict):
        raw_candidates = obj.get("candidates", [])
        if isinstance(raw_candidates, list):
            for item in raw_candidates:
                if not isinstance(item, dict):
                    continue
                actions = item.get("actions")
                if not isinstance(actions, list):
                    continue
                plan = _plan_name_tuple_to_ids(actions)
                if plan is None:
                    continue
                score = item.get("prior_score")
                try:
                    score = float(score)
                except Exception:
                    continue
                if not isfinite(score) or not (0.0 <= score <= 1.0):
                    continue
                previous = by_plan.get(plan)
                if previous is None or score > previous.raw_prior_score:
                    by_plan[plan] = CandidatePlan(plan, score, "llm")

    valid = sorted(
        by_plan.values(),
        key=lambda item: (-item.raw_prior_score, item.actions),
    )[: cfg.k_candidates]

    if len(valid) < cfg.k_candidates:
        fallback = deterministic_fallback_plans(
            existing=(item.actions for item in valid),
            needed=cfg.k_candidates - len(valid),
            seed=cfg.fallback_seed,
        )
        valid.extend(CandidatePlan(plan, 0.0, "fallback_uniform") for plan in fallback)

    plans = np.asarray([item.actions for item in valid], dtype=np.int64)
    scores = np.asarray([item.raw_prior_score for item in valid], dtype=np.float32)
    prefs = normalize_prior_scores(scores)
    sources = tuple(item.source for item in valid)

    if plans.shape != (FORMAL_K_CANDIDATES, FORMAL_HORIZON):
        raise RuntimeError("formal prior output shape mismatch")
    if np.any(plans < 0) or np.any(plans >= N_ACTIONS):
        raise RuntimeError("formal prior emitted invalid action ID")

    return PriorBatch(plans=plans, raw_prior_scores=scores, prior_preferences=prefs, sources=sources)


def build_formal_prompt(
    state,
    *,
    agent_name: str | None = None,
    config: FormalLLMPriorConfig | None = None,
) -> list[dict[str, str]]:
    cfg = config if config is not None else FormalLLMPriorConfig()
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (FORMAL_STATE_DIM,):
        raise ValueError("formal state must have shape (27,)")
    if not np.isfinite(state).all():
        raise ValueError("formal state must be finite")

    state_named = {
        str(name): float(state[i])
        for i, name in enumerate(FORMAL_STATE_FEATURE_NAMES)
    }

    action_desc = {
        "no_op": "Take no explicit response; runtime maps this to CC4 Sleep.",
        "analyse": "Collect host evidence using CC4 Analyse; target resolved downstream.",
        "remove": "Remove malicious artefacts/sessions on a resolved host target.",
        "restore": "Restore a resolved host; longest response action.",
    }

    system = (
        "You are the candidate-plan prior for a cyber incident-response planner. "
        "Use only the supplied planner-visible state. Generate candidate high-level "
        "response plans, not host targets. Return strict JSON only."
    )
    user = {
        "task": "generate_candidate_response_plans",
        "prompt_version": cfg.prompt_version,
        "agent_name": None if agent_name is None else str(agent_name),
        "k_candidates": cfg.k_candidates,
        "plan_horizon": cfg.horizon,
        "available_actions": list(FORMAL_ACTION_NAMES),
        "action_descriptions": action_desc,
        "formal_state": state_named,
        "output_schema": {
            "candidates": [
                {
                    "actions": ["no_op", "analyse", "remove", "restore"],
                    "prior_score": 0.0,
                    "reason": "short planner-visible rationale",
                }
            ]
        },
        "constraints": [
            "Return exactly JSON with key candidates.",
            "Return exactly 6 candidate objects before parser fallback.",
            "Each actions list must have exactly 4 entries.",
            "Use only no_op, analyse, remove, restore.",
            "prior_score must be a number in [0,1].",
            "Do not choose or invent host targets.",
            "Do not use hidden compromise truth, future rewards, attack labels, or test information.",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, sort_keys=True)},
    ]
