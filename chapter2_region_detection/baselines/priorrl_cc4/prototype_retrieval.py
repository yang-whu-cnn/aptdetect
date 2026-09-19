"""Frozen train-only, agent-local nearest-prototype retrieval for PriorRL."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from formal_experiments.ours.llm_prior_v2 import (
    FORMAL_HORIZON, FORMAL_K_CANDIDATES, PriorBatch,
)
from formal_experiments.ours.prior_cache import exact_state_float32, exact_state_sha256
from shared.action_contract import N_ACTIONS
from shared.formal_state import BLUE_AGENTS, FORMAL_STATE_DIM

FORMAT_VERSION = "priorrl_agent_local_prototypes_v2"
DISTANCE_VERSION = "weighted_standardized_l2_div_sqrt_weight_sum_v1"
TIE_BREAK = "distance_then_state_sha256_then_cache_key"
RADIUS_RULE = "validation_agent_local_nn_distance_quantile_v1"
# Fail-closed runtime semantics require every validation state to be inside the
# frozen support boundary. A lower quantile was tried in the development gate
# and rejected before any result was produced because it made ordinary
# train-seed trajectories unrecoverably miss the cache.
DEFAULT_RADIUS_QUANTILE = 1.0
DEFAULT_WEIGHTS = tuple([1.0] * FORMAL_STATE_DIM)


def _sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PrototypeInput:
    state: np.ndarray
    agent_name: str
    action_prior: np.ndarray
    cache_key: str
    cache_entry_sha256: str
    plans: np.ndarray
    raw_prior_scores: np.ndarray
    prior_preferences: np.ndarray
    sources: tuple[str, ...]


def _validate_prior(value) -> np.ndarray:
    prior = np.asarray(value, dtype=np.float32)
    if prior.shape != (4,) or not np.isfinite(prior).all() or np.any(prior <= 0):
        raise ValueError("prototype action prior must be positive finite A4")
    return prior / prior.sum()


def _validate_k6_prior(plans, raw_scores, preferences, sources) -> PriorBatch:
    plans = np.asarray(plans, dtype=np.int64)
    raw_scores = np.asarray(raw_scores, dtype=np.float32)
    preferences = np.asarray(preferences, dtype=np.float32)
    sources = tuple(str(value) for value in sources)
    if plans.shape != (FORMAL_K_CANDIDATES, FORMAL_HORIZON):
        raise ValueError("prototype prior must contain K=6,H=4 plans")
    if np.any(plans < 0) or np.any(plans >= N_ACTIONS):
        raise ValueError("prototype prior plan contains an action outside A4")
    if raw_scores.shape != (FORMAL_K_CANDIDATES,) or not np.isfinite(raw_scores).all():
        raise ValueError("prototype raw prior scores must be finite K6")
    if np.any(raw_scores < 0.0) or np.any(raw_scores > 1.0):
        raise ValueError("prototype raw prior scores must be in [0,1]")
    if preferences.shape != (FORMAL_K_CANDIDATES,) or not np.isfinite(preferences).all():
        raise ValueError("prototype prior preferences must be finite K6")
    if np.any(preferences < 0.0) or not np.isclose(float(preferences.sum()), 1.0, atol=1e-5):
        raise ValueError("prototype prior preferences must be nonnegative and sum to one")
    if len(sources) != FORMAL_K_CANDIDATES:
        raise ValueError("prototype prior sources must contain K6 values")
    return PriorBatch(plans.copy(), raw_scores.copy(), preferences.copy(), sources)


def fit_train_prototypes(records: Iterable[PrototypeInput], *, source_sha256: str,
                         weights=DEFAULT_WEIGHTS) -> dict:
    """Fit per-agent scalers using train prototypes only; radius remains unset."""
    if len(str(source_sha256)) != 64:
        raise ValueError("train source SHA256 is required")
    weight = np.asarray(weights, dtype=np.float32)
    if weight.shape != (27,) or not np.isfinite(weight).all() or np.any(weight <= 0):
        raise ValueError("distance weights must be positive finite D27")
    grouped = {agent: [] for agent in BLUE_AGENTS}
    seen = set()
    for item in records:
        if item.agent_name not in grouped:
            raise ValueError("prototype has invalid agent")
        state = exact_state_float32(item.state)
        state_sha = exact_state_sha256(state)
        key = (item.agent_name, state_sha)
        if key in seen:
            raise ValueError("duplicate agent-local prototype state")
        seen.add(key)
        if len(item.cache_key) != 64 or len(item.cache_entry_sha256) != 64:
            raise ValueError("prototype cache provenance must contain SHA256 identifiers")
        prior = _validate_k6_prior(
            item.plans, item.raw_prior_scores, item.prior_preferences, item.sources
        )
        grouped[item.agent_name].append({
            "state": state.tolist(), "state_sha256": state_sha,
            "action_prior": _validate_prior(item.action_prior).tolist(),
            "cache_key": item.cache_key, "cache_entry_sha256": item.cache_entry_sha256,
            "plans": prior.plans.tolist(),
            "raw_prior_scores": prior.raw_prior_scores.tolist(),
            "prior_preferences": prior.prior_preferences.tolist(),
            "sources": list(prior.sources),
        })
    agents = {}
    for agent, rows in grouped.items():
        if not rows:
            raise ValueError(f"train prototypes missing agent {agent}")
        rows.sort(key=lambda x: (x["state_sha256"], x["cache_key"]))
        states = np.asarray([x["state"] for x in rows], dtype=np.float32)
        mean = states.mean(0, dtype=np.float64).astype(np.float32)
        scale = states.std(0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(scale, np.float32(1e-6))
        agents[agent] = {"scaler_mean": mean.tolist(), "scaler_scale": scale.tolist(),
                         "prototype_count": len(rows), "prototypes": rows}
    payload = {
        "format_version": FORMAT_VERSION, "state_dim": 27, "n_actions": 4,
        "fit_split": "train", "radius_selection_split": None, "radius": None,
        "distance_version": DISTANCE_VERSION, "weights": weight.tolist(),
        "tie_break": TIE_BREAK, "max_radius_semantics": "distance_lte_radius",
        "radius_rule": RADIUS_RULE, "radius_quantile": DEFAULT_RADIUS_QUANTILE,
        "train_source_sha256": source_sha256, "agents": agents,
    }
    payload["prototype_sha256"] = _sha(payload)
    return payload


def _distance_rows(payload: dict, states: Iterable[tuple[str, np.ndarray]]) -> list[float]:
    weights = np.asarray(payload["weights"], dtype=np.float64)
    distances = []
    for agent, raw_state in states:
        if agent not in BLUE_AGENTS:
            raise ValueError("validation record has invalid agent")
        item = payload["agents"][agent]
        state = exact_state_float32(raw_state).astype(np.float64)
        mean, scale = np.asarray(item["scaler_mean"]), np.asarray(item["scaler_scale"])
        prototypes = np.asarray([x["state"] for x in item["prototypes"]], dtype=np.float64)
        delta = ((prototypes - state) / scale) ** 2 * weights
        distances.append(float(np.sqrt(delta.sum(1) / weights.sum()).min()))
    if not distances:
        raise ValueError("validation states are empty")
    return distances


def freeze_validation_radius(train_payload: dict, validation_states: Iterable[tuple[str, np.ndarray]],
                             *, validation_source_sha256: str,
                             quantile: float = DEFAULT_RADIUS_QUANTILE) -> dict:
    validate_payload(train_payload, require_frozen=False)
    if train_payload["radius"] is not None:
        raise ValueError("radius is already frozen")
    if not 0 < float(quantile) <= 1 or len(validation_source_sha256) != 64:
        raise ValueError("invalid validation radius provenance")
    distances = _distance_rows(train_payload, validation_states)
    payload = json.loads(json.dumps(train_payload))
    payload.pop("prototype_sha256", None)
    payload.update({
        "radius_selection_split": "validation",
        "radius": float(np.quantile(np.asarray(distances), quantile, method="higher")),
        "radius_quantile": float(quantile),
        "validation_source_sha256": validation_source_sha256,
        "validation_distance_count": len(distances),
        "validation_distance_min": min(distances),
        "validation_distance_max": max(distances),
    })
    payload["prototype_sha256"] = _sha(payload)
    validate_payload(payload, require_frozen=True)
    return payload


def validate_payload(payload: dict, *, require_frozen: bool = True) -> None:
    if payload.get("format_version") != FORMAT_VERSION or payload.get("fit_split") != "train":
        raise ValueError("prototype format/split mismatch")
    checksum = payload.get("prototype_sha256")
    material = dict(payload); material.pop("prototype_sha256", None)
    if checksum != _sha(material):
        raise ValueError("prototype SHA256 mismatch")
    if payload.get("distance_version") != DISTANCE_VERSION or payload.get("tie_break") != TIE_BREAK:
        raise ValueError("distance/tie-break contract mismatch")
    weights = np.asarray(payload.get("weights"), dtype=np.float32)
    if weights.shape != (27,) or np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("invalid frozen weights")
    if set(payload.get("agents", {})) != set(BLUE_AGENTS):
        raise ValueError("prototype artifact must contain all five agents")
    for agent in BLUE_AGENTS:
        item = payload["agents"][agent]
        mean = np.asarray(item.get("scaler_mean"), dtype=np.float32)
        scale = np.asarray(item.get("scaler_scale"), dtype=np.float32)
        rows = item.get("prototypes")
        if mean.shape != (27,) or scale.shape != (27,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError(f"{agent}: invalid frozen scaler")
        if not isinstance(rows, list) or len(rows) != int(item.get("prototype_count", -1)) or not rows:
            raise ValueError(f"{agent}: invalid prototype count")
        for row in rows:
            state = exact_state_float32(row.get("state"))
            if exact_state_sha256(state) != row.get("state_sha256"):
                raise ValueError(f"{agent}: prototype state SHA mismatch")
            _validate_prior(row.get("action_prior"))
            _validate_k6_prior(
                row.get("plans"), row.get("raw_prior_scores"),
                row.get("prior_preferences"), row.get("sources"),
            )
            for key in ("cache_key", "cache_entry_sha256"):
                value = str(row.get(key, ""))
                if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError(f"{agent}: invalid {key}")
    if require_frozen:
        if payload.get("radius_selection_split") != "validation":
            raise ValueError("radius must be selected on validation only")
        radius = payload.get("radius")
        if radius is None or not np.isfinite(float(radius)) or float(radius) < 0:
            raise ValueError("max radius is not frozen")


class PrototypeLookupMiss(RuntimeError):
    """Auditable fail-closed miss carrying only the observable query state."""

    def __init__(self, *, agent_name: str, state: np.ndarray, distance: float, radius: float):
        self.agent_name = str(agent_name)
        self.state = exact_state_float32(state).copy()
        self.state_sha256 = exact_state_sha256(self.state)
        self.distance = float(distance)
        self.radius = float(radius)
        super().__init__(
            f"nearest prototype distance {self.distance:.9g} exceeds frozen radius "
            f"{self.radius:.9g} for {self.agent_name} state_sha256={self.state_sha256} "
            "(fail closed)"
        )


class FrozenPrototypeRetriever:
    def __init__(self, payload: dict):
        validate_payload(payload, require_frozen=True)
        self.payload = json.loads(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def _winner(self, state, *, agent_name: str) -> dict:
        if agent_name not in BLUE_AGENTS:
            raise RuntimeError("unknown agent (fail closed)")
        item = self.payload["agents"].get(agent_name)
        if not item or not item.get("prototypes"):
            raise RuntimeError("agent-local prototype/cache miss (fail closed)")
        query = exact_state_float32(state).astype(np.float64)
        scale = np.asarray(item["scaler_scale"], dtype=np.float64)
        weights = np.asarray(self.payload["weights"], dtype=np.float64)
        ranked = []
        for row in item["prototypes"]:
            prototype = np.asarray(row["state"], dtype=np.float64)
            distance = float(np.sqrt(((((prototype - query) / scale) ** 2) * weights).sum() / weights.sum()))
            ranked.append((distance, row["state_sha256"], row["cache_key"], row))
        distance, _, _, winner = min(ranked, key=lambda x: (x[0], x[1], x[2]))
        radius = float(self.payload["radius"])
        if distance > radius:
            raise PrototypeLookupMiss(
                agent_name=agent_name, state=query, distance=distance, radius=radius
            )
        return winner

    def lookup(self, state, *, agent_name: str) -> torch.Tensor:
        """Return the frozen A4 marginal used by PriorRL's KL term."""
        winner = self._winner(state, agent_name=agent_name)
        return torch.as_tensor(winner["action_prior"], dtype=torch.float32)

    def lookup_prior_batch(self, state, *, agent_name: str) -> PriorBatch:
        """Return the exact cached K6/H4 teacher output of the nearest prototype."""
        winner = self._winner(state, agent_name=agent_name)
        return _validate_k6_prior(
            winner["plans"], winner["raw_prior_scores"],
            winner["prior_preferences"], winner["sources"],
        )


class FrozenPrototypePriorAdapter:
    """Read-only Table-2 adapter over a train-only frozen prototype artifact."""

    online_allowed = False

    def __init__(self, retriever: FrozenPrototypeRetriever):
        if not isinstance(retriever, FrozenPrototypeRetriever):
            raise TypeError("retriever must be FrozenPrototypeRetriever")
        self.retriever = retriever

    @classmethod
    def load_artifact(cls, path: str | Path):
        return cls(FrozenPrototypeRetriever.load(path))

    def load(self, state, *, agent_name: str, split: str) -> PriorBatch:
        if split not in ("train", "validation", "test"):
            raise ValueError("split isolation violation")
        return self.retriever.lookup_prior_batch(state, agent_name=agent_name)


def coverage_report(retriever: FrozenPrototypeRetriever,
                    splits: dict[str, Iterable[tuple[str, np.ndarray]]]) -> dict:
    report = {"format_version": FORMAT_VERSION, "prototype_sha256": retriever.payload["prototype_sha256"],
              "radius": retriever.payload["radius"], "splits": {}}
    for split, rows in splits.items():
        hit = miss = 0; per_agent = {a: [0, 0] for a in BLUE_AGENTS}
        for agent, state in rows:
            try:
                retriever.lookup(state, agent_name=agent); hit += 1; per_agent[agent][0] += 1
            except RuntimeError:
                miss += 1
            per_agent[agent][1] += 1
        report["splits"][split] = {"hits": hit, "misses": miss, "total": hit + miss,
            "coverage": hit / (hit + miss) if hit + miss else None,
            "per_agent": {a: {"hits": v[0], "total": v[1]} for a, v in per_agent.items()}}
    return report
