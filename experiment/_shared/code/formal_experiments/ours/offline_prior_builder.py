"""
Offline LLM prior dataset builder.

Purpose:
    Generate final LLM prior cache entries offline before PPO experiments.

Design:
    state bank -> LLM teacher -> PriorCache

The builder intentionally separates expensive LLM generation from policy
optimization. Formal PPO runs should consume the generated cache and avoid
online provider calls.

This module does not access hidden compromise labels or future rewards.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import time
from typing import Iterable, Callable

import numpy as np

from formal_experiments.ours.prior_cache import PriorCache, make_identity


@dataclass(frozen=True)
class OfflinePriorBuildConfig:
    split: str
    model_alias: str
    provider: str
    exact_model_id: str
    prompt_version: str
    target_size: int
    output_root: str


@dataclass(frozen=True)
class OfflinePriorBuildRecord:
    index: int
    cache_key: str
    state_sha256: str
    success: bool
    latency_ms: float


def _sha256_array(state: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(state, dtype="<f4").tobytes(order="C")
    ).hexdigest()


def build_offline_prior_cache(
    *,
    states: Iterable[np.ndarray],
    config: OfflinePriorBuildConfig,
    generate_prior: Callable[[np.ndarray], tuple[object, dict]],
    registry_version: int,
    registry_sha256: str,
) -> list[OfflinePriorBuildRecord]:
    """Build immutable prior cache entries.

    generate_prior must return:
        (PriorBatch, metadata)

    The provider-specific implementation is intentionally injected so the
    builder can be tested without paid API calls.
    """

    cache = PriorCache(config.output_root)
    records: list[OfflinePriorBuildRecord] = []

    for index, state in enumerate(states):
        if index >= int(config.target_size):
            break

        start = time.perf_counter()
        state = np.asarray(state, dtype="<f4")

        prior, metadata = generate_prior(state)

        identity = make_identity(
            split=config.split,
            model_alias=config.model_alias,
            provider=config.provider,
            exact_model_id=config.exact_model_id,
            registry_version=registry_version,
            registry_sha256=registry_sha256,
            prompt_version=config.prompt_version,
            generation_config={"mode": "offline_teacher"},
            state=state,
        )

        cache_path = cache.path_for(identity)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "format_version": 2,
            "cache_key": identity.cache_key,
            "identity": identity.canonical_payload(),
            "prior": {
                "plans": np.asarray(prior.plans).tolist(),
                "raw_prior_scores": np.asarray(prior.raw_prior_scores).tolist(),
                "prior_preferences": np.asarray(prior.prior_preferences).tolist(),
                "sources": list(prior.sources),
            },
            "metadata": dict(metadata),
        }

        cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        records.append(
            OfflinePriorBuildRecord(
                index=index,
                cache_key=identity.cache_key,
                state_sha256=_sha256_array(state),
                success=True,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        )

    return records


if __name__ == "__main__":
    raise SystemExit(
        "Use build_offline_prior_cache() with a project-specific GPT-5.6 Sol provider adapter."
    )
