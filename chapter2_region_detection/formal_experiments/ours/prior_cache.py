from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Callable, Mapping

import numpy as np

from formal_experiments.ours.llm_prior_v2 import (
    FORMAL_HORIZON,
    FORMAL_K_CANDIDATES,
    PriorBatch,
)
from shared.formal_state import FORMAL_STATE_DIM


CACHE_FORMAT_VERSION = 1
ALLOWED_SPLITS = {"train", "validation", "test"}
SENSITIVE_KEY_FRAGMENTS = ("api_key", "apikey", "authorization", "secret")


class PriorCacheError(RuntimeError):
    pass


class PriorCacheCorruptError(PriorCacheError):
    pass


def _safe_component(value: str, name: str) -> str:
    text = str(value).strip()
    if not text or not re.fullmatch(r"[A-Za-z0-9_.-]+", text):
        raise ValueError(f"{name} contains unsafe path characters: {value!r}")
    return text


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def exact_state_float32(state) -> np.ndarray:
    array = np.asarray(state, dtype="<f4")
    if array.shape != (FORMAL_STATE_DIM,):
        raise ValueError(f"state must have shape ({FORMAL_STATE_DIM},)")
    if not np.isfinite(array).all():
        raise ValueError("state must be finite")
    return np.ascontiguousarray(array)


def exact_state_sha256(state) -> str:
    return hashlib.sha256(exact_state_float32(state).tobytes(order="C")).hexdigest()


def _assert_no_sensitive_keys(value, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if any(fragment in key_text for fragment in SENSITIVE_KEY_FRAGMENTS):
                raise ValueError(f"sensitive field forbidden in cache metadata: {path}.{key}")
            _assert_no_sensitive_keys(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_sensitive_keys(item, f"{path}[{index}]")


@dataclass(frozen=True)
class PriorCacheIdentity:
    split: str
    model_alias: str
    provider: str
    exact_model_id: str
    prompt_version: str
    k_candidates: int
    horizon: int
    generation_config: Mapping[str, object]
    state_sha256: str

    def __post_init__(self) -> None:
        if self.split not in ALLOWED_SPLITS:
            raise ValueError(f"cache split must be one of {sorted(ALLOWED_SPLITS)}")
        _safe_component(self.model_alias, "model_alias")
        _safe_component(self.prompt_version, "prompt_version")
        if not str(self.provider).strip() or not str(self.exact_model_id).strip():
            raise ValueError("provider/model ID must be nonempty")
        if int(self.k_candidates) != FORMAL_K_CANDIDATES:
            raise ValueError("cache K violates formal contract")
        if int(self.horizon) != FORMAL_HORIZON:
            raise ValueError("cache H violates formal contract")
        if not re.fullmatch(r"[0-9a-f]{64}", str(self.state_sha256)):
            raise ValueError("state_sha256 must be lowercase SHA256 hex")
        _assert_no_sensitive_keys(self.generation_config, "generation_config")

    def canonical_payload(self) -> dict:
        return {
            "split": self.split,
            "model_alias": self.model_alias,
            "provider": self.provider,
            "exact_model_id": self.exact_model_id,
            "prompt_version": self.prompt_version,
            "k_candidates": int(self.k_candidates),
            "horizon": int(self.horizon),
            "generation_config": dict(self.generation_config),
            "state_sha256": self.state_sha256,
        }

    @property
    def cache_key(self) -> str:
        return hashlib.sha256(_canonical_json(self.canonical_payload()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedPrior:
    identity: PriorCacheIdentity
    prior: PriorBatch
    metadata: Mapping[str, object]
    cache_key: str
    path: Path


@dataclass(frozen=True)
class CacheLookupResult:
    cached_prior: CachedPrior
    cache_hit: bool


def make_identity(
    *,
    split: str,
    model_alias: str,
    provider: str,
    exact_model_id: str,
    prompt_version: str,
    generation_config: Mapping[str, object],
    state,
    k_candidates: int = FORMAL_K_CANDIDATES,
    horizon: int = FORMAL_HORIZON,
) -> PriorCacheIdentity:
    return PriorCacheIdentity(
        split=str(split),
        model_alias=str(model_alias),
        provider=str(provider),
        exact_model_id=str(exact_model_id),
        prompt_version=str(prompt_version),
        k_candidates=int(k_candidates),
        horizon=int(horizon),
        generation_config=dict(generation_config),
        state_sha256=exact_state_sha256(state),
    )


def _validate_prior(prior: PriorBatch) -> PriorBatch:
    if not isinstance(prior, PriorBatch):
        raise TypeError("prior must be PriorBatch")
    plans = np.asarray(prior.plans, dtype=np.int64)
    scores = np.asarray(prior.raw_prior_scores, dtype=np.float32)
    prefs = np.asarray(prior.prior_preferences, dtype=np.float32)
    sources = tuple(str(x) for x in prior.sources)
    if plans.shape != (FORMAL_K_CANDIDATES, FORMAL_HORIZON):
        raise ValueError("cache prior plans shape mismatch")
    if scores.shape != (FORMAL_K_CANDIDATES,) or prefs.shape != (FORMAL_K_CANDIDATES,):
        raise ValueError("cache prior score shape mismatch")
    if len(sources) != FORMAL_K_CANDIDATES:
        raise ValueError("cache prior source length mismatch")
    if not np.isfinite(scores).all() or not np.isfinite(prefs).all():
        raise ValueError("cache prior scores/preferences must be finite")
    if np.any(plans < 0) or np.any(plans >= 4):
        raise ValueError("cache prior contains invalid action IDs")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("cache raw prior scores out of range")
    if np.any(prefs < 0.0) or abs(float(prefs.sum()) - 1.0) > 1e-5:
        raise ValueError("cache prior preferences must be nonnegative and sum to 1")
    return PriorBatch(
        plans=plans.copy(),
        raw_prior_scores=scores.copy(),
        prior_preferences=prefs.copy(),
        sources=sources,
    )


class PriorCache:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def path_for(self, identity: PriorCacheIdentity) -> Path:
        alias = _safe_component(identity.model_alias, "model_alias")
        prompt = _safe_component(identity.prompt_version, "prompt_version")
        return self.root / identity.split / alias / prompt / f"{identity.cache_key}.json"

    def load(self, identity: PriorCacheIdentity) -> CachedPrior | None:
        path = self.path_for(identity)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            raise PriorCacheCorruptError(f"failed to parse cache file: {path}") from exc

        try:
            if int(payload["format_version"]) != CACHE_FORMAT_VERSION:
                raise ValueError("unsupported cache format version")
            if str(payload["cache_key"]) != identity.cache_key:
                raise ValueError("cache key mismatch")
            if payload["identity"] != identity.canonical_payload():
                raise ValueError("cache identity mismatch")
            prior_raw = payload["prior"]
            prior = _validate_prior(
                PriorBatch(
                    plans=np.asarray(prior_raw["plans"], dtype=np.int64),
                    raw_prior_scores=np.asarray(prior_raw["raw_prior_scores"], dtype=np.float32),
                    prior_preferences=np.asarray(prior_raw["prior_preferences"], dtype=np.float32),
                    sources=tuple(prior_raw["sources"]),
                )
            )
            metadata = payload.get("metadata", {})
            _assert_no_sensitive_keys(metadata, "metadata")
        except Exception as exc:
            raise PriorCacheCorruptError(f"cache integrity failure: {path}") from exc

        return CachedPrior(
            identity=identity,
            prior=prior,
            metadata=dict(metadata),
            cache_key=identity.cache_key,
            path=path,
        )

    def store(
        self,
        identity: PriorCacheIdentity,
        prior: PriorBatch,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> CachedPrior:
        prior = _validate_prior(prior)
        metadata = {} if metadata is None else dict(metadata)
        _assert_no_sensitive_keys(metadata, "metadata")
        path = self.path_for(identity)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "format_version": CACHE_FORMAT_VERSION,
            "cache_key": identity.cache_key,
            "identity": identity.canonical_payload(),
            "prior": {
                "plans": np.asarray(prior.plans, dtype=np.int64).tolist(),
                "raw_prior_scores": np.asarray(prior.raw_prior_scores, dtype=np.float32).tolist(),
                "prior_preferences": np.asarray(prior.prior_preferences, dtype=np.float32).tolist(),
                "sources": list(prior.sources),
            },
            "metadata": metadata,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{identity.cache_key}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())

        try:
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

        cached = self.load(identity)
        if cached is None:
            raise PriorCacheError("cache write completed but entry cannot be reloaded")
        return cached

    def get_or_generate(
        self,
        identity: PriorCacheIdentity,
        generator: Callable[[], tuple[PriorBatch, Mapping[str, object]]],
    ) -> CacheLookupResult:
        existing = self.load(identity)
        if existing is not None:
            return CacheLookupResult(cached_prior=existing, cache_hit=True)

        generated = generator()
        if not isinstance(generated, tuple) or len(generated) != 2:
            raise TypeError("generator must return (PriorBatch, metadata)")
        prior, metadata = generated
        stored = self.store(identity, prior, metadata=metadata)
        return CacheLookupResult(cached_prior=stored, cache_hit=False)
