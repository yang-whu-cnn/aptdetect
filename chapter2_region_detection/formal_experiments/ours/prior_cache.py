from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Callable, Mapping

import numpy as np

from formal_experiments.ours.llm_prior_v2 import (
    FORMAL_HORIZON,
    FORMAL_K_CANDIDATES,
    PriorBatch,
)
from shared.action_contract import N_ACTIONS
from shared.formal_state import FORMAL_STATE_DIM


CACHE_FORMAT_VERSION = 1
STATE_DTYPE = "float32_le"
STATE_SHAPE = (FORMAL_STATE_DIM,)
ALLOWED_SPLITS = {"train", "validation", "test"}
SENSITIVE_KEY_FRAGMENTS = ("api_key", "apikey", "authorization", "secret")
REQUIRED_METADATA_KEYS = {
    "api_success",
    "fallback_count",
    "prompt_sha256",
    "response_sha256",
    "latency_ms",
    "retry_count",
}


class PriorCacheError(RuntimeError):
    pass


class PriorCacheCorruptError(PriorCacheError):
    pass


class PriorCacheFormatError(PriorCacheError):
    pass


def _safe_component(value: str, name: str) -> str:
    text = str(value).strip()
    if not text or not re.fullmatch(r"[A-Za-z0-9_.-]+", text):
        raise ValueError(f"{name} contains unsafe path characters: {value!r}")
    return text


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _valid_sha256(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value)))


def exact_state_float32(state) -> np.ndarray:
    array = np.asarray(state, dtype="<f4")
    if array.shape != STATE_SHAPE:
        raise ValueError(f"state must have shape {STATE_SHAPE}")
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


def _finite_nonnegative(value, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and >=0")
    return number


@dataclass(frozen=True)
class PriorCacheIdentity:
    split: str
    model_alias: str
    provider: str
    exact_model_id: str
    registry_version: int
    registry_sha256: str
    prompt_version: str
    n_actions: int
    k_candidates: int
    horizon: int
    generation_config: Mapping[str, object]
    state_dtype: str
    state_shape: tuple[int, ...]
    state_sha256: str

    def __post_init__(self) -> None:
        if self.split not in ALLOWED_SPLITS:
            raise ValueError(f"cache split must be one of {sorted(ALLOWED_SPLITS)}")
        _safe_component(self.model_alias, "model_alias")
        _safe_component(self.prompt_version, "prompt_version")
        if not str(self.provider).strip() or not str(self.exact_model_id).strip():
            raise ValueError("provider/model ID must be nonempty")
        if int(self.registry_version) <= 0:
            raise ValueError("registry_version must be positive")
        if not _valid_sha256(self.registry_sha256):
            raise ValueError("registry_sha256 must be lowercase SHA256 hex")
        if int(self.n_actions) != N_ACTIONS:
            raise ValueError("cache A violates formal contract")
        if int(self.k_candidates) != FORMAL_K_CANDIDATES:
            raise ValueError("cache K violates formal contract")
        if int(self.horizon) != FORMAL_HORIZON:
            raise ValueError("cache H violates formal contract")
        if self.state_dtype != STATE_DTYPE or tuple(self.state_shape) != STATE_SHAPE:
            raise ValueError("cache state dtype/shape violates D27 float32 contract")
        if not _valid_sha256(self.state_sha256):
            raise ValueError("state_sha256 must be lowercase SHA256 hex")
        _assert_no_sensitive_keys(self.generation_config, "generation_config")

    @property
    def generation_config_sha256(self) -> str:
        return _sha256_json(dict(self.generation_config))

    def canonical_payload(self) -> dict:
        return {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "split": self.split,
            "model_alias": self.model_alias,
            "provider": self.provider,
            "exact_model_id": self.exact_model_id,
            "registry_version": int(self.registry_version),
            "registry_sha256": self.registry_sha256,
            "prompt_version": self.prompt_version,
            "n_actions": int(self.n_actions),
            "k_candidates": int(self.k_candidates),
            "horizon": int(self.horizon),
            "generation_config": dict(self.generation_config),
            "generation_config_sha256": self.generation_config_sha256,
            "state_dtype": self.state_dtype,
            "state_shape": list(self.state_shape),
            "state_sha256": self.state_sha256,
        }

    @property
    def cache_key(self) -> str:
        return _sha256_json(self.canonical_payload())


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
    quarantined_corrupt_entry: bool = False


def make_identity(
    *,
    split: str,
    model_alias: str,
    provider: str,
    exact_model_id: str,
    registry_version: int,
    registry_sha256: str,
    prompt_version: str,
    generation_config: Mapping[str, object],
    state,
    n_actions: int = N_ACTIONS,
    k_candidates: int = FORMAL_K_CANDIDATES,
    horizon: int = FORMAL_HORIZON,
) -> PriorCacheIdentity:
    return PriorCacheIdentity(
        split=str(split),
        model_alias=str(model_alias),
        provider=str(provider),
        exact_model_id=str(exact_model_id),
        registry_version=int(registry_version),
        registry_sha256=str(registry_sha256),
        prompt_version=str(prompt_version),
        n_actions=int(n_actions),
        k_candidates=int(k_candidates),
        horizon=int(horizon),
        generation_config=dict(generation_config),
        state_dtype=STATE_DTYPE,
        state_shape=STATE_SHAPE,
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
    if np.any(plans < 0) or np.any(plans >= N_ACTIONS):
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


def _validate_metadata(metadata: Mapping[str, object], prior: PriorBatch) -> dict:
    data = dict(metadata)
    _assert_no_sensitive_keys(data, "metadata")
    missing = REQUIRED_METADATA_KEYS - set(data)
    if missing:
        raise ValueError(f"cache metadata missing required keys: {sorted(missing)}")
    if not isinstance(data["api_success"], bool):
        raise ValueError("metadata.api_success must be bool")
    if data["api_success"] is not True:
        raise ValueError("formal live cache requires api_success=True")
    fallback_count = int(data["fallback_count"])
    actual_fallback = sum(source != "llm" for source in prior.sources)
    if fallback_count != actual_fallback:
        raise ValueError("metadata fallback_count does not match PriorBatch sources")
    for key in ("prompt_sha256", "response_sha256"):
        if not _valid_sha256(str(data[key])):
            raise ValueError(f"metadata.{key} must be SHA256")
    data["latency_ms"] = _finite_nonnegative(data["latency_ms"], "metadata.latency_ms")
    retry_count = int(data["retry_count"])
    if retry_count < 0:
        raise ValueError("metadata.retry_count must be >=0")
    data["retry_count"] = retry_count
    if "cost_usd" in data and data["cost_usd"] is not None:
        data["cost_usd"] = _finite_nonnegative(data["cost_usd"], "metadata.cost_usd")
    return data


class _FileLock:
    def __init__(self, path: Path, timeout_seconds: float = 10.0):
        self.path = path
        self.timeout_seconds = float(timeout_seconds)
        self.fd: int | None = None

    def __enter__(self):
        deadline = time.monotonic() + self.timeout_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, f"pid={os.getpid()}\n".encode("utf-8"))
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise PriorCacheError(f"timed out waiting for cache lock: {self.path}")
                time.sleep(0.05)

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


class PriorCache:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def path_for(self, identity: PriorCacheIdentity) -> Path:
        alias = _safe_component(identity.model_alias, "model_alias")
        prompt = _safe_component(identity.prompt_version, "prompt_version")
        return (
            self.root
            / identity.split
            / alias
            / f"registry_v{identity.registry_version}"
            / prompt
            / f"{identity.cache_key}.json"
        )

    def lock_path_for(self, identity: PriorCacheIdentity) -> Path:
        return self.path_for(identity).with_suffix(".lock")

    def _quarantine(self, path: Path) -> Path:
        stamp = f"{time.time_ns()}"
        quarantine = path.with_name(path.name + f".corrupt.{stamp}")
        os.replace(path, quarantine)
        return quarantine

    def _decode_payload(self, path: Path, identity: PriorCacheIdentity, payload: dict) -> CachedPrior:
        if int(payload.get("format_version", -1)) != CACHE_FORMAT_VERSION:
            raise PriorCacheFormatError(
                f"incompatible cache format version at {path}: {payload.get('format_version')}"
            )
        try:
            if str(payload["cache_key"]) != identity.cache_key:
                raise ValueError("cache key mismatch")
            if payload["identity"] != identity.canonical_payload():
                raise ValueError("cache identity mismatch")
            expected_checksum = str(payload["entry_sha256"])
            checksum_material = {key: value for key, value in payload.items() if key != "entry_sha256"}
            if _sha256_json(checksum_material) != expected_checksum:
                raise ValueError("entry checksum mismatch")
            prior_raw = payload["prior"]
            prior = _validate_prior(
                PriorBatch(
                    plans=np.asarray(prior_raw["plans"], dtype=np.int64),
                    raw_prior_scores=np.asarray(prior_raw["raw_prior_scores"], dtype=np.float32),
                    prior_preferences=np.asarray(prior_raw["prior_preferences"], dtype=np.float32),
                    sources=tuple(prior_raw["sources"]),
                )
            )
            metadata = _validate_metadata(payload["metadata"], prior)
        except PriorCacheFormatError:
            raise
        except Exception as exc:
            raise PriorCacheCorruptError(f"cache integrity failure: {path}") from exc

        return CachedPrior(
            identity=identity,
            prior=prior,
            metadata=metadata,
            cache_key=identity.cache_key,
            path=path,
        )

    def load(self, identity: PriorCacheIdentity) -> CachedPrior | None:
        path = self.path_for(identity)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            quarantine = self._quarantine(path)
            raise PriorCacheCorruptError(
                f"failed to parse cache file; quarantined to {quarantine}"
            ) from exc

        try:
            return self._decode_payload(path, identity, payload)
        except PriorCacheFormatError:
            raise
        except PriorCacheCorruptError as exc:
            quarantine = self._quarantine(path)
            raise PriorCacheCorruptError(
                f"cache integrity failure; quarantined to {quarantine}"
            ) from exc

    def _store_unlocked(
        self,
        identity: PriorCacheIdentity,
        prior: PriorBatch,
        metadata: Mapping[str, object],
    ) -> CachedPrior:
        prior = _validate_prior(prior)
        metadata = _validate_metadata(metadata, prior)
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
        payload["entry_sha256"] = _sha256_json(payload)

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

    def store(
        self,
        identity: PriorCacheIdentity,
        prior: PriorBatch,
        *,
        metadata: Mapping[str, object],
    ) -> CachedPrior:
        with _FileLock(self.lock_path_for(identity)):
            try:
                existing = self.load(identity)
            except PriorCacheCorruptError:
                existing = None
            if existing is not None:
                return existing
            return self._store_unlocked(identity, prior, metadata)

    def get_or_generate(
        self,
        identity: PriorCacheIdentity,
        generator: Callable[[], tuple[PriorBatch, Mapping[str, object]]],
    ) -> CacheLookupResult:
        quarantined = False
        try:
            existing = self.load(identity)
        except PriorCacheCorruptError:
            existing = None
            quarantined = True
        if existing is not None:
            return CacheLookupResult(cached_prior=existing, cache_hit=True)

        with _FileLock(self.lock_path_for(identity)):
            try:
                existing = self.load(identity)
            except PriorCacheCorruptError:
                existing = None
                quarantined = True
            if existing is not None:
                return CacheLookupResult(
                    cached_prior=existing,
                    cache_hit=True,
                    quarantined_corrupt_entry=quarantined,
                )

            generated = generator()
            if not isinstance(generated, tuple) or len(generated) != 2:
                raise TypeError("generator must return (PriorBatch, metadata)")
            prior, metadata = generated
            stored = self._store_unlocked(identity, prior, metadata)
            return CacheLookupResult(
                cached_prior=stored,
                cache_hit=False,
                quarantined_corrupt_entry=quarantined,
            )
