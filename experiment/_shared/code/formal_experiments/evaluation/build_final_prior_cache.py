"""Build the final offline GPT-5.6 Sol prior cache from train replay states.

The default is a fail-closed dry run. Paid provider calls require the explicit
``--execute-live`` flag. Formal training/evaluation only consumes PriorCache.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Callable, Iterable, Mapping

import numpy as np
import yaml

from formal_experiments.ours.llm_prior_v2 import (
    FORMAL_HORIZON, FORMAL_K_CANDIDATES, FormalLLMPriorConfig, PriorBatch,
)
from formal_experiments.ours.prior_cache import (
    CACHE_FORMAT_VERSION, PriorCache, exact_state_float32, exact_state_sha256, make_identity,
)
from shared.formal_state import BLUE_AGENTS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_BANK = "outputs/formal_replay_final_20260917/train.jsonl"
DEFAULT_OUTPUT = "outputs/lwm_rl_final_20260917/prior_cache_final"
DEFAULT_REGISTRY = "configs/llm_model_registry_v1.yaml"
DEFAULT_PRIMARY_SELECTION = "configs/formal_v3/primary_model_selection.yaml"
DEFAULT_TARGET_SIZE = 2000
MODEL_ALIAS = "llm_h_gpt56_sol"
PROVIDER = "ofox"
EXACT_MODEL_ID = "openai/gpt-5.6-sol"
PROMPT_VERSION = "lwm_rl_gate_b_v2_1"
GENERATION_CONFIG = {
    "mode": "offline_teacher", "temperature": 0.2, "structured_output": "json_schema"
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_registry(path: str | Path) -> tuple[dict, str]:
    """Load and validate the registry values that define cache provenance."""
    registry_path = resolve_path(path)
    with registry_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("model registry must be a mapping")
    header = payload.get("registry")
    tier = payload.get("models", {}).get("tier_h")
    if not isinstance(header, dict) or not isinstance(tier, dict):
        raise ValueError("model registry must contain registry and models.tier_h mappings")
    expected = {
        "registry.version": (header.get("version"), 1),
        "registry.gateway": (header.get("gateway"), PROVIDER),
        "registry.prompt_version": (header.get("prompt_version"), PROMPT_VERSION),
        "registry.k_candidates": (header.get("k_candidates"), FORMAL_K_CANDIDATES),
        "registry.horizon": (header.get("horizon"), FORMAL_HORIZON),
        "registry.requested_temperature": (
            header.get("requested_temperature"), GENERATION_CONFIG["temperature"]
        ),
        "models.tier_h.experiment_alias": (tier.get("experiment_alias"), MODEL_ALIAS),
        "models.tier_h.provider": (tier.get("provider"), PROVIDER),
        "models.tier_h.exact_model_id": (tier.get("exact_model_id"), EXACT_MODEL_ID),
        "models.tier_h.requested_temperature": (
            tier.get("requested_temperature"), GENERATION_CONFIG["temperature"]
        ),
        "models.tier_h.actual_temperature": (
            tier.get("actual_temperature"), GENERATION_CONFIG["temperature"]
        ),
        "models.tier_h.structured_output_requested": (
            tier.get("structured_output_requested"), GENERATION_CONFIG["structured_output"]
        ),
        "models.tier_h.structured_output_verified": (
            tier.get("structured_output_verified"), True
        ),
    }
    mismatches = [
        f"{name}: registry={actual!r}, expected={wanted!r}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    config = FormalLLMPriorConfig()
    if config.api_model != EXACT_MODEL_ID or config.prompt_version != PROMPT_VERSION:
        mismatches.append("FormalLLMPriorConfig model/prompt differs from frozen builder")
    if config.k_candidates != FORMAL_K_CANDIDATES or config.horizon != FORMAL_HORIZON:
        mismatches.append("FormalLLMPriorConfig K/H differs from frozen builder")
    if float(config.temperature) != float(GENERATION_CONFIG["temperature"]):
        mismatches.append("FormalLLMPriorConfig temperature differs from frozen builder")
    if mismatches:
        raise ValueError("frozen registry provenance mismatch: " + "; ".join(mismatches))
    return payload, sha256_file(registry_path)


def load_primary_model_selection(path: str | Path, *, registry_sha256: str) -> tuple[dict, str]:
    """Validate the independent user-approval record against frozen model identity."""
    selection_path = resolve_path(path)
    with selection_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    selection = payload.get("selection") if isinstance(payload, dict) else None
    if not isinstance(selection, dict):
        raise ValueError("primary model selection must contain a selection mapping")
    expected = {
        "schema": "formal_v3_primary_model_selection_v1",
        "selected": True,
        "model_alias": MODEL_ALIAS,
        "exact_model_id": EXACT_MODEL_ID,
        "provider": PROVIDER,
        "registry_version": 1,
        "registry_sha256": registry_sha256,
        "authority": "explicit_user_approval",
        "scope": "final_offline_teacher_cache",
    }
    mismatches = [
        f"selection.{key}: actual={selection.get(key)!r}, expected={wanted!r}"
        for key, wanted in expected.items() if selection.get(key) != wanted
    ]
    if mismatches:
        raise ValueError("primary model selection mismatch: " + "; ".join(mismatches))
    return payload, sha256_file(selection_path)


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class StateBankRecord:
    source_index: int
    episode_seed: int
    agent_name: str
    action_id: int
    evidence: int
    state: np.ndarray
    state_sha256: str


def load_train_state_bank(path: str | Path) -> list[StateBankRecord]:
    """Load observable D27 states while preserving public agent identity."""
    records, seen = [], set()
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            agent_name = str(row.get("agent_name", ""))
            if agent_name not in BLUE_AGENTS:
                raise ValueError(f"invalid or missing agent_name at line {source_index + 1}")
            state = exact_state_float32(row["state"])
            digest = exact_state_sha256(state)
            key = (agent_name, digest)
            if key in seen:
                continue
            seen.add(key)
            evidence = int(float(state[17]))
            if evidence not in (0, 1):
                raise ValueError(f"D27 feature 17 must be binary at line {source_index + 1}")
            action_id = int(row.get("requested_action_id", row.get("executed_index", -1)))
            if action_id not in range(4):
                raise ValueError(f"invalid A4 action at line {source_index + 1}: {action_id}")
            records.append(StateBankRecord(
                source_index, int(row["episode_seed"]), agent_name, action_id,
                evidence, state, digest,
            ))
    if not records:
        raise ValueError("train state bank is empty")
    return records


def select_representative_states(
    records: Iterable[StateBankRecord], target_size: int
) -> list[StateBankRecord]:
    """Deterministically round-robin agent/action/evidence strata."""
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    cells = defaultdict(list)
    for record in records:
        cells[(record.agent_name, record.action_id, record.evidence)].append(record)
    for values in cells.values():
        values.sort(key=lambda item: (item.episode_seed, item.source_index))
    selected, offset = [], 0
    while len(selected) < target_size:
        added = False
        for key in sorted(cells):
            if offset < len(cells[key]):
                selected.append(cells[key][offset])
                added = True
                if len(selected) == target_size:
                    break
        if not added:
            break
        offset += 1
    if len(selected) != target_size:
        raise ValueError(
            f"insufficient unique train states: requested {target_size}, selected {len(selected)}"
        )
    return selected


def coverage(records: Iterable[StateBankRecord]) -> dict:
    rows = list(records)
    return {
        "total": len(rows),
        "per_agent": dict(sorted(Counter(x.agent_name for x in rows).items())),
        "per_action": {str(k): v for k, v in sorted(Counter(x.action_id for x in rows).items())},
        "per_evidence": {str(k): v for k, v in sorted(Counter(x.evidence for x in rows).items())},
        "per_agent_action_evidence": {
            f"{a}:{b}:{c}": n for (a, b, c), n in sorted(
                Counter((x.agent_name, x.action_id, x.evidence) for x in rows).items()
            )
        },
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=".manifest.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_cache(
    *, selected: list[StateBankRecord], output_root: str | Path,
    registry_sha256: str,
    generator: Callable[[np.ndarray, str], tuple[PriorBatch, Mapping[str, object]]] | None = None,
    generator_factory: Callable[[], Callable[[np.ndarray, str], tuple[PriorBatch, Mapping[str, object]]]] | None = None,
    workers: int = 1, max_attempts: int = 3, retry_backoff_seconds: float = 1.0,
    progress_path: str | Path | None = None,
    classify_exception: Callable[[Exception], str] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    result_validator: Callable[[PriorBatch, Mapping[str, object]], None] | None = None,
) -> dict:
    """Concurrent, resumable cache build with one generator/client per worker."""
    if workers < 1 or workers > 32:
        raise ValueError("workers must be in [1,32]")
    if max_attempts < 1 or max_attempts > 3:
        raise ValueError("max_attempts must be in [1,3]")
    if (generator is None) == (generator_factory is None):
        raise ValueError("provide exactly one of generator or generator_factory")
    if classify_exception is None:
        from formal_experiments.ours.ofox_prior_client import classify_live_exception
        classify_exception = classify_live_exception
    retryable = {"quota_blocked", "network_blocked", "server_error"}
    cache = PriorCache(resolve_path(output_root))
    identities = [make_identity(
        split="train", model_alias=MODEL_ALIAS, provider=PROVIDER,
        exact_model_id=EXACT_MODEL_ID, registry_version=1,
        registry_sha256=registry_sha256, prompt_version=PROMPT_VERSION,
        agent_name=record.agent_name, generation_config=GENERATION_CONFIG,
        state=record.state,
    ) for record in selected]
    cache_keys = [identity.cache_key for identity in identities]
    if len(cache_keys) != len(set(cache_keys)):
        raise ValueError("selected states do not map to unique cache keys")
    local = threading.local()
    progress_lock = threading.Lock()
    progress = resolve_path(progress_path) if progress_path is not None else None
    if progress is not None:
        progress.parent.mkdir(parents=True, exist_ok=True)

    def get_generator():
        if generator is not None:
            return generator
        if not hasattr(local, "generator"):
            local.generator = generator_factory()
        return local.generator

    def emit(event: dict) -> None:
        if progress is None:
            return
        safe = {k: event[k] for k in (
            "selection_index", "state_sha256", "cache_key", "status", "attempt", "error_class"
        ) if k in event}
        with progress_lock, progress.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(safe, sort_keys=True, separators=(",", ":")) + "\n")

    def process(item):
        selection_index, record, identity = item
        stored = cache.load(identity)
        if stored is not None:
            payload = json.loads(stored.path.read_text(encoding="utf-8"))
            emit({"selection_index": selection_index, "state_sha256": record.state_sha256,
                  "cache_key": identity.cache_key, "status": "verified_resume_hit", "attempt": 0})
            return {"kind": "hit", "entry_sha256": str(payload["entry_sha256"])}
        for attempt in range(1, max_attempts + 1):
            started = time.perf_counter()
            try:
                prior, supplied = get_generator()(record.state, record.agent_name)
                if result_validator is not None:
                    result_validator(prior, supplied)
            except Exception as exc:
                error_class = str(classify_exception(exc))
                emit({"selection_index": selection_index, "state_sha256": record.state_sha256,
                      "cache_key": identity.cache_key, "status": "retry" if error_class in retryable and attempt < max_attempts else "failed",
                      "attempt": attempt, "error_class": error_class})
                if error_class not in retryable or attempt >= max_attempts:
                    return {"kind": "failure", "selection_index": selection_index,
                            "state_sha256": record.state_sha256, "cache_key": identity.cache_key,
                            "attempts": attempt, "error_class": error_class}
                sleep_fn(float(retry_backoff_seconds) * (2 ** (attempt - 1)))
                continue
            metadata = dict(supplied)
            metadata.setdefault("api_success", True)
            metadata.setdefault("fallback_count", sum(x != "llm" for x in prior.sources))
            metadata.setdefault("latency_ms", (time.perf_counter() - started) * 1000.0)
            metadata["retry_count"] = attempt - 1
            metadata.update({
                "source_episode_seed": record.episode_seed,
                "source_index": record.source_index,
                "source_action_id": record.action_id,
                "source_evidence": record.evidence,
                "agent_name": record.agent_name,
                "teacher_model": EXACT_MODEL_ID,
                "prompt_version": PROMPT_VERSION,
                "offline_generation": True,
            })
            stored = cache.store(identity, prior, metadata=metadata)
            payload = json.loads(stored.path.read_text(encoding="utf-8"))
            emit({"selection_index": selection_index, "state_sha256": record.state_sha256,
                  "cache_key": identity.cache_key, "status": "stored_verified", "attempt": attempt})
            return {"kind": "generated", "entry_sha256": str(payload["entry_sha256"])}
        raise AssertionError("unreachable retry loop")

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="prior-cache") as executor:
        results = list(executor.map(process, (
            (index, record, identities[index]) for index, record in enumerate(selected)
        )))
    generated = sum(x["kind"] == "generated" for x in results)
    hits = sum(x["kind"] == "hit" for x in results)
    failures = sorted((x for x in results if x["kind"] == "failure"), key=lambda x: x["selection_index"])
    entry_hashes = sorted(x["entry_sha256"] for x in results if "entry_sha256" in x)
    return {
        "generated": generated, "cache_hits": hits,
        "verified_entries": len(entry_hashes),
        "failed_entries": len(failures), "failures": failures,
        "workers": workers, "max_attempts": max_attempts,
        "entry_sha256_set_sha256": _canonical_sha256(entry_hashes),
    }


def _live_generator():
    from formal_experiments.ours.ofox_prior_client import OFOXPriorLiveClient
    client = OFOXPriorLiveClient(config=FormalLLMPriorConfig())

    def generate(state: np.ndarray, agent_name: str):
        started = time.perf_counter()
        result = client.generate(state, agent_name=agent_name)
        validate_live_result(result)
        return result.prior, {
            "api_success": bool(result.api_call_succeeded),
            "fallback_count": sum(x != "llm" for x in result.prior.sources),
            "prompt_sha256": result.prompt_sha256,
            "response_sha256": result.response_sha256,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "retry_count": 0, "network_mode": result.network_mode,
            "requested_model": EXACT_MODEL_ID, "actual_model": result.model,
            "requested_temperature": GENERATION_CONFIG["temperature"],
            "actual_temperature": result.temperature,
            "structured_response_verified": True,
        }
    return generate


def validate_live_result(result) -> None:
    """Reject gateway model/config drift before an entry can be stored."""
    if str(result.model) != EXACT_MODEL_ID:
        raise RuntimeError(
            f"provider returned unexpected model: {result.model!r}; expected {EXACT_MODEL_ID!r}"
        )
    if float(result.temperature) != float(GENERATION_CONFIG["temperature"]):
        raise RuntimeError(
            "provider result temperature differs from frozen generation configuration"
        )


def validate_probe_generation(prior: PriorBatch, metadata: Mapping[str, object]) -> None:
    """Fail before storage unless the live response matches the frozen contract."""
    if str(metadata.get("actual_model")) != EXACT_MODEL_ID:
        raise RuntimeError("probe actual model differs from frozen model")
    if float(metadata.get("actual_temperature", float("nan"))) != float(GENERATION_CONFIG["temperature"]):
        raise RuntimeError("probe actual temperature differs from frozen configuration")
    if metadata.get("structured_response_verified") is not True:
        raise RuntimeError("probe structured response was not verified")
    if not isinstance(prior, PriorBatch):
        raise TypeError("probe result must contain a validated PriorBatch")


def run_probe(*, selected: list[StateBankRecord], output_root: str | Path,
              registry_sha256: str, primary_selection_sha256: str,
              state_selection_sha256: str, generator_factory, workers: int = 1,
              max_attempts: int = 3, retry_backoff_seconds: float = 1.0) -> dict:
    """Execute exactly one formal identity and verify it by a fresh cache reload."""
    if len(selected) != DEFAULT_TARGET_SIZE:
        raise ValueError("probe requires the complete frozen 2000-state selection")
    output = resolve_path(output_root)
    result = build_cache(
        selected=selected[:1], output_root=output, registry_sha256=registry_sha256,
        generator_factory=generator_factory, workers=workers, max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        progress_path=output / "probe_progress.jsonl",
        result_validator=validate_probe_generation,
    )
    if result["verified_entries"] != 1 or result["failed_entries"]:
        raise RuntimeError("probe did not produce one verified cache entry")
    row = selected[0]
    identity = make_identity(
        split="train", model_alias=MODEL_ALIAS, provider=PROVIDER,
        exact_model_id=EXACT_MODEL_ID, registry_version=1,
        registry_sha256=registry_sha256, prompt_version=PROMPT_VERSION,
        agent_name=row.agent_name, generation_config=GENERATION_CONFIG, state=row.state,
    )
    reloaded = PriorCache(output).load(identity)
    if reloaded is None:
        raise RuntimeError("probe cache reload failed")
    report = {
        "status": "PROBE_PASS", "provider_calls": result["generated"],
        "cache_hits": result["cache_hits"], "verified_entries": 1,
        "cache_key": identity.cache_key, "entry_sha256_set_sha256": result["entry_sha256_set_sha256"],
        "registry_sha256": registry_sha256,
        "primary_model_selection_sha256": primary_selection_sha256,
        "state_selection_sha256": state_selection_sha256,
        "full_selection_size": len(selected), "probe_selection_index": 0,
        "actual_model": reloaded.metadata.get("actual_model"),
        "actual_temperature": reloaded.metadata.get("actual_temperature"),
        "structured_response_verified": reloaded.metadata.get("structured_response_verified"),
    }
    validate_probe_generation(reloaded.prior, reloaded.metadata)
    _atomic_json(output / "probe_manifest.json", report)
    return report


def deterministic_final_manifest_sha256(*, target_size: int, registry_sha256: str,
                                        primary_model_selection_sha256: str,
                                        selection_sha256: str, result: dict,
                                        status: str) -> str:
    """Hash only final logical content, excluding timestamps and execution scheduling."""
    return _canonical_sha256({
        "target_size": int(target_size),
        "registry_sha256": registry_sha256,
        "primary_model_selection_sha256": primary_model_selection_sha256,
        "selection_sha256": selection_sha256,
        "entry_sha256_set_sha256": result["entry_sha256_set_sha256"],
        "verified_entries": int(result["verified_entries"]),
        "failed_entries": int(result["failed_entries"]),
        "status": status,
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-bank", default=DEFAULT_STATE_BANK)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--primary-selection", default=DEFAULT_PRIMARY_SELECTION)
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.0)
    live = parser.add_mutually_exclusive_group()
    live.add_argument("--execute-live", action="store_true",
                      help="explicitly authorize paid offline teacher calls")
    live.add_argument("--probe-live", action="store_true",
                      help="execute one resumable formal cache connectivity probe")
    args = parser.parse_args(argv)

    source = resolve_path(args.state_bank)
    _, registry_sha256 = load_frozen_registry(args.registry)
    _, primary_selection_sha256 = load_primary_model_selection(
        args.primary_selection, registry_sha256=registry_sha256
    )
    selected = select_representative_states(load_train_state_bank(source), args.target_size)
    report = {
        "mode": "probe_live" if args.probe_live else ("live" if args.execute_live else "dry_run"),
        "split": "train",
        "target_size": args.target_size, "state_bank": str(source),
        "state_bank_sha256": sha256_file(source),
        "selection": "round_robin_agent_action_evidence_v1",
        "selection_sha256": _canonical_sha256([(x.agent_name, x.state_sha256) for x in selected]),
        "coverage": coverage(selected), "cache_format_version": CACHE_FORMAT_VERSION,
        "model_alias": MODEL_ALIAS, "exact_model_id": EXACT_MODEL_ID,
        "prompt_version": PROMPT_VERSION, "generation_config": GENERATION_CONFIG,
        "registry_sha256": registry_sha256,
        "primary_model_selection": str(resolve_path(args.primary_selection)),
        "primary_model_selection_sha256": primary_selection_sha256,
    }
    if not args.execute_live and not args.probe_live:
        report["status"] = "DRY_RUN_PASS_NO_PROVIDER_CALLS"
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.probe_live:
        probe = run_probe(
            selected=selected, output_root=args.output, registry_sha256=registry_sha256,
            primary_selection_sha256=primary_selection_sha256,
            state_selection_sha256=report["selection_sha256"], generator_factory=_live_generator,
            workers=1, max_attempts=args.max_attempts,
            retry_backoff_seconds=args.retry_backoff_seconds,
        )
        print(json.dumps(probe, indent=2, sort_keys=True))
        return 0

    result = build_cache(
        selected=selected, output_root=args.output, registry_sha256=registry_sha256,
        generator_factory=_live_generator, workers=args.workers, max_attempts=args.max_attempts,
        retry_backoff_seconds=args.retry_backoff_seconds,
        progress_path=resolve_path(args.output) / "progress.jsonl",
    )
    report.update(result)
    report.update({
        "status": "PASS" if (
            args.target_size == DEFAULT_TARGET_SIZE
            and result["verified_entries"] == DEFAULT_TARGET_SIZE
            and result["failed_entries"] == 0
        ) else "FAIL",
        "registry_sha256": registry_sha256,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    report["deterministic_final_manifest_sha256"] = deterministic_final_manifest_sha256(
        target_size=args.target_size, registry_sha256=registry_sha256,
        primary_model_selection_sha256=primary_selection_sha256,
        selection_sha256=report["selection_sha256"], result=result,
        status=report["status"],
    )
    _atomic_json(resolve_path(args.output) / "manifest.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
