from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from formal_experiments.ours.llm_prior_v2 import (
    FORMAL_HORIZON,
    FORMAL_K_CANDIDATES,
    FormalLLMPriorConfig,
)


DEFAULT_REGISTRY = "configs/lwm_rl_llm_model_registry_v2_3.yaml"
EXPECTED_TIERS = ("tier_h", "tier_m", "tier_l")
ALLOWED_CAPABILITY_TIERS = {"high", "medium", "low_cost"}
ALLOWED_ENDPOINT_FAMILIES = {"openai_chat_completions"}
ALLOWED_STRUCTURED_OUTPUT = {"json_schema", "none"}


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int
    retryable_classes: tuple[str, ...]
    nonretryable_classes: tuple[str, ...]


@dataclass(frozen=True)
class PricingSnapshot:
    currency: str
    unit: str
    input: float
    output: float
    source: str
    snapshot_date: str


@dataclass(frozen=True)
class LLMModelSpec:
    registry_tier: str
    experiment_alias: str
    capability_tier: str
    provider: str
    upstream_family: str
    exact_model_id: str
    endpoint_family: str
    structured_output_requested: str
    structured_output_verified: bool
    requested_temperature: float | None
    actual_temperature: float | None
    usage_metadata_expected: bool
    pricing_snapshot: PricingSnapshot
    retry_policy: RetryPolicy

    def to_prior_config(
        self,
        *,
        prompt_version: str,
        k_candidates: int,
        horizon: int,
        fallback_seed: int = 20260916,
    ) -> FormalLLMPriorConfig:
        temperature = (
            self.actual_temperature
            if self.actual_temperature is not None
            else self.requested_temperature
        )
        if temperature is None:
            temperature = 0.0
        return FormalLLMPriorConfig(
            k_candidates=int(k_candidates),
            horizon=int(horizon),
            provider=str(self.provider),
            model_label=str(self.experiment_alias),
            api_model=str(self.exact_model_id),
            temperature=float(temperature),
            prompt_version=str(prompt_version),
            fallback_seed=int(fallback_seed),
        )


@dataclass(frozen=True)
class LLMModelRegistry:
    version: int
    experiment_contract: str
    frozen_date: str
    status: str
    gateway: str
    base_url: str
    protocol: str
    api_key_env: str
    prompt_version: str
    k_candidates: int
    horizon: int
    requested_temperature: float
    primary_model_selected: bool
    models: Mapping[str, LLMModelSpec]
    uniform_baseline: Mapping[str, object]

    def get(self, tier: str) -> LLMModelSpec:
        try:
            return self.models[str(tier)]
        except KeyError as exc:
            raise KeyError(f"unknown LLM registry tier: {tier}") from exc

    def by_alias(self, alias: str) -> LLMModelSpec:
        matches = [
            spec for spec in self.models.values()
            if spec.experiment_alias == str(alias)
        ]
        if len(matches) != 1:
            raise KeyError(f"unknown or ambiguous model alias: {alias}")
        return matches[0]


def _require_mapping(value, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _finite_nonnegative(value, name: str) -> float:
    number = float(value)
    if number < 0.0 or number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite and >=0")
    return number


def _parse_model(tier: str, raw: Mapping[str, object]) -> LLMModelSpec:
    capability = str(raw.get("capability_tier", ""))
    if capability not in ALLOWED_CAPABILITY_TIERS:
        raise ValueError(f"invalid capability_tier for {tier}: {capability}")

    endpoint = str(raw.get("endpoint_family", ""))
    if endpoint not in ALLOWED_ENDPOINT_FAMILIES:
        raise ValueError(f"unsupported endpoint family for {tier}: {endpoint}")

    structured = str(raw.get("structured_output_requested", ""))
    if structured not in ALLOWED_STRUCTURED_OUTPUT:
        raise ValueError(f"invalid structured-output request for {tier}")

    requested_temp = raw.get("requested_temperature")
    if requested_temp is not None:
        requested_temp = _finite_nonnegative(requested_temp, f"{tier}.requested_temperature")

    actual_temp = raw.get("actual_temperature")
    if actual_temp is not None:
        actual_temp = _finite_nonnegative(actual_temp, f"{tier}.actual_temperature")

    pricing_raw = _require_mapping(raw.get("pricing_snapshot"), f"{tier}.pricing_snapshot")
    pricing = PricingSnapshot(
        currency=str(pricing_raw.get("currency", "")),
        unit=str(pricing_raw.get("unit", "")),
        input=_finite_nonnegative(pricing_raw.get("input"), f"{tier}.price.input"),
        output=_finite_nonnegative(pricing_raw.get("output"), f"{tier}.price.output"),
        source=str(pricing_raw.get("source", "")),
        snapshot_date=str(pricing_raw.get("snapshot_date", "")),
    )
    if pricing.currency != "USD" or pricing.unit != "per_1m_tokens":
        raise ValueError(f"unexpected pricing units for {tier}")
    if not pricing.source.startswith("https://"):
        raise ValueError(f"pricing source must be https for {tier}")

    retry_raw = _require_mapping(raw.get("retry_policy"), f"{tier}.retry_policy")
    max_attempts = int(retry_raw.get("max_attempts", 0))
    if max_attempts < 1 or max_attempts > 5:
        raise ValueError(f"retry max_attempts out of range for {tier}")
    retryable = tuple(str(x) for x in retry_raw.get("retryable_classes", ()))
    nonretryable = tuple(str(x) for x in retry_raw.get("nonretryable_classes", ()))
    if not retryable or not nonretryable or set(retryable) & set(nonretryable):
        raise ValueError(f"invalid retry classes for {tier}")

    alias = str(raw.get("experiment_alias", "")).strip()
    provider = str(raw.get("provider", "")).strip()
    model_id = str(raw.get("exact_model_id", "")).strip()
    upstream = str(raw.get("upstream_family", "")).strip()
    if not alias or not provider or not model_id or not upstream:
        raise ValueError(f"missing identity field for {tier}")
    if "/" not in model_id:
        raise ValueError(f"model ID must be provider/model format for {tier}")

    return LLMModelSpec(
        registry_tier=tier,
        experiment_alias=alias,
        capability_tier=capability,
        provider=provider,
        upstream_family=upstream,
        exact_model_id=model_id,
        endpoint_family=endpoint,
        structured_output_requested=structured,
        structured_output_verified=bool(raw.get("structured_output_verified", False)),
        requested_temperature=requested_temp,
        actual_temperature=actual_temp,
        usage_metadata_expected=bool(raw.get("usage_metadata_expected", False)),
        pricing_snapshot=pricing,
        retry_policy=RetryPolicy(
            max_attempts=max_attempts,
            retryable_classes=retryable,
            nonretryable_classes=nonretryable,
        ),
    )


def load_model_registry(path: str | Path = DEFAULT_REGISTRY) -> LLMModelRegistry:
    path = Path(path)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    root = _require_mapping(payload, "registry file")
    meta = _require_mapping(root.get("registry"), "registry")
    models_raw = _require_mapping(root.get("models"), "models")

    if tuple(models_raw.keys()) != EXPECTED_TIERS:
        raise ValueError(
            f"registry tiers must be exactly {EXPECTED_TIERS}; got {tuple(models_raw.keys())}"
        )

    models = {
        tier: _parse_model(tier, _require_mapping(models_raw[tier], tier))
        for tier in EXPECTED_TIERS
    }

    aliases = [item.experiment_alias for item in models.values()]
    model_ids = [item.exact_model_id for item in models.values()]
    if len(set(aliases)) != len(aliases):
        raise ValueError("model aliases must be unique")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("exact model IDs must be unique")

    k = int(meta.get("k_candidates", 0))
    h = int(meta.get("horizon", 0))
    if k != FORMAL_K_CANDIDATES or h != FORMAL_HORIZON:
        raise ValueError("registry K/H must match frozen formal contract")

    requested_temperature = _finite_nonnegative(
        meta.get("requested_temperature"), "registry.requested_temperature"
    )

    uniform = _require_mapping(root.get("uniform_baseline"), "uniform_baseline")
    if int(uniform.get("candidate_count", 0)) != k or int(uniform.get("horizon", 0)) != h:
        raise ValueError("uniform baseline K/H mismatch")
    if str(uniform.get("prior")) != "uniform":
        raise ValueError("uniform baseline prior must be uniform")

    registry = LLMModelRegistry(
        version=int(meta.get("version", 0)),
        experiment_contract=str(meta.get("experiment_contract", "")),
        frozen_date=str(meta.get("frozen_date", "")),
        status=str(meta.get("status", "")),
        gateway=str(meta.get("gateway", "")),
        base_url=str(meta.get("base_url", "")),
        protocol=str(meta.get("protocol", "")),
        api_key_env=str(meta.get("api_key_env", "")),
        prompt_version=str(meta.get("prompt_version", "")),
        k_candidates=k,
        horizon=h,
        requested_temperature=requested_temperature,
        primary_model_selected=bool(meta.get("primary_model_selected", False)),
        models=models,
        uniform_baseline=uniform,
    )

    if registry.version != 1:
        raise ValueError("unsupported registry version")
    if registry.experiment_contract != "v2.3":
        raise ValueError("registry must bind to v2.3")
    if registry.gateway != "ofox" or registry.protocol != "openai_chat_completions":
        raise ValueError("v2.3 candidate registry must use the common OFOX gateway/protocol")
    if registry.base_url != "https://api.ofox.ai/v1":
        raise ValueError("unexpected OFOX base URL")
    if registry.api_key_env != "OFOX_API_KEY":
        raise ValueError("unexpected API key environment variable")
    if registry.primary_model_selected:
        raise ValueError("primary model must remain unselected before prior/end-to-end validation")

    for tier, spec in registry.models.items():
        if spec.provider != "ofox":
            raise ValueError(f"all candidate models must use common OFOX gateway: {tier}")
        if spec.requested_temperature != registry.requested_temperature:
            raise ValueError(f"temperature mismatch for {tier}")

    return registry
