from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from typing import Mapping
from urllib.parse import urlsplit

from formal_experiments.ours.llm_prior_v2 import (
    FORMAL_ACTION_NAMES,
    FORMAL_HORIZON,
    FORMAL_K_CANDIDATES,
    FormalLLMPriorConfig,
    PriorBatch,
    build_formal_prompt,
    parse_prior_response,
)


OFOX_BASE_URL = "https://api.ofox.ai/v1"
OFOX_API_KEY_ENV = "OFOX_API_KEY"


def api_key_source(env: Mapping[str, str] | None = None) -> str | None:
    values = os.environ if env is None else env
    if str(values.get(OFOX_API_KEY_ENV, "")).strip():
        return OFOX_API_KEY_ENV
    return None


def require_api_key_env(env: Mapping[str, str] | None = None) -> str:
    source = api_key_source(env)
    if source is None:
        raise RuntimeError(
            "OFOX API key is not configured. Set OFOX_API_KEY in the "
            "environment; never commit the key to source or config."
        )
    return source


def _truthy_env(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def ofox_network_mode(env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    proxy = str(values.get("OFOX_PROXY_URL", "")).strip()
    disable = _truthy_env(values.get("OFOX_DISABLE_ENV_PROXY"))
    if proxy and disable:
        raise ValueError(
            "OFOX_PROXY_URL and OFOX_DISABLE_ENV_PROXY are mutually exclusive"
        )
    if proxy:
        return "explicit_proxy"
    if disable:
        return "direct_no_env_proxy"
    return "sdk_environment_proxy"


def _validated_proxy_url(value: str) -> str:
    proxy = str(value).strip()
    parsed = urlsplit(proxy)
    if parsed.scheme not in {"http", "https", "socks5", "socks5h"}:
        raise ValueError("OFOX_PROXY_URL must use http/https/socks5/socks5h")
    if not parsed.hostname or parsed.port is None:
        raise ValueError("OFOX_PROXY_URL must include host and port")
    return proxy


def build_openai_http_client(env: Mapping[str, str] | None = None):
    values = os.environ if env is None else env
    mode = ofox_network_mode(values)
    if mode == "sdk_environment_proxy":
        return None

    try:
        import httpx
    except Exception as exc:
        raise RuntimeError("httpx is required for explicit OFOX network mode") from exc

    if mode == "direct_no_env_proxy":
        return httpx.Client(trust_env=False)

    proxy = _validated_proxy_url(values["OFOX_PROXY_URL"])
    return httpx.Client(proxy=proxy, trust_env=False)


def masked_detected_proxies(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return proxy endpoints with userinfo removed for diagnostics only."""
    import urllib.request

    if env is None:
        raw_detected = urllib.request.getproxies()
        detected = {
            key: value
            for key, value in raw_detected.items()
            if str(key).lower() in {"http", "https", "all", "ftp"}
        }
    else:
        detected = {
            key[:-6].lower(): value
            for key, value in env.items()
            if key.lower() in {"http_proxy", "https_proxy", "all_proxy"}
            and str(value).strip()
        }

    out: dict[str, str] = {}
    for name, raw in detected.items():
        try:
            parsed = urlsplit(str(raw))
            host = parsed.hostname or "?"
            port = f":{parsed.port}" if parsed.port is not None else ""
            out[str(name)] = f"{parsed.scheme or '?'}://{host}{port}"
        except Exception:
            out[str(name)] = "<configured>"
    return out


def classify_live_exception(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}".lower()
    if (
        "429" in message
        or "quota" in message
        or "rate limit" in message
        or "too_many_requests" in message
        or "insufficient_quota" in message
    ):
        return "quota_blocked"
    if (
        "401" in message
        or "403" in message
        or "authentication" in message
        or "api key" in message
        or "unauthorized" in message
        or "forbidden" in message
    ):
        return "auth_blocked"
    if (
        "ssl" in message
        or "connecterror" in message
        or "connection" in message
        or "timeout" in message
        or "proxy" in message
    ):
        return "network_blocked"
    if "404" in message or "model" in message and "not found" in message:
        return "model_blocked"
    return "api_error"


def prior_response_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": FORMAL_K_CANDIDATES,
                "maxItems": FORMAL_K_CANDIDATES,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "actions": {
                            "type": "array",
                            "minItems": FORMAL_HORIZON,
                            "maxItems": FORMAL_HORIZON,
                            "items": {
                                "type": "string",
                                "enum": list(FORMAL_ACTION_NAMES),
                            },
                        },
                        "prior_score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["actions", "prior_score", "reason"],
                },
            }
        },
        "required": ["candidates"],
    }


def response_format_json_schema() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "lwm_rl_candidate_plans",
            "schema": prior_response_schema(),
        },
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OFOXPriorLiveResult:
    prior: PriorBatch
    api_call_succeeded: bool
    key_source: str
    network_mode: str
    base_url: str
    model: str
    temperature: float
    prompt_sha256: str
    response_sha256: str
    raw_response_text: str


class OFOXPriorLiveClient:
    """Live OFOX OpenAI-compatible adapter for the frozen B2 prior contract."""

    def __init__(
        self,
        *,
        config: FormalLLMPriorConfig | None = None,
        client=None,
        require_env_key: bool = True,
    ):
        self.config = config if config is not None else FormalLLMPriorConfig()
        self.network_mode = ofox_network_mode()
        self.key_source = (
            require_api_key_env()
            if require_env_key
            else (api_key_source() or "TEST_CLIENT")
        )

        if client is None:
            try:
                from openai import OpenAI
            except Exception as exc:
                raise RuntimeError(
                    "openai is required for B2-live OFOX. Install with: "
                    "python -m pip install -U openai"
                ) from exc

            key = str(os.environ.get(OFOX_API_KEY_ENV, "")).strip()
            if require_env_key and not key:
                raise RuntimeError("OFOX_API_KEY is required")

            http_client = build_openai_http_client()
            kwargs = {
                "base_url": OFOX_BASE_URL,
                "api_key": key if key else "test-not-used",
            }
            if http_client is not None:
                kwargs["http_client"] = http_client
            client = OpenAI(**kwargs)

        if not hasattr(client, "chat") or not hasattr(client.chat, "completions"):
            raise TypeError("OFOX client must expose chat.completions.create")
        if not hasattr(client.chat.completions, "create"):
            raise TypeError("OFOX client must expose chat.completions.create")
        self.client = client

    def generate(
        self,
        state,
        *,
        agent_name: str | None = None,
    ) -> OFOXPriorLiveResult:
        messages = build_formal_prompt(
            state,
            agent_name=agent_name,
            config=self.config,
        )

        response = self.client.chat.completions.create(
            model=self.config.api_model,
            messages=messages,
            temperature=float(self.config.temperature),
            response_format=response_format_json_schema(),
        )

        try:
            text = str(response.choices[0].message.content or "").strip()
        except Exception as exc:
            raise RuntimeError("OFOX returned an invalid Chat Completions response") from exc
        if not text:
            raise RuntimeError("OFOX returned empty message content")

        prior = parse_prior_response(text, config=self.config)
        prompt_material = "\n".join(str(item["content"]) for item in messages)

        return OFOXPriorLiveResult(
            prior=prior,
            api_call_succeeded=True,
            key_source=self.key_source,
            network_mode=self.network_mode,
            base_url=OFOX_BASE_URL,
            model=self.config.api_model,
            temperature=float(self.config.temperature),
            prompt_sha256=sha256_text(prompt_material),
            response_sha256=sha256_text(text),
            raw_response_text=text,
        )
