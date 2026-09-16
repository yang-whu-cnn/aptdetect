from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from typing import Mapping
from urllib.parse import urlsplit

import numpy as np

from formal_experiments.ours.llm_prior_v2 import (
    FORMAL_ACTION_NAMES,
    FORMAL_HORIZON,
    FORMAL_K_CANDIDATES,
    FormalLLMPriorConfig,
    PriorBatch,
    build_formal_prompt,
    parse_prior_response,
)


def api_key_source(env: Mapping[str, str] | None = None) -> str | None:
    values = os.environ if env is None else env
    # Match Google GenAI SDK precedence.
    if str(values.get("GOOGLE_API_KEY", "")).strip():
        return "GOOGLE_API_KEY"
    if str(values.get("GEMINI_API_KEY", "")).strip():
        return "GEMINI_API_KEY"
    return None


def require_api_key_env(env: Mapping[str, str] | None = None) -> str:
    source = api_key_source(env)
    if source is None:
        raise RuntimeError(
            "Gemini API key is not configured. Set GEMINI_API_KEY or "
            "GOOGLE_API_KEY in the environment; never commit the key to source."
        )
    return source


def _truthy_env(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def gemini_network_mode(env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    proxy = str(values.get("GEMINI_PROXY_URL", "")).strip()
    disable = _truthy_env(values.get("GEMINI_DISABLE_ENV_PROXY"))
    if proxy and disable:
        raise ValueError(
            "GEMINI_PROXY_URL and GEMINI_DISABLE_ENV_PROXY are mutually exclusive"
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
        raise ValueError("GEMINI_PROXY_URL must use http/https/socks5/socks5h")
    if not parsed.hostname or parsed.port is None:
        raise ValueError("GEMINI_PROXY_URL must include host and port")
    return proxy


def build_genai_http_options(env: Mapping[str, str] | None = None):
    """Build google-genai HttpOptions without exposing proxy credentials."""
    values = os.environ if env is None else env
    mode = gemini_network_mode(values)
    if mode == "sdk_environment_proxy":
        return None

    try:
        from google.genai import types
    except Exception as exc:
        raise RuntimeError(
            "google-genai is required to configure Gemini network options"
        ) from exc

    if mode == "direct_no_env_proxy":
        return types.HttpOptions(client_args={"trust_env": False})

    proxy = _validated_proxy_url(values["GEMINI_PROXY_URL"])
    return types.HttpOptions(
        client_args={
            "proxy": proxy,
            "trust_env": False,
        }
    )


def masked_detected_proxies(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return proxy endpoints with userinfo removed for diagnostics only."""
    import urllib.request

    # urllib.request.getproxies() reflects the same family of OS/env proxy
    # discovery used by the SDK/httpx environment path.
    detected = urllib.request.getproxies() if env is None else {
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
                        "reason": {
                            "type": "string",
                        },
                    },
                    "required": ["actions", "prior_score", "reason"],
                },
            }
        },
        "required": ["candidates"],
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GeminiPriorLiveResult:
    prior: PriorBatch
    api_call_succeeded: bool
    key_source: str
    network_mode: str
    model: str
    temperature: float
    prompt_sha256: str
    response_sha256: str
    raw_response_text: str


class GeminiPriorLiveClient:
    """
    Live Gemini adapter for the frozen B2 prior contract.

    The API key is never accepted as a constructor argument and is never
    logged. Authentication is delegated to google-genai environment-variable
    discovery (GOOGLE_API_KEY takes precedence over GEMINI_API_KEY).
    """

    def __init__(
        self,
        *,
        config: FormalLLMPriorConfig | None = None,
        client=None,
        require_env_key: bool = True,
    ):
        self.config = config if config is not None else FormalLLMPriorConfig()
        self.network_mode = gemini_network_mode()
        self.key_source = (
            require_api_key_env()
            if require_env_key
            else (api_key_source() or "TEST_CLIENT")
        )

        if client is None:
            try:
                from google import genai
            except Exception as exc:
                raise RuntimeError(
                    "google-genai is required for B2-live. Install with: "
                    "python -m pip install -U google-genai"
                ) from exc
            http_options = build_genai_http_options()
            client = genai.Client(http_options=http_options) if http_options is not None else genai.Client()

        if not hasattr(client, "interactions") or not hasattr(client.interactions, "create"):
            raise TypeError("Gemini client must expose interactions.create")
        self.client = client

    def generate(
        self,
        state,
        *,
        agent_name: str | None = None,
    ) -> GeminiPriorLiveResult:
        messages = build_formal_prompt(
            state,
            agent_name=agent_name,
            config=self.config,
        )
        system_instruction = messages[0]["content"]
        user_input = messages[1]["content"]

        interaction = self.client.interactions.create(
            model=self.config.api_model,
            system_instruction=system_instruction,
            input=user_input,
            generation_config={
                "temperature": float(self.config.temperature),
            },
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": prior_response_schema(),
            },
            store=False,
        )

        text = str(getattr(interaction, "output_text", "") or "").strip()
        if not text:
            raise RuntimeError("Gemini returned empty output_text")

        prior = parse_prior_response(text, config=self.config)
        prompt_material = system_instruction + "\n" + user_input

        return GeminiPriorLiveResult(
            prior=prior,
            api_call_succeeded=True,
            key_source=self.key_source,
            network_mode=self.network_mode,
            model=self.config.api_model,
            temperature=float(self.config.temperature),
            prompt_sha256=sha256_text(prompt_material),
            response_sha256=sha256_text(text),
            raw_response_text=text,
        )
