"""Anthropic list prices and per-request cost.

Prices are USD per million tokens, from https://platform.claude.com/docs/en/about-claude/pricing
(checked 2026-10-27). Claude Haiku 5.5 is priced per request by prompt length: a request whose
prompt (input + cache writes + cache reads) exceeds 100,000 tokens pays the higher tier, so cost
must be computed per request, never on summed usage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Rate:
    """USD per million tokens."""

    input: float
    output: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float


# model -> list of (max_prompt_tokens or None, Rate), ascending.
PRICES: dict[str, list[tuple[int | None, Rate]]] = {
    "claude-haiku-5-5": [
        (100_000, Rate(0.10, 0.50, 0.125, 0.20, 0.01)),
        (None, Rate(0.50, 2.50, 0.625, 1.00, 0.05)),
    ],
    "claude-sonnet-5-5": [(None, Rate(2.0, 10.0, 2.50, 4.0, 0.10))],
    "claude-opus-5-5": [(None, Rate(4.0, 20.0, 5.0, 8.0, 0.20))],
    "claude-sonnet-4-6": [(None, Rate(3.0, 15.0, 3.75, 6.0, 0.30))],
    "claude-haiku-4-5": [(None, Rate(1.0, 5.0, 1.25, 2.0, 0.10))],
}


def normalize(model: str) -> str:
    """'anthropic/claude-haiku-5-5', 'claude-haiku-5-5-20261001' -> 'claude-haiku-5-5'."""
    name = (model or "").strip().split("/")[-1]
    name = re.sub(r"-\d{8}$", "", name)
    name = re.sub(r"@.*$", "", name)
    return name


def rate(model: str, prompt_tokens: int) -> Rate:
    tiers = PRICES.get(normalize(model))
    if tiers is None:
        raise KeyError(f"no price for model {model!r}; add it to h2x/pricing.py")
    for limit, r in tiers:
        if limit is None or prompt_tokens <= limit:
            return r
    return tiers[-1][1]


def request_cost(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_write_5m: int = 0,
    cache_write_1h: int = 0,
    cache_read: int = 0,
) -> float:
    """Cost of ONE API request. `input_tokens` excludes cache reads and writes."""
    prompt = input_tokens + cache_write_5m + cache_write_1h + cache_read
    r = rate(model, prompt)
    return (
        input_tokens * r.input
        + output_tokens * r.output
        + cache_write_5m * r.cache_write_5m
        + cache_write_1h * r.cache_write_1h
        + cache_read * r.cache_read
    ) / 1_000_000


def litellm_usage_cost(model: str, usage: dict | None) -> float:
    """Cost of one request from a LiteLLM-style usage dict (prompt_tokens includes cached)."""
    if not usage:
        return 0.0
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    cached = int(usage.get("cache_read_input_tokens") or usage.get("cached_tokens") or 0)
    written = int(usage.get("cache_creation_input_tokens") or 0)
    plain = max(0, prompt - cached - written)
    return request_cost(
        model,
        input_tokens=plain,
        output_tokens=completion,
        cache_write_5m=written,
        cache_read=cached,
    )


def anthropic_usage_cost(model: str, usage) -> float:
    """Cost of one request from an Anthropic SDK `Usage` object or dict."""
    if usage is None:
        return 0.0

    def get(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    created = get(usage, "cache_creation")
    w1h = int(get(created, "ephemeral_1h_input_tokens") or 0) if created else 0
    w5m_total = int(get(usage, "cache_creation_input_tokens") or 0)
    w5m = max(0, w5m_total - w1h)
    return request_cost(
        model,
        input_tokens=int(get(usage, "input_tokens") or 0),
        output_tokens=int(get(usage, "output_tokens") or 0),
        cache_write_5m=w5m,
        cache_write_1h=w1h,
        cache_read=int(get(usage, "cache_read_input_tokens") or 0),
    )


def litellm_registration(models: list[str]) -> dict:
    """A `litellm.register_model` payload with base-tier prices, so LiteLLM stops logging
    'model not mapped' errors. h2x computes the authoritative cost itself, per request."""
    out = {}
    for model in models:
        tiers = PRICES.get(normalize(model))
        if not tiers:
            continue
        r = tiers[0][1]
        entry = {
            "input_cost_per_token": r.input / 1e6,
            "output_cost_per_token": r.output / 1e6,
            "cache_creation_input_token_cost": r.cache_write_5m / 1e6,
            "cache_read_input_token_cost": r.cache_read / 1e6,
            "litellm_provider": "anthropic",
            "mode": "chat",
            "max_input_tokens": 1_000_000,
            "max_output_tokens": 128_000,
            "max_tokens": 128_000,
            "supports_function_calling": True,
            "supports_tool_choice": True,
        }
        out[normalize(model)] = entry
        out[f"anthropic/{normalize(model)}"] = entry
    return out
