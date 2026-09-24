"""Usage -> dollars, and worst-case pre-call estimates. Prices are USD per million tokens."""

from __future__ import annotations

import json
import math
from typing import Any

from ..config import ModelSpec
from .types import Usage

_PER = 1_000_000


def cost_usd(model: ModelSpec, usage: Usage) -> float:
    return (
        usage.input_tokens * model.input
        + usage.output_tokens * model.output
        + usage.cache_read_tokens * model.cache_read
        + usage.cache_write_5m_tokens * model.cache_write_5m
        + usage.cache_write_1h_tokens * model.cache_write_1h
    ) / _PER


def estimate_tokens(chars: int, chars_per_token: float) -> int:
    return math.ceil(chars / chars_per_token)


def request_chars(system: list[dict[str, Any]], messages: list[dict[str, Any]], output_schema: dict[str, Any]) -> int:
    """Characters that will be sent as input (text blocks plus the JSON schema)."""
    total = sum(len(b.get("text", "")) for b in system)
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            total += len(content)
        else:
            total += sum(len(b.get("text", "")) for b in content)
    return total + len(json.dumps(output_schema))


def worst_case_cost(model: ModelSpec, input_chars: int, max_tokens: int, chars_per_token: float) -> float:
    """Upper bound: all input billed as a cache write (the priciest input rate) + max output."""
    in_tokens = estimate_tokens(input_chars, chars_per_token)
    in_rate = max(model.input, model.cache_write_5m, model.cache_write_1h)
    return (in_tokens * in_rate + max_tokens * model.output) / _PER


def embedding_cost(price_per_mtok: float, tokens: int) -> float:
    return tokens * price_per_mtok / _PER
