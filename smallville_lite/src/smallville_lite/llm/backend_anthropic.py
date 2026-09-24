"""Real backend: one Messages API call per request, via the official ``anthropic`` SDK."""

from __future__ import annotations

import time
from typing import Any

from .types import BackendRequest, BackendResponse, Usage


class AnthropicBackend:
    def __init__(self, client: Any | None = None, *, max_retries: int = 2, timeout_s: float = 120.0) -> None:
        if client is None:
            import anthropic

            # Credentials come from the environment (ANTHROPIC_API_KEY or an `ant auth` profile).
            client = anthropic.Anthropic(max_retries=max_retries, timeout=timeout_s)
        self._client = client

    def build_kwargs(self, req: BackendRequest) -> dict[str, Any]:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": req.output_schema}}
        if req.effort is not None:
            output_config["effort"] = req.effort
        kwargs: dict[str, Any] = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "messages": req.messages,
            "output_config": output_config,
        }
        if req.system:
            kwargs["system"] = req.system
        if req.thinking is not None:
            kwargs["thinking"] = req.thinking
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        return kwargs

    def complete(self, req: BackendRequest) -> BackendResponse:
        kwargs = self.build_kwargs(req)
        t0 = time.perf_counter()
        response = self._client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        return BackendResponse(
            text=text,
            stop_reason=response.stop_reason,
            usage=usage_from_response(response.usage),
            request_id=getattr(response, "_request_id", None),
            latency_ms=latency_ms,
        )


def usage_from_response(u: Any) -> Usage:
    write_total = getattr(u, "cache_creation_input_tokens", None) or 0
    breakdown = getattr(u, "cache_creation", None)
    w5 = getattr(breakdown, "ephemeral_5m_input_tokens", None) if breakdown is not None else None
    w1 = getattr(breakdown, "ephemeral_1h_input_tokens", None) if breakdown is not None else None
    if w5 is None and w1 is None:
        w5, w1 = write_total, 0
    return Usage(
        input_tokens=u.input_tokens or 0,
        output_tokens=u.output_tokens or 0,
        cache_read_tokens=getattr(u, "cache_read_input_tokens", None) or 0,
        cache_write_5m_tokens=w5 or 0,
        cache_write_1h_tokens=w1 or 0,
    )
