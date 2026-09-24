"""The real backend's request/response mapping, tested against a fake SDK object (no network)."""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from smallville_lite.llm import LLMClient, PromptPrefix
from smallville_lite.llm.backend_anthropic import AnthropicBackend, usage_from_response
from smallville_lite.llm.schemas import Ping


class _FakeMessages:
    def __init__(self) -> None:
        self.kwargs: list[dict] = []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        usage = SimpleNamespace(
            input_tokens=120, output_tokens=30, cache_read_input_tokens=1500, cache_creation_input_tokens=0,
            cache_creation=SimpleNamespace(ephemeral_5m_input_tokens=0, ephemeral_1h_input_tokens=0),
        )
        content = [
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text='{"pong": "pong", "echo": "x"}'),
        ]
        return SimpleNamespace(content=content, stop_reason="end_turn", usage=usage, _request_id="req_1")


def _client(cfg):
    messages = _FakeMessages()
    backend = AnthropicBackend(client=SimpleNamespace(messages=messages))
    return LLMClient(cfg, backend), messages


def test_sonnet_request_kwargs(cfg):
    client, messages = _client(cfg)
    result = client.call("plan_day", output=Ping, user="plan", prefix=PromptPrefix("rules", "summary"))
    kw = messages.kwargs[0]
    assert kw["model"] == "claude-sonnet-5"
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"]["effort"] == "low"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in kw  # rejected by Sonnet 5
    assert kw["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert result.value.pong == "pong"  # thinking blocks are skipped; the text block is parsed
    # 120 * $2 + 30 * $10 + 1500 * $0.20 per MTok
    assert result.cost_usd == pytest.approx((120 * 2.0 + 30 * 10.0 + 1500 * 0.20) / 1e6)


def test_haiku_request_kwargs(cfg):
    client, messages = _client(cfg)
    client.call("importance", output=Ping, user="rate")
    kw = messages.kwargs[0]
    assert kw["model"] == "claude-haiku-4-5-20251001"
    assert kw["temperature"] == 0.0
    assert "thinking" not in kw and "effort" not in kw["output_config"]
    assert "system" not in kw


def test_usage_mapping_without_ttl_breakdown():
    u = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=None,
                        cache_creation_input_tokens=300, cache_creation=None)
    mapped = usage_from_response(u)
    assert mapped.cache_write_5m_tokens == 300 and mapped.cache_read_tokens == 0
