"""Live checks against the real API. Skipped unless SMALLVILLE_LIVE_TESTS=1 and ANTHROPIC_API_KEY are set.
Cost: well under $0.01 per run."""

from __future__ import annotations

import pytest

from smallville_lite.config import load_config
from smallville_lite.llm import PromptPrefix, build_llm
from smallville_lite.llm.schemas import Ping
from smallville_lite.llm.verify import verify_models

pytestmark = pytest.mark.live

RULES = "You are part of a small-town simulation test. " + "Town directory entry. " * 900


def test_configured_models_exist():
    cfg = load_config()
    results = verify_models({t.model for t in cfg.tasks.values()})
    assert all(r.ok for r in results), [r for r in results if not r.ok]


def test_structured_ping_on_both_routes():
    llm = build_llm(load_config(overrides={"budget": {"max_usd": 0.05}}))
    for task in ("ping", "ping_sonnet"):
        assert llm.call(task, output=Ping, user='Set pong to "pong" and echo to "hi".').value.pong.lower() == "pong"


def test_second_identical_prefix_reads_cache():
    llm = build_llm(load_config(overrides={"budget": {"max_usd": 0.05}}))
    prefix = PromptPrefix(RULES, "Agent: Cache Test")
    llm.call("ping_sonnet", output=Ping, user="first", prefix=prefix)
    second = llm.call("ping_sonnet", output=Ping, user="second", prefix=prefix)
    assert second.usage.cache_read_tokens > 0
