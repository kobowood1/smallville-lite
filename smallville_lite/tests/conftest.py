from __future__ import annotations

import os

import pytest

from smallville_lite.config import Config, load_config
from smallville_lite.llm import FakeBackend, LLMClient
from smallville_lite.sim.events import ListSink


@pytest.fixture
def cfg() -> Config:
    return load_config(overrides={"run": {"stub": True}})


@pytest.fixture
def sink() -> ListSink:
    return ListSink()


@pytest.fixture
def fake(cfg: Config) -> FakeBackend:
    return FakeBackend(seed=cfg.seed, min_cache_tokens={m.id: m.min_cache_tokens for m in cfg.models.values()})


@pytest.fixture
def llm(cfg: Config, fake: FakeBackend, sink: ListSink) -> LLMClient:
    return LLMClient(cfg, fake, sink=sink, log_prompts=True)


def live_enabled() -> bool:
    return os.environ.get("SMALLVILLE_LIVE_TESTS") == "1" and bool(os.environ.get("ANTHROPIC_API_KEY"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if live_enabled():
        return
    skip = pytest.mark.skip(reason="live API test: set SMALLVILLE_LIVE_TESTS=1 and ANTHROPIC_API_KEY")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
