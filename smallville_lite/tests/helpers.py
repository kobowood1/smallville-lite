"""Builders for cognition tests: a Mind over the party world with the stub LLM and stub embeddings."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable

from smallville_lite.agent import Agent
from smallville_lite.clock import GameClock
from smallville_lite.cognition import Mind
from smallville_lite.config import load_config
from smallville_lite.embed import CachedEmbedder, EmbeddingCache
from smallville_lite.embed.stub import StubEmbedder
from smallville_lite.llm import FakeBackend, FakeReply, LLMClient
from smallville_lite.llm.stub_brain import make_responders
from smallville_lite.memory import MemoryStream, NewMemory
from smallville_lite.sim.events import ListSink
from smallville_lite.sim.scenario import load_scenario, render_rules

START = datetime(2023, 2, 13, 6, 0)


class Rig:
    """Everything a cognition test needs, with direct access to the fake backend and events."""

    def __init__(self, overrides: dict[str, Any] | None = None, responders: dict[str, Callable] | None = None) -> None:
        self.scenario = load_scenario("party")
        ov = {"run": {"stub": True}, "budget": {"max_usd": 100.0}}
        for k, v in (overrides or {}).items():
            ov[k] = {**ov.get(k, {}), **v}
        self.config = load_config(overrides=ov)
        self.sink = ListSink()
        self.fake = FakeBackend(
            seed=self.config.seed,
            min_cache_tokens={m.id: m.min_cache_tokens for m in self.config.models.values()},
            chars_per_token={m.id: m.chars_per_token for m in self.config.models.values()},
            responders=make_responders(self.scenario.facts, self.scenario.events),
        )
        for task, fn in (responders or {}).items():
            self.fake.register(task, fn)
        self.llm = LLMClient(self.config, self.fake, sink=self.sink, log_prompts=True)
        self.embedder = CachedEmbedder(StubEmbedder(), EmbeddingCache(None), sink=self.sink)
        self.world = self.scenario.world
        self.mind = Mind(config=self.config, llm=self.llm, embedder=self.embedder, sink=self.sink, world=self.world,
                         clock=GameClock(START, self.config.sim.tick_minutes), rules=render_rules(self.world))
        self.mind.now = START
        self.agents: dict[str, Agent] = {}
        for ident in self.scenario.identities:
            agent = Agent(ident, MemoryStream(ident.name, self.embedder, sink=self.sink), retention=self.config.sim.retention)
            self.agents[ident.first_name] = agent
            self.mind.agents[ident.name] = agent

    def at(self, hour: int, minute: int = 0, day: int = 0) -> datetime:
        self.mind.now = START.replace(hour=hour, minute=minute) + timedelta(days=day)
        return self.mind.now

    def seed(self, first: str, texts: list[str], importance: int = 3, when: datetime | None = None, type: str = "observation"):
        agent = self.agents[first]
        return agent.memory.add_many(when or self.mind.now,
                                     [NewMemory(type, "seed", t, importance=importance) for t in texts])

    def requests(self, task: str):
        return [r for r in self.fake.requests if r.task == task]

    def events(self, type: str):
        return self.sink.of_type(type)


def script(*replies: Any) -> Callable:
    """A responder that returns the given replies in order (dicts or FakeReply), repeating the last."""
    items = list(replies)

    def responder(req, rng):
        return items.pop(0) if len(items) > 1 else items[0]

    return responder


__all__ = ["FakeReply", "Rig", "START", "script"]
