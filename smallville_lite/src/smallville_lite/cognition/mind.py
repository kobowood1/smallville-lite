"""Shared context for the cognitive modules: config, model layer, world, clock, agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterable

from ..clock import GameClock
from ..config import Config, SimSettings
from ..embed import CachedEmbedder
from ..llm import LLMClient, PromptPrefix, SummaryCache
from ..memory import ScoredMemory, Weights, retrieve
from ..sim.events import EventSink
from ..world import World

if TYPE_CHECKING:
    from ..agent import Agent


@dataclass
class Mind:
    config: Config
    llm: LLMClient
    embedder: CachedEmbedder
    sink: EventSink
    world: World
    clock: GameClock
    rules: str                                   # system[0]: simulation rules + town directory
    agents: dict[str, "Agent"] = field(default_factory=dict)
    now: datetime | None = None
    summaries: SummaryCache = field(init=False)

    def __post_init__(self) -> None:
        from .summary import regenerate_summary

        self.summaries = SummaryCache(
            regenerate=lambda name: regenerate_summary(self, self.agents[name]),
            refresh_on=self.config.summary_refresh_on,
            on_update=self._summary_updated,
        )
        s = self.config.sim
        self.weights = Weights(recency=s.w_recency, importance=s.w_importance, relevance=s.w_relevance,
                               decay=s.recency_decay)

    @property
    def sim(self) -> SimSettings:
        return self.config.sim

    def time(self) -> datetime:
        if self.now is None:
            raise RuntimeError("Mind.now is not set")
        return self.now

    def prefix(self, agent: "Agent") -> PromptPrefix:
        """Stable per-agent prompt prefix: rules + world (static) and the cached App. A summary."""
        summary = self.summaries.get(agent.name).text
        first = agent.identity.first_name
        background = "\n".join(f"- {phrase}" for phrase in agent.identity.bio)
        block = (f"You are simulating {agent.name}. This is who {first} is:\n{summary}\n\n"
                 f"{first}'s background (what {first} knew at the start of the simulation):\n{background}")
        return PromptPrefix(rules=self.rules, agent_block=block)

    def retrieve(
        self,
        agent: "Agent",
        query: str,
        *,
        k: int | None = None,
        types: Iterable[str] | None = None,
        exclude_subtypes: Iterable[str] = (),
        purpose: str = "",
    ) -> list[ScoredMemory]:
        return retrieve(agent.memory, query, self.time(), self.weights, k=k or self.sim.top_k, types=types,
                        exclude_subtypes=exclude_subtypes, purpose=purpose)

    def fallback(self, task: str, agent: "Agent | None", reason: str, used: Any) -> None:
        """Record that a deterministic fallback replaced a failed LLM result."""
        self.sink.emit("llm_fallback", {"task": task, "reason": reason, "used": str(used)[:300]},
                       agent=agent.name if agent else None)

    def known_places(self, agent: "Agent") -> list[str]:
        return agent.known_places(self.world.public_places())

    def _summary_updated(self, name: str, entry: Any) -> None:
        self.sink.emit("summary_updated", {"version": entry.version, "text": entry.text}, agent=name,
                       game_time=self.now.isoformat() if self.now else None)
