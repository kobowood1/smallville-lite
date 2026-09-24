"""The stable, cacheable prompt prefix (DESIGN §6.3) and the App. A summary cache.

Request layout::

    system[0]  rules + world   (static for the whole run)
    system[1]  agent summary   (changes only when the summary is regenerated)  <- cache breakpoint
    messages   task            (volatile: time, observations, memories, instructions)

Nothing volatile may go into the prefix: any byte change invalidates the cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class PromptPrefix:
    rules: str
    agent_block: str | None = None

    def system_blocks(self, cache_ttl: str = "5m") -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = [{"type": "text", "text": self.rules}]
        if self.agent_block:
            blocks.append({"type": "text", "text": self.agent_block})
        cache_control: dict[str, Any] = {"type": "ephemeral"}
        if cache_ttl == "1h":
            cache_control["ttl"] = "1h"
        blocks[-1] = {**blocks[-1], "cache_control": cache_control}
        return blocks

    @property
    def text(self) -> str:
        return self.rules + ("\n" + self.agent_block if self.agent_block else "")


@dataclass
class SummaryEntry:
    text: str
    version: int
    stale: bool = False
    stale_reason: str | None = None


class SummaryCache:
    """Holds each agent's cached summary; regenerates only when a configured event marks it stale.

    ``regenerate(agent) -> str`` is supplied by the cognition layer (Phase 3: three retrievals +
    one ``summary`` call). ``on_update(agent, entry)`` lets the caller log ``summary_updated``.
    """

    def __init__(
        self,
        regenerate: Callable[[str], str],
        refresh_on: Iterable[str] = ("reflection", "new_day"),
        on_update: Callable[[str, SummaryEntry], None] | None = None,
    ) -> None:
        self._regenerate = regenerate
        self.refresh_on = frozenset(refresh_on)
        self._on_update = on_update
        self._entries: dict[str, SummaryEntry] = {}

    def get(self, agent: str) -> SummaryEntry:
        entry = self._entries.get(agent)
        if entry is None or entry.stale:
            version = 1 if entry is None else entry.version + 1
            entry = SummaryEntry(text=self._regenerate(agent), version=version)
            self._entries[agent] = entry
            if self._on_update:
                self._on_update(agent, entry)
        return entry

    def notify(self, agent: str, event: str) -> bool:
        """Mark the summary stale if ``event`` is a refresh trigger. Returns True if it was marked."""
        if event not in self.refresh_on or agent not in self._entries:
            return False
        entry = self._entries[agent]
        entry.stale, entry.stale_reason = True, event
        return True

    def state(self) -> dict[str, dict[str, Any]]:
        return {a: {"text": e.text, "version": e.version, "stale": e.stale} for a, e in self._entries.items()}

    def load_state(self, state: dict[str, dict[str, Any]]) -> None:
        self._entries = {
            a: SummaryEntry(text=s["text"], version=s["version"], stale=s.get("stale", False))
            for a, s in state.items()
        }
