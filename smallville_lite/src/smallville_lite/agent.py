"""Agent = identity + memory stream + short-term state (plans, current action, cooldowns)."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Iterable

from .memory import MemoryStream

_SLEEP = re.compile(r"\b(sleep|sleeping|asleep|nap|napping|bed(time)?)\b", re.I)


def is_sleep_activity(text: str) -> bool:
    return bool(_SLEEP.search(text))


@dataclass
class Identity:
    name: str
    age: int
    traits: str
    home: str                     # place name
    home_area: str | None         # area path inside home, e.g. "Hobbs Cafe/Isabella's apartment"
    start: str                    # place or area path
    bio: list[str]                # semicolon-delimited phrases (paper §3.1)
    extra_places: list[str] = field(default_factory=list)
    wake_hint: str = ""

    @property
    def first_name(self) -> str:
        return self.name.split()[0]


@dataclass
class Step:
    """A 5-15 minute action (JIT decomposition of an hour block)."""

    id: str
    start: datetime
    minutes: int
    activity: str
    object: str | None = None        # object path
    object_state: str | None = None
    emoji: str = "🙂"
    label: str = ""

    @property
    def end(self) -> datetime:
        return self.start + timedelta(minutes=self.minutes)


@dataclass
class HourBlock:
    id: str
    start: datetime
    end: datetime
    activity: str
    place: str
    source: str = "plan"              # plan | replan
    steps: list[Step] = field(default_factory=list)
    decomposed_until: datetime | None = None
    emoji: str = "🙂"
    label: str = ""

    def covers(self, t: datetime) -> bool:
        return self.start <= t < self.end

    @property
    def is_sleep(self) -> bool:
        return is_sleep_activity(self.activity)


@dataclass
class DayPlan:
    day: date
    wake: datetime
    sleep: datetime
    items: list[tuple[datetime, str]]
    blocks: list[HourBlock]
    revision: int = 0

    def block_at(self, t: datetime) -> HourBlock | None:
        for b in self.blocks:
            if b.covers(t):
                return b
        return None


@dataclass
class CurrentAction:
    """What the agent is doing now, resolved from its plan (or a chat / sleep)."""

    id: str
    activity: str
    place: str
    kind: str                          # planned | chat | sleep
    object: str | None = None
    object_state: str | None = None
    emoji: str = "🙂"
    label: str = ""
    start: datetime | None = None
    minutes: int | None = None


@dataclass
class Scratch:
    day_plan: DayPlan | None = None
    last_recap: str | None = None
    action: CurrentAction | None = None
    engaged_until: datetime | None = None        # busy in a conversation
    engaged_label: str | None = None
    cooldowns: dict[str, datetime] = field(default_factory=dict)   # other agent -> no new chat before
    recent_observations: deque[tuple[str, str]] = field(default_factory=lambda: deque(maxlen=10))
    emitted_until: datetime | None = None        # action_started events emitted up to this time
    visited: set[str] = field(default_factory=set)
    counter: int = 0

    def next_id(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}{self.counter}"


class Agent:
    def __init__(self, identity: Identity, memory: MemoryStream, retention: int = 10) -> None:
        self.identity = identity
        self.memory = memory
        self.scratch = Scratch(recent_observations=deque(maxlen=retention))

    @property
    def name(self) -> str:
        return self.identity.name

    def known_places(self, public_places: Iterable[str]) -> list[str]:
        """Public places + home + listed extras + visited, in stable order (paper §5.1 subgraph)."""
        ordered: list[str] = []
        for p in [self.identity.home, *public_places, *self.identity.extra_places, *sorted(self.scratch.visited)]:
            if p not in ordered:
                ordered.append(p)
        return ordered

    def is_engaged(self, now: datetime) -> bool:
        return self.scratch.engaged_until is not None and now < self.scratch.engaged_until

    def is_asleep(self, now: datetime) -> bool:
        plan = self.scratch.day_plan
        if plan is None or plan.day != now.date():
            return now.hour < 5  # before the day's plan exists, treat early hours as night
        if now < plan.wake or now >= plan.sleep:
            return True
        block = plan.block_at(now)
        return block is not None and block.is_sleep

    def debug_state(self) -> dict[str, Any]:
        a = self.scratch.action
        return {"name": self.name, "action": a.activity if a else None, "place": a.place if a else None}
