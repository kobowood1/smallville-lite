"""JSONL event log: the single source of truth for viewers, reports, and replay (DESIGN §8).

Events are buffered per tick and flushed on ``commit_tick()``. ``abort_tick()`` discards the
tick's events *except* cost events (``llm_call``, ``embed_call``, budget events), which are
flushed with ``aborted_tick: true`` because that money was really spent.

Phase 2 defines the envelope, the writer, and the data models for cost events. Later phases
register models for the remaining event types in ``EVENT_MODELS``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class EventSink(Protocol):
    def emit(self, type: str, data: dict[str, Any], agent: str | None = None, game_time: str | None = None) -> None: ...


class NullSink:
    def emit(self, type: str, data: dict[str, Any], agent: str | None = None, game_time: str | None = None) -> None:
        return None


class ListSink:
    """Collects events in memory (tests, one-off scripts)."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, type: str, data: dict[str, Any], agent: str | None = None, game_time: str | None = None) -> None:
        self.events.append({"type": type, "agent": agent, "game_time": game_time, "data": data})

    def of_type(self, type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == type]


# ---------------------------------------------------------------------------
# Event data models (validated on write)


class _Data(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMCallData(_Data):
    call_id: str
    task: str
    model: str
    agent: str | None = None
    prompt_id: str | None = None
    attempt: int
    ok: bool
    stop_reason: str | None = None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    request_id: str | None = None
    fallback: str | None = None
    aborted_tick: bool = False
    prompt: dict[str, Any] | None = None
    response: str | None = None


class EmbedCallData(_Data):
    provider: str
    model: str
    kind: str
    n_texts: int
    n_cached: int
    tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    aborted_tick: bool = False


class BudgetData(_Data):
    spent_usd: float
    limit_usd: float
    next_estimate_usd: float | None = None
    context: str | None = None
    aborted_tick: bool = False


class RunStartedData(_Data):
    run_id: str
    scenario: str | None = None
    config_hash: str
    seed: int
    stub: bool
    models: dict[str, str]
    budget_usd: float
    config: dict[str, Any] | None = None


class RunEndedData(_Data):
    reason: Literal["completed", "budget", "error", "interrupted"]
    last_tick: int | None = None
    spent_usd: float
    detail: str | None = None


class RunResumedData(_Data):
    from_tick: int
    segment: int
    checkpoint: str | None = None
    budget_usd: float | None = None


class TimeSkippedData(_Data):
    from_tick: int
    to_tick: int
    reason: str


class WorldInitData(_Data):
    town: str
    places: list[dict[str, Any]]
    travel_ticks_default: int
    tick_minutes: int
    start: str
    layout: dict[str, Any] | None = None


class AgentInitData(_Data):
    age: int
    traits: str
    bio: list[str]
    home: str
    start: str
    known_places: list[str]


class StateSnapshotData(_Data):
    agents: dict[str, dict[str, Any]]
    object_states: dict[str, str]


class ActionStartedData(_Data):
    action_id: str
    description: str
    emoji: str
    label: str
    place: str
    object: str | None = None
    object_state: str | None = None
    start: str
    duration_min: int | None = None
    kind: Literal["planned", "reaction", "chat", "wait", "sleep", "travel"]
    brief: bool = False       # shorter than a tick; superseded within the same tick


class MoveStartedData(_Data):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    origin: str = Field(alias="from")
    to: str
    depart_tick: int
    arrive_tick: int


class MoveArrivedData(_Data):
    place: str


class ObjectStateChangedData(_Data):
    object_path: str
    state: str
    by: str | None = None


class MemoryAddedData(_Data):
    node: dict[str, Any]


class MemoryAccessedData(_Data):
    query: str
    purpose: str
    results: list[dict[str, Any]]


class PerceivedData(_Data):
    observations: list[dict[str, Any]]


class ReactDecisionData(_Data):
    observation_node_id: str
    react: bool
    kind: str
    reaction: str
    reason: str


class PlanCreatedData(_Data):
    level: Literal["day", "hour", "detail"]
    date: str
    items: list[dict[str, Any]]
    wake: str | None = None
    sleep: str | None = None
    block_id: str | None = None


class PlanRevisedData(_Data):
    reason: str
    cause: str | None = None
    from_time: str
    removed_ids: list[str]
    added: list[dict[str, Any]]


class ReflectionData(_Data):
    trigger_sum: float
    questions: list[str]
    insights: list[dict[str, Any]]


class SummaryUpdatedData(_Data):
    version: int
    text: str


class ConversationStartedData(_Data):
    conv_id: str
    participants: list[str]
    initiator: str
    place: str
    reason: str


class UtteranceData(_Data):
    conv_id: str
    turn: int
    speaker: str
    listener: str
    text: str
    end_conversation: bool
    retrieved_node_ids: list[str]


class ConversationEndedData(_Data):
    conv_id: str
    turns: int
    ended_by: str
    topic_label: str
    summary: str
    duration_minutes: int


class InterviewData(_Data):
    question: str
    answer: str
    cited_node_ids: list[str]
    source: str


class LLMFallbackData(_Data):
    task: str
    reason: str
    used: str


class ReportData(_Data):
    path: str
    facts: dict[str, int]
    density: dict[str, float | None]
    attendees: dict[str, list[str]]


EVENT_MODELS: dict[str, type[BaseModel]] = {
    "llm_call": LLMCallData,
    "embed_call": EmbedCallData,
    "budget_warning": BudgetData,
    "budget_exhausted": BudgetData,
    "run_started": RunStartedData,
    "run_ended": RunEndedData,
    "run_resumed": RunResumedData,
    "time_skipped": TimeSkippedData,
    "world_init": WorldInitData,
    "agent_init": AgentInitData,
    "state_snapshot": StateSnapshotData,
    "action_started": ActionStartedData,
    "move_started": MoveStartedData,
    "move_arrived": MoveArrivedData,
    "object_state_changed": ObjectStateChangedData,
    "memory_added": MemoryAddedData,
    "memory_accessed": MemoryAccessedData,
    "perceived": PerceivedData,
    "react_decision": ReactDecisionData,
    "plan_created": PlanCreatedData,
    "plan_revised": PlanRevisedData,
    "reflection": ReflectionData,
    "summary_updated": SummaryUpdatedData,
    "conversation_started": ConversationStartedData,
    "utterance": UtteranceData,
    "conversation_ended": ConversationEndedData,
    "interview": InterviewData,
    "llm_fallback": LLMFallbackData,
    "report": ReportData,
}

COST_EVENT_TYPES = frozenset({"llm_call", "embed_call", "budget_warning", "budget_exhausted"})


class EventLog:
    """Append-only JSONL writer with a per-tick buffer.

    ``autocommit=True`` writes every event immediately (used outside the tick loop, e.g. the
    CLI demo or interviews against a saved checkpoint).
    """

    def __init__(
        self,
        path: str | Path,
        run_id: str,
        *,
        segment: int = 0,
        start_seq: int = 0,
        autocommit: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.segment = segment
        self.autocommit = autocommit
        self._seq = start_seq
        self._tick: int | None = None
        self._game_time: str | None = None
        self._buffer: list[dict[str, Any]] = []
        self._fh = self.path.open("a", encoding="utf-8")

    # -- clock -------------------------------------------------------------
    def set_time(self, tick: int | None, game_time: datetime | str | None) -> None:
        self._tick = tick
        self._game_time = game_time.isoformat() if isinstance(game_time, datetime) else game_time

    # -- writing -----------------------------------------------------------
    def emit(self, type: str, data: dict[str, Any], agent: str | None = None, game_time: str | None = None) -> None:
        model = EVENT_MODELS.get(type)
        if model is not None:
            data = model.model_validate(data).model_dump(exclude_none=True, by_alias=True)
        self._seq += 1
        event = {
            "seq": self._seq,
            "run_id": self.run_id,
            "segment": self.segment,
            "tick": self._tick,
            "game_time": game_time or self._game_time,
            "wall_time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "type": type,
            "agent": agent,
            "data": data,
        }
        if self.autocommit:
            self._write([event])
        else:
            self._buffer.append(event)

    def commit_tick(self) -> None:
        self._write(self._buffer)
        self._buffer = []

    def abort_tick(self) -> None:
        kept = []
        for event in self._buffer:
            if event["type"] in COST_EVENT_TYPES:
                event["data"] = {**event["data"], "aborted_tick": True}
                kept.append(event)
        self._write(kept)
        self._buffer = []

    def close(self) -> None:
        if self._buffer:
            self.commit_tick()
        self._fh.close()

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def seq(self) -> int:
        return self._seq

    def _write(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            self._fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._fh.flush()


def read_events(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield events in order, dropping ones superseded by a later resume segment.

    When segment N+1 resumes from tick T, events from earlier segments with tick >= T are
    stale (their tick was aborted or re-run) and are skipped.
    """
    events = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    yield from filter_superseded(events)


def filter_superseded(events: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Drop non-cost events of earlier segments at or after a later segment's resume tick."""
    resume_from: dict[int, int] = {}
    for e in events:
        if e["type"] == "run_resumed":
            resume_from[e["segment"]] = e["data"]["from_tick"]
    for e in events:
        stale = any(
            e["segment"] < seg and e["tick"] is not None and e["tick"] >= from_tick
            and e["type"] not in COST_EVENT_TYPES
            for seg, from_tick in resume_from.items()
        )
        if not stale:
            yield e
