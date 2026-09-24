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

from pydantic import BaseModel, ConfigDict


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


EVENT_MODELS: dict[str, type[BaseModel]] = {
    "llm_call": LLMCallData,
    "embed_call": EmbedCallData,
    "budget_warning": BudgetData,
    "budget_exhausted": BudgetData,
    "run_started": RunStartedData,
    "run_ended": RunEndedData,
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
            data = model.model_validate(data).model_dump(exclude_none=True)
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
