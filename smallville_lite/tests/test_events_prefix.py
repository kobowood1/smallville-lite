from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from smallville_lite.llm.prefix import PromptPrefix, SummaryCache
from smallville_lite.llm.prompts import load_prompt
from smallville_lite.sim.events import EventLog, read_events

LLM_CALL = {"call_id": "c1", "task": "importance", "model": "m", "attempt": 1, "ok": True, "cost_usd": 0.01}


def _lines(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def test_commit_and_abort_tick(tmp_path):
    path = tmp_path / "events.jsonl"
    with EventLog(path, "run1") as log:
        log.set_time(0, "2023-02-13T06:00:00")
        log.emit("perceived", {"observations": []}, agent="Sam Moore")
        log.commit_tick()
        log.set_time(1, "2023-02-13T06:10:00")
        log.emit("perceived", {"observations": []}, agent="Sam Moore")
        log.emit("llm_call", LLM_CALL)
        log.abort_tick()
    events = _lines(path)
    assert [e["type"] for e in events] == ["perceived", "llm_call"]
    assert events[1]["data"]["aborted_tick"] is True  # spent money is never dropped
    assert events[0]["tick"] == 0 and events[1]["tick"] == 1
    assert [e["seq"] for e in events] == [1, 3]


def test_known_event_types_are_validated(tmp_path):
    with EventLog(tmp_path / "e.jsonl", "r", autocommit=True) as log:
        with pytest.raises(ValidationError):
            log.emit("llm_call", {"task": "x"})


def test_read_events_drops_superseded_segment(tmp_path):
    path = tmp_path / "events.jsonl"
    with EventLog(path, "run1", autocommit=True) as log:
        for t in range(3):
            log.set_time(t, None)
            log.emit("tick", {"n": t})
            log.emit("llm_call", {**LLM_CALL, "call_id": f"a{t}"})
    with EventLog(path, "run1", segment=1, start_seq=6, autocommit=True) as log:
        log.set_time(2, None)
        log.emit("run_resumed", {"from_tick": 2, "segment": 1})
        log.emit("tick", {"n": 2})
    kept = list(read_events(path))
    ticks = [(e["segment"], e["data"]["n"]) for e in kept if e["type"] == "tick"]
    assert ticks == [(0, 0), (0, 1), (1, 2)]
    assert len([e for e in kept if e["type"] == "llm_call"]) == 3  # cost history is kept


def test_prefix_blocks():
    blocks = PromptPrefix("rules", "summary").system_blocks("1h")
    assert blocks == [
        {"type": "text", "text": "rules"},
        {"type": "text", "text": "summary", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
    ]
    assert PromptPrefix("rules").system_blocks()[0]["cache_control"] == {"type": "ephemeral"}


def test_summary_regenerates_only_on_configured_events():
    calls: list[str] = []
    updates: list[int] = []

    def regenerate(agent: str) -> str:
        calls.append(agent)
        return f"{agent} summary #{len(calls)}"

    cache = SummaryCache(regenerate, refresh_on=("reflection", "new_day"),
                         on_update=lambda a, e: updates.append(e.version))
    assert cache.get("Sam").text == "Sam summary #1"
    for _ in range(50):
        cache.get("Sam")                      # every call uses the cached text
    assert calls == ["Sam"]
    assert cache.notify("Sam", "observation") is False
    assert cache.get("Sam").version == 1
    assert cache.notify("Sam", "reflection") is True
    entry = cache.get("Sam")
    assert entry.version == 2 and entry.text == "Sam summary #2" and updates == [1, 2]

    restored = SummaryCache(regenerate)
    restored.load_state(cache.state())
    assert restored.get("Sam").text == "Sam summary #2" and len(calls) == 2


def test_prompt_template_render_is_strict():
    p = load_prompt("ping")
    assert p.ref == "ping@1"
    assert "[tok]" in p.render(model_label="m", token="tok")
    with pytest.raises(KeyError):
        p.render(model_label="m")
