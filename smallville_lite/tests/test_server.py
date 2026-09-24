"""Web viewer API against a stub run."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from smallville_lite.sim.runner import start_run  # noqa: E402
from smallville_lite.web.server import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    start_run("party", budget=5.0, stub=True, hours=8, runs_dir=runs, run_id="v1", out=lambda s: None)
    return TestClient(create_app(runs))


def test_page_and_static(client):
    assert "Smallville Log" in client.get("/").text
    assert "reduceState" in client.get("/static/state.js").text


def test_runs_and_meta(client):
    runs = client.get("/api/runs").json()
    assert runs[0]["run_id"] == "v1" and runs[0]["status"] == "completed" and runs[0]["has_report"]
    meta = client.get("/api/runs/v1/meta").json()
    assert "raw_config" not in meta and meta["checkpoints"]


def test_events_paging_and_prompt_stripping(client):
    first = client.get("/api/runs/v1/events").json()
    assert first["events"] and first["last_seq"] == first["events"][-1]["seq"]
    assert all("prompt" not in e["data"] for e in first["events"] if e["type"] == "llm_call")
    mid = first["events"][len(first["events"]) // 2]["seq"]
    later = client.get(f"/api/runs/v1/events?after={mid}").json()["events"]
    assert later and all(e["seq"] > mid for e in later)
    full = client.get("/api/runs/v1/events?full=true").json()["events"]
    assert any("prompt" in e["data"] for e in full if e["type"] == "llm_call")    # stub logs prompts


def test_report_presets_and_interview(client):
    assert "facts" in client.get("/api/runs/v1/report").json()
    presets = client.get("/api/presets").json()
    assert sum(len(v) for v in presets.values()) == 25
    r = client.post("/api/runs/v1/interview", json={"agent": "Isabella Rodriguez", "question": "What's your occupation?"})
    body = r.json()
    assert r.status_code == 200 and body["stub"] is True and body["answer"]
    assert client.get("/api/runs/v1/interviews").json()[-1]["data"]["question"] == "What's your occupation?"
    assert client.post("/api/runs/v1/interview", json={"agent": "Nobody", "question": "hi"}).status_code == 400


def test_bad_run_ids(client):
    assert client.get("/api/runs/..%2F..%2Fetc/events").status_code in (400, 404)
    assert client.get("/api/runs/nope/events").status_code == 404


def test_layout_endpoint(client):
    layout = client.get("/api/runs/v1/layout").json()
    assert layout["source"] == "layout.toml" and len(layout["places"]) == 6


def _sse(text):
    events, cur = [], {}
    for line in text.splitlines():
        if not line:
            if cur:
                events.append(cur)
            cur = {}
        elif not line.startswith(":"):
            key, _, value = line.partition(": ")
            cur[key] = value
    return events


def test_stream_replays_then_ends_for_a_finished_run(client):
    all_events = client.get("/api/runs/v1/events").json()["events"]
    with client.stream("GET", "/api/runs/v1/stream") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    sse = _sse(body)
    logs = [e for e in sse if e.get("event") == "log"]
    assert [int(e["id"]) for e in logs] == [e["seq"] for e in all_events]
    assert sse[-1]["event"] == "end"


def test_stream_resumes_after_last_event_id(client):
    all_events = client.get("/api/runs/v1/events").json()["events"]
    cut = all_events[-5]["seq"]
    with client.stream("GET", "/api/runs/v1/stream", headers={"Last-Event-ID": str(cut)}) as r:
        logs = [e for e in _sse("".join(r.iter_text())) if e.get("event") == "log"]
    assert [int(e["id"]) for e in logs] == [e["seq"] for e in all_events[-4:]]
