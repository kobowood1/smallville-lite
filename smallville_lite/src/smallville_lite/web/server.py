"""Web log viewer (DESIGN §8.3): a FastAPI app serving one static page and a small read-only JSON API.

Everything the page shows comes from ``events.jsonl`` (plus ``report.json`` and ``interviews.jsonl``).
The only write is an interview, which runs against a saved checkpoint and is logged to
``interviews.jsonl``; the run itself is never modified.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..cognition.interview import load_presets
from ..sim.checkpoint import list_checkpoints
from ..sim.events import filter_superseded
from ..sim.runner import RunPaths, interview_run, list_runs

STATIC = Path(__file__).with_name("static")
_RUN_ID = re.compile(r"^[A-Za-z0-9_.\-]+$")
_HEAVY = ("prompt", "response")


class InterviewRequest(BaseModel):
    agent: str
    question: str
    tick: int | None = None


class _EventCache:
    """Parsed events per run, extended incrementally as the file grows (a live run appends)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[Path, tuple[int, list[dict[str, Any]]]] = {}

    def get(self, path: Path) -> list[dict[str, Any]]:
        with self._lock:
            size = path.stat().st_size if path.exists() else 0
            offset, events = self._data.get(path, (0, []))
            if size < offset:
                offset, events = 0, []
            if size > offset:
                with path.open("rb") as fh:
                    fh.seek(offset)
                    chunk = fh.read(size - offset)
                # Only consume complete lines; a writer may be mid-line.
                end = chunk.rfind(b"\n") + 1
                for line in chunk[:end].decode("utf-8").splitlines():
                    if line.strip():
                        events.append(json.loads(line))
                offset += end
                self._data[path] = (offset, events)
            return events


def create_app(runs_dir: str | Path, *, stub_interviews: bool = False, interview_budget: float = 0.5) -> FastAPI:
    runs_root = Path(runs_dir)
    cache = _EventCache()
    session = {"spent": 0.0}
    app = FastAPI(title="smallville_lite viewer", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def run_paths(run_id: str) -> RunPaths:
        if not _RUN_ID.match(run_id):
            raise HTTPException(400, "bad run id")
        paths = RunPaths(runs_root / run_id)
        if not paths.config.exists():
            raise HTTPException(404, f"no run {run_id}")
        return paths

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        return list_runs(runs_root)

    @app.get("/api/runs/{run_id}/events")
    def events(run_id: str, after: int = 0, full: bool = False) -> dict[str, Any]:
        paths = run_paths(run_id)
        raw = cache.get(paths.events)
        kept = list(filter_superseded(raw))
        out = []
        for e in kept:
            if e["seq"] <= after:
                continue
            if not full and e["type"] == "llm_call" and any(k in e["data"] for k in _HEAVY):
                e = {**e, "data": {k: v for k, v in e["data"].items() if k not in _HEAVY}}
            out.append(e)
        last = max((e["seq"] for e in raw), default=0)
        return {"events": out, "last_seq": last}

    @app.get("/api/runs/{run_id}/meta")
    def meta(run_id: str) -> dict[str, Any]:
        paths = run_paths(run_id)
        data = json.loads(paths.config.read_text(encoding="utf-8"))
        data.pop("raw_config", None)
        data["checkpoints"] = [t for t, _ in list_checkpoints(paths.checkpoints)]
        return data

    @app.get("/api/runs/{run_id}/report")
    def report(run_id: str) -> dict[str, Any]:
        paths = run_paths(run_id)
        if not paths.report_json.exists():
            raise HTTPException(404, "no report yet")
        return json.loads(paths.report_json.read_text(encoding="utf-8"))

    @app.get("/api/runs/{run_id}/interviews")
    def interviews(run_id: str) -> list[dict[str, Any]]:
        paths = run_paths(run_id)
        return [e for e in cache.get(paths.interviews) if e["type"] == "interview"] if paths.interviews.exists() else []

    @app.post("/api/runs/{run_id}/interview")
    def ask(run_id: str, req: InterviewRequest) -> dict[str, Any]:
        paths = run_paths(run_id)
        run_meta = json.loads(paths.config.read_text(encoding="utf-8"))
        stub = True if stub_interviews or run_meta.get("stub") else None
        if not stub and not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise HTTPException(400, "this is a live run and ANTHROPIC_API_KEY is not set; restart with --stub-interviews")
        remaining = interview_budget - session["spent"]
        if not stub and remaining <= 0:
            raise HTTPException(402, f"viewer interview budget of ${interview_budget:.2f} is used up")
        try:
            result = interview_run(paths.root, req.agent, req.question, tick=req.tick, stub=stub,
                                   budget=max(remaining, 0.0) if not stub else 1.0)
        except KeyError as exc:
            raise HTTPException(400, str(exc)) from None
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from None
        if not stub:
            session["spent"] += result["cost_usd"]
        result["session_spent_usd"] = round(session["spent"], 6)
        result["stub"] = bool(stub)
        return result

    @app.get("/api/presets")
    def presets() -> dict[str, list[str]]:
        return load_presets()

    return app
