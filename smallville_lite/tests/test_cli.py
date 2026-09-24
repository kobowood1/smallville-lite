"""CLI: run, resume, report, interview (all in stub mode)."""

from __future__ import annotations

import json

from smallville_lite.cli import main
from smallville_lite.sim.events import read_events


def _events(run_dir):
    return list(read_events(run_dir / "events.jsonl"))


def test_run_resume_interview_report(tmp_path, capsys):
    runs = tmp_path / "runs"
    assert main(["run", "--stub", "--hours", "10", "--budget", "5", "--runs-dir", str(runs), "--run-id", "r1"]) == 0
    run = runs / "r1"
    for name in ("events.jsonl", "config.resolved.json", "baseline.json", "report.json", "report.md"):
        assert (run / name).exists(), name
    events = _events(run)
    assert events[-1]["type"] == "run_ended" and events[-1]["data"]["reason"] == "completed"
    assert any(e["type"] == "report" for e in events)
    baseline = json.loads((run / "baseline.json").read_text(encoding="utf-8"))
    assert len(baseline) == 12                                     # every ordered pair of 4 agents
    assert (run / "checkpoints" / "latest").exists()

    assert main(["resume", str(run), "--hours", "14"]) == 0
    events = _events(run)
    assert [e["type"] for e in events].count("run_resumed") == 1
    assert events[-1]["type"] == "run_ended" and events[-1]["tick"] == 84   # 14 h of 10-minute ticks
    resumed = next(e for e in events if e["type"] == "run_resumed")
    assert resumed["segment"] == 1 and resumed["data"]["from_tick"] == 60

    capsys.readouterr()
    assert main(["interview", str(run), "--agent", "Sam Moore", "--tick", "30", "Who is running for mayor?"]) == 0
    out = capsys.readouterr().out
    assert "Sam Moore:" in out and "tick_000030.json" in out
    assert (run / "interviews.jsonl").exists()
    # Interviews never touch the run's own log.
    assert _events(run)[-1]["type"] == "run_ended"

    assert main(["report", str(run), "--no-llm"]) == 0
    assert json.loads((run / "report.json").read_text(encoding="utf-8"))["interviews"] is False


def test_budget_stop_is_clean(tmp_path):
    runs = tmp_path / "runs"
    assert main(["run", "--stub", "--hours", "10", "--budget", "0.30", "--runs-dir", str(runs), "--run-id", "b"]) == 0
    events = _events(runs / "b")
    ended = events[-1]
    assert ended["type"] == "run_ended" and ended["data"]["reason"] == "budget"
    assert ended["data"]["spent_usd"] <= 0.30
    report = json.loads((runs / "b" / "report.json").read_text(encoding="utf-8"))
    assert report["interviews"] is False                          # partial report, no new calls


def test_missing_key_refuses_live_run(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert main(["run", "--hours", "1", "--budget", "1", "--runs-dir", str(tmp_path)]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())
