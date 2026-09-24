"""Run lifecycle: start, resume, report, and checkpoint-backed interviews (DESIGN §8.4).

A run lives in ``runs/<run_id>/``::

    events.jsonl          the single source of truth (append-only; one segment per start/resume)
    config.resolved.json  scenario, fully resolved config, end tick, budget
    checkpoints/          tick_<n>.json after every committed tick (+ ``latest``)
    baseline.json         start-of-run "Do you know of X?" survey (paper §7.1)
    report.json / .md     post-run report
    interviews.jsonl      ad-hoc interviews from the CLI or web viewer (never touch the run itself)

The whole run, report included, stays under the ``--budget`` cap: the simulation runs under
``budget - [report].reserve_usd`` and the report may use what is left.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from ..config import Config, deep_merge, parse_config
from ..llm import BudgetExhausted, BudgetGuard
from ..llm.usage import UsageSummary
from ..report import build_report, relationship_survey, render_markdown
from .checkpoint import apply_checkpoint, checkpoint_at_or_before, latest_checkpoint, load_checkpoint, save_checkpoint
from .events import EventLog, read_events
from .scenario import Scenario, build_simulation, load_scenario, scenario_config, wire_simulation

Out = Callable[[str], None]


@dataclass
class RunPaths:
    root: Path

    @property
    def events(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def config(self) -> Path:
        return self.root / "config.resolved.json"

    @property
    def baseline(self) -> Path:
        return self.root / "baseline.json"

    @property
    def report_json(self) -> Path:
        return self.root / "report.json"

    @property
    def report_md(self) -> Path:
        return self.root / "report.md"

    @property
    def interviews(self) -> Path:
        return self.root / "interviews.jsonl"


@dataclass
class RunResult:
    run_dir: Path
    run_id: str
    reason: str
    last_tick: int
    spent_usd: float
    report: dict[str, Any] | None = None


def _raw(config: Config, section: str, key: str, default: Any) -> Any:
    return config.raw.get(section, {}).get(key, default)


def _embed_cache(config: Config) -> str | None:
    # Stub vectors are free to recompute; real runs share the on-disk cache.
    return None if config.stub else "config"


# ---------------------------------------------------------------------------
# start / resume

def start_run(
    scenario_name: str,
    *,
    budget: float,
    stub: bool = False,
    days: float | None = None,
    hours: float | None = None,
    runs_dir: str | Path = "runs",
    run_id: str | None = None,
    make_report: bool = True,
    config_dir: str | Path | None = None,
    pace: float = 0.0,
    out: Out = print,
) -> RunResult:
    scenario = load_scenario(scenario_name)
    config = scenario_config(scenario, config_dir, overrides={"run": {"stub": stub}, "budget": {"max_usd": budget}})
    span = timedelta(hours=hours) if hours is not None else timedelta(days=days if days is not None else scenario.days)
    run_id = run_id or f"{scenario.name}-{datetime.now():%Y%m%d-%H%M%S}"
    paths = RunPaths(Path(runs_dir) / run_id)
    if paths.root.exists():
        raise FileExistsError(f"run directory already exists: {paths.root}")
    paths.root.mkdir(parents=True)
    end_time = scenario.start + span

    log = EventLog(paths.events, run_id)
    sim = build_simulation(scenario, config, log, embed_cache_path=_embed_cache(config), call_id_prefix="s0-c")
    end_tick = sim.clock.tick_at(end_time)
    _write_meta(paths, {
        "run_id": run_id, "scenario": scenario.name, "scenario_dir": str(scenario.directory.resolve()),
        "raw_config": config.raw, "stub": config.stub, "start": scenario.start.isoformat(),
        "end_tick": end_tick, "end_time": end_time.isoformat(), "budget_usd": budget,
        "created": datetime.now().isoformat(timespec="seconds"),
    })
    log.set_time(None, scenario.start)
    log.emit("run_started", {
        "run_id": run_id, "scenario": scenario.name, "config_hash": config.config_hash, "seed": config.seed,
        "stub": config.stub, "models": {k: v.model for k, v in sorted(config.tasks.items())},
        "budget_usd": budget, "config": config.raw,
    })
    log.commit_tick()
    out(f"run {run_id}: {scenario.name}, {len(sim.agents)} agents, until {end_time:%Y-%m-%d %H:%M} "
        f"({'STUB' if config.stub else 'live'}, budget ${budget:.2f})")

    guard = sim.mind.llm.budget
    reserve = float(_raw(config, "report", "reserve_usd", 0.40)) if make_report else 0.0
    guard.max_usd = max(0.0, budget - reserve)
    baseline = None
    if make_report:
        try:
            out("baseline survey: 'Do you know of X?' for every pair ...")
            baseline = relationship_survey(sim, "report-start")
            paths.baseline.write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")
        except BudgetExhausted as exc:
            out(f"baseline survey skipped: {exc}")
        log.commit_tick()
    _checkpoint(sim, config, paths)
    return _execute(sim, scenario, config, paths, log, end_tick, budget, make_report, baseline, run_id, out, pace)


def resume_run(
    run_dir: str | Path,
    *,
    budget: float | None = None,
    days: float | None = None,
    hours: float | None = None,
    make_report: bool = True,
    pace: float = 0.0,
    out: Out = print,
) -> RunResult:
    paths = RunPaths(Path(run_dir))
    meta = _read_meta(paths)
    scenario = load_scenario(meta["scenario_dir"])
    budget = float(budget if budget is not None else meta["budget_usd"])
    config = parse_config(deep_merge(meta["raw_config"], {"budget": {"max_usd": budget}}))
    ckpt_path = latest_checkpoint(paths.checkpoints)
    if ckpt_path is None:
        raise FileNotFoundError(f"no checkpoint in {paths.checkpoints}")
    data = load_checkpoint(ckpt_path)

    last_seq, last_segment = 0, 0
    for e in _raw_events(paths.events):
        last_seq, last_segment = max(last_seq, e["seq"]), max(last_segment, e["segment"])
    segment = last_segment + 1
    log = EventLog(paths.events, meta["run_id"], segment=segment, start_seq=last_seq)
    sim = wire_simulation(scenario, config, log, embed_cache_path=_embed_cache(config), call_id_prefix=f"s{segment}-c")
    apply_checkpoint(sim, data)
    # Spend comes from the event log, not the checkpoint: calls made in an aborted tick were paid for too.
    sim.mind.llm.budget.spent = UsageSummary.from_log(paths.events).cost_usd

    end_tick = meta["end_tick"]
    if hours is not None or days is not None:
        span = timedelta(hours=hours) if hours is not None else timedelta(days=days or 0)
        end_tick = sim.clock.tick_at(datetime.fromisoformat(meta["start"]) + span)
    meta.update(end_tick=end_tick, end_time=sim.clock.time_at(end_tick).isoformat(), budget_usd=budget)
    _write_meta(paths, meta)

    log.set_time(sim.tick, sim.clock.time_at(sim.tick))
    log.emit("run_resumed", {"from_tick": sim.tick, "segment": segment, "checkpoint": ckpt_path.name, "budget_usd": budget})
    log.commit_tick()
    out(f"resumed {meta['run_id']} at tick {sim.tick} (segment {segment}); spent so far ${sim.mind.llm.budget.spent:.4f} "
        f"of ${budget:.2f}; running to tick {end_tick}")
    reserve = float(_raw(config, "report", "reserve_usd", 0.40)) if make_report else 0.0
    sim.mind.llm.budget.max_usd = max(0.0, budget - reserve)
    baseline = json.loads(paths.baseline.read_text(encoding="utf-8")) if paths.baseline.exists() else None
    return _execute(sim, scenario, config, paths, log, end_tick, budget, make_report, baseline, meta["run_id"], out, pace)


def _execute(sim, scenario: Scenario, config: Config, paths: RunPaths, log: EventLog, end_tick: int, budget: float,
             make_report: bool, baseline: dict[str, Any] | None, run_id: str, out: Out,
             pace: float = 0.0) -> RunResult:
    def on_commit(s) -> None:
        _checkpoint(s, config, paths)
        if pace > 0:
            time.sleep(pace)  # wall-clock pacing, so a stub run can be watched live in the viewer

    sim.on_commit = on_commit
    reason, detail = "completed", None
    try:
        sim.run(end_tick)
    except BudgetExhausted as exc:
        reason, detail = "budget", str(exc)
        out(f"budget reached at tick {sim.tick}: {exc}")
    except KeyboardInterrupt:
        log.abort_tick()
        reason = "interrupted"
        out(f"interrupted at tick {sim.tick}; resume with: smallville resume {paths.root}")
    except Exception:
        log.abort_tick()
        reason, detail = "error", traceback.format_exc()
        out(detail)

    guard = sim.mind.llm.budget
    report = None
    if make_report and reason in ("completed", "budget"):
        guard.max_usd = budget
        report = write_report(sim, scenario, paths, baseline, use_llm=reason == "completed",
                              run_info={"run_id": run_id, "reason": reason, "last_tick": sim.tick}, out=out)
    log.set_time(sim.tick, sim.clock.time_at(sim.tick))
    log.emit("run_ended", {"reason": reason, "last_tick": sim.tick, "spent_usd": round(guard.spent, 6),
                           "detail": (detail or "")[-4000:] or None})
    log.close()
    out("")
    out(sim.mind.llm.usage.render())
    out(f"\nrun directory: {paths.root}")
    return RunResult(paths.root, run_id, reason, sim.tick, guard.spent, report)


def write_report(sim, scenario: Scenario, paths: RunPaths, baseline: dict[str, Any] | None, *, use_llm: bool,
                 run_info: dict[str, Any], out: Out = print) -> dict[str, Any]:
    log = sim.log
    log.set_time(sim.tick, sim.clock.time_at(sim.tick))
    out("writing report" + (" (interviews + judge)" if use_llm else " (memory evidence only)") + " ...")
    events = list(read_events(paths.events))
    report = build_report(sim, scenario, events, baseline=baseline, use_llm=use_llm,
                          run_info={**run_info, "spent_usd": sim.mind.llm.budget.spent})
    paths.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    paths.report_md.write_text(render_markdown(report), encoding="utf-8")
    log.emit("report", {
        "path": paths.report_json.name,
        "facts": {k: v["knew"] for k, v in report["facts"].items()},
        "density": {k: (v or {}).get("density") for k, v in report["relationships"].items()},
        "attendees": {k: sorted(v["attendees"]) for k, v in report["coordination"].items()},
    })
    log.commit_tick()
    return report


def _checkpoint(sim, config: Config, paths: RunPaths) -> None:
    save_checkpoint(sim, paths.checkpoints, keep=int(_raw(config, "run", "checkpoint_keep", 5)),
                    keep_every=int(_raw(config, "run", "checkpoint_keep_every", 6)))


# ---------------------------------------------------------------------------
# read-only access for interviews and re-reports

def open_run(
    run_dir: str | Path,
    *,
    tick: int | None = None,
    stub: bool | None = None,
    budget: float = 0.50,
    log: EventLog | None = None,
):
    """Restore the run's state at the latest checkpoint at or before ``tick`` into a throwaway simulation.

    Events (interviews, their LLM calls) go to ``interviews.jsonl`` by default, never to ``events.jsonl``.
    """
    paths = RunPaths(Path(run_dir))
    meta = _read_meta(paths)
    scenario = load_scenario(meta["scenario_dir"])
    raw = deep_merge(meta["raw_config"], {"budget": {"max_usd": budget}})
    if stub is not None:
        raw = deep_merge(raw, {"run": {"stub": stub}})
    config = parse_config(raw)
    ckpt = checkpoint_at_or_before(paths.checkpoints, tick) if tick is not None else latest_checkpoint(paths.checkpoints)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint at or before tick {tick} in {paths.checkpoints}")
    if log is None:
        start_seq = sum(1 for _ in _raw_events(paths.interviews))
        log = EventLog(paths.interviews, meta["run_id"], segment=-1, start_seq=start_seq, autocommit=True)
    sim = wire_simulation(scenario, config, log, embed_cache_path=_embed_cache(config), call_id_prefix="iv-")
    apply_checkpoint(sim, load_checkpoint(ckpt))
    sim.mind.llm.budget = BudgetGuard(budget, sink=log)
    log.set_time(sim.tick, sim.clock.time_at(sim.tick))
    return sim, scenario, paths, ckpt


def interview_run(run_dir: str | Path, agent: str, question: str, *, tick: int | None = None,
                  stub: bool | None = None, budget: float = 0.50) -> dict[str, Any]:
    from ..cognition.interview import interview

    sim, _, _, ckpt = open_run(run_dir, tick=tick, stub=stub, budget=budget)
    try:
        if agent not in sim.agents:
            raise KeyError(f"unknown agent {agent!r}; choose from {sorted(sim.agents)}")
        result = interview(sim.mind, sim.agents[agent], question, sim.mind.time(), remember=False, source="interview")
        cited = [{"node_id": nid, "text": sim.agents[agent].memory.get(nid).description} for nid in result.cited_node_ids]
        return {"agent": agent, "question": question, "answer": result.answer, "cited": cited, "ok": result.ok,
                "tick": sim.tick, "game_time": sim.mind.time().isoformat(), "checkpoint": ckpt.name,
                "cost_usd": round(sim.mind.llm.budget.spent, 6)}
    finally:
        sim.log.close()


def regenerate_report(run_dir: str | Path, *, use_llm: bool = True, budget: float = 0.50, out: Out = print) -> dict[str, Any]:
    paths = RunPaths(Path(run_dir))
    events_log = EventLog(paths.interviews, "report", segment=-1,
                          start_seq=sum(1 for _ in _raw_events(paths.interviews)), autocommit=True)
    sim, scenario, _, _ = open_run(run_dir, budget=budget, log=events_log)
    baseline = json.loads(paths.baseline.read_text(encoding="utf-8")) if paths.baseline.exists() else None
    try:
        events = list(read_events(paths.events))
        report = build_report(sim, scenario, events, baseline=baseline, use_llm=use_llm,
                              run_info={"run_id": _read_meta(paths)["run_id"], "reason": "regenerated",
                                        "spent_usd": sim.mind.llm.budget.spent})
        paths.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        paths.report_md.write_text(render_markdown(report), encoding="utf-8")
        out(f"report written: {paths.report_md}")
        return report
    finally:
        events_log.close()


# ---------------------------------------------------------------------------
# helpers

def _write_meta(paths: RunPaths, meta: dict[str, Any]) -> None:
    paths.config.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _read_meta(paths: RunPaths) -> dict[str, Any]:
    if not paths.config.exists():
        raise FileNotFoundError(f"not a run directory (missing {paths.config.name}): {paths.root}")
    return json.loads(paths.config.read_text(encoding="utf-8"))


def _raw_events(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def list_runs(runs_dir: str | Path) -> list[dict[str, Any]]:
    """Summaries for the viewer: status comes from the last run_started/run_resumed/run_ended event."""
    out = []
    root = Path(runs_dir)
    if not root.exists():
        return out
    for d in sorted(root.iterdir(), reverse=True):
        paths = RunPaths(d)
        if not paths.config.exists() or not paths.events.exists():
            continue
        meta = _read_meta(paths)
        status, spent, last_tick = "running", 0.0, 0
        for e in _raw_events(paths.events):
            if e["type"] in ("run_started", "run_resumed"):
                status = "running"
            elif e["type"] == "run_ended":
                status, spent, last_tick = e["data"]["reason"], e["data"]["spent_usd"], e["data"].get("last_tick") or 0
            elif e["type"] in ("llm_call", "embed_call"):
                spent_live = e["data"].get("cost_usd", 0.0)
                if status == "running":
                    spent += spent_live
            if e.get("tick") is not None and status == "running":
                last_tick = max(last_tick, e["tick"])
        out.append({"run_id": meta["run_id"], "scenario": meta["scenario"], "stub": meta["stub"], "status": status,
                    "spent_usd": round(spent, 4), "budget_usd": meta["budget_usd"], "last_tick": last_tick,
                    "end_tick": meta["end_tick"], "created": meta.get("created"),
                    "has_report": paths.report_json.exists()})
    return out
