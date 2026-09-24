"""Post-run report metrics (paper §7.1)."""

from __future__ import annotations

from datetime import datetime

import pytest

from smallville_lite.report import attendees, build_report, density, presence_intervals, render_markdown
from smallville_lite.sim.events import EventLog, read_events
from smallville_lite.sim.scenario import build_simulation, load_scenario, scenario_config

T = lambda h, m=0: datetime(2023, 2, 14, h, m)  # noqa: E731


def test_density_matches_the_paper_formula():
    agents = ["A", "B", "C", "D"]
    knows = {("A", "B"): True, ("B", "A"): True, ("A", "C"): True, ("C", "A"): False, ("C", "D"): True, ("D", "C"): True}
    edges, eta = density(agents, knows)
    assert edges == [("A", "B"), ("C", "D")]           # mutual only
    assert eta == pytest.approx(2 * 2 / (4 * 3))       # 2|E| / (|V|(|V|-1))
    assert density(agents, {})[1] == 0.0


def test_presence_intervals_and_attendance():
    ev = lambda typ, agent, t, **data: {"type": typ, "agent": agent, "game_time": t.isoformat(), "data": data}  # noqa: E731
    events = [
        {"type": "world_init", "agent": None, "game_time": T(6).isoformat(), "data": {"start": T(6).isoformat()}},
        ev("agent_init", "Ann", T(6), start="Home/bedroom"),
        ev("agent_init", "Bob", T(6), start="Cafe"),
        ev("move_started", "Ann", T(16, 50), **{"from": "Home", "to": "Cafe"}),
        ev("move_arrived", "Ann", T(17, 0), place="Cafe"),
        ev("move_started", "Ann", T(18, 0), **{"from": "Cafe", "to": "Home"}),
        ev("move_arrived", "Ann", T(18, 10), place="Home"),
        ev("move_started", "Bob", T(17, 5), **{"from": "Cafe", "to": "Park"}),
        ev("move_arrived", "Bob", T(17, 15), place="Park"),
    ]
    intervals = presence_intervals(events, lambda p: p.split("/")[0], T(23))
    assert ("Cafe", T(17), T(18)) in intervals["Ann"]
    present = attendees(intervals, "Cafe", T(17), T(19), min_minutes=10)
    assert present == {"Ann": 60}                      # Bob left after 5 minutes


@pytest.fixture
def fresh_sim(tmp_path):
    scenario = load_scenario("party")
    cfg = scenario_config(scenario, overrides={"run": {"stub": True}, "budget": {"max_usd": 100.0}})
    log = EventLog(tmp_path / "events.jsonl", "r")
    sim = build_simulation(scenario, cfg, log, embed_cache_path=None)
    yield sim, scenario, tmp_path / "events.jsonl"
    log.close()


def test_ungrounded_yes_is_flagged_as_hallucination(fresh_sim):
    sim, scenario, path = fresh_sim
    sim.mind.llm.backend.register("judge", lambda req, rng: {"knows": True, "reason": "claims to know"})
    sim.log.commit_tick()
    report = build_report(sim, scenario, list(read_events(path)), baseline=None, use_llm=True)
    party = report["facts"]["party"]["agents"]
    assert party["Isabella Rodriguez"]["grounded"] and not party["Isabella Rodriguez"]["hallucinated"]
    for name in ("Sam Moore", "Maria Lopez", "Klaus Mueller"):     # nobody has heard yet at tick 0
        assert party[name]["claims"] is True and party[name]["hallucinated"] is True
    assert report["relationships"]["start"] is None and report["relationships"]["end"]["density"] == 1.0
    assert "Hallucinated" in render_markdown(report)


def test_report_without_llm_uses_memory_evidence(fresh_sim):
    sim, scenario, path = fresh_sim
    sim.log.commit_tick()
    calls = len(sim.mind.llm.backend.requests)
    report = build_report(sim, scenario, list(read_events(path)), baseline=None, use_llm=False)
    assert len(sim.mind.llm.backend.requests) == calls                # no new calls
    assert report["facts"]["mayor"]["knew"] == 1                       # only Sam, from his seed memory
    assert report["coordination"]["party"]["happened"] is False
