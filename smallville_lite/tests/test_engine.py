"""End-to-end stub runs of the party scenario through the tick engine."""

from __future__ import annotations

import json
from collections import Counter
from datetime import timedelta

import pytest

from smallville_lite.llm import BudgetExhausted
from smallville_lite.sim.events import EventLog
from smallville_lite.sim.scenario import build_simulation, load_scenario, scenario_config


def _run(tmp_path, name: str, hours: int = 24, **overrides):
    scenario = load_scenario("party")
    ov = {"run": {"stub": True}, "budget": {"max_usd": 100.0}}
    ov.update(overrides)
    cfg = scenario_config(scenario, overrides=ov)
    path = tmp_path / f"{name}.jsonl"
    with EventLog(path, name) as log:
        sim = build_simulation(scenario, cfg, log, embed_cache_path=None)
        sim.run(sim.clock.tick_at(scenario.start + timedelta(hours=hours)))
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return sim, events


def test_one_day_runs_end_to_end(tmp_path):
    sim, events = _run(tmp_path, "day")
    counts = Counter(e["type"] for e in events)
    for t in ("world_init", "agent_init", "plan_created", "action_started", "move_started", "move_arrived",
              "perceived", "react_decision", "conversation_started", "utterance", "conversation_ended",
              "reflection", "summary_updated", "state_snapshot", "time_skipped", "llm_call"):
        assert counts[t] > 0, t
    assert counts["agent_init"] == 4
    assert {e["agent"] for e in events if e["type"] == "action_started"} == set(sim.agents)
    # Seed memories: one per semicolon-delimited bio phrase (paper §3.1).
    for agent in sim.agents.values():
        assert len(agent.memory.of_type("observation", "seed")) == len(agent.identity.bio)
    assert counts["conversation_started"] == counts["conversation_ended"]
    # Every move eventually arrives.
    assert counts["move_started"] - counts["move_arrived"] <= len(sim.agents)
    # Sonnet calls hit the prompt cache once each agent's prefix is warm.
    assert sim.mind.llm.usage.by_model["claude-sonnet-5"].cache_read_tokens > 0


def test_news_spreads_in_stub(tmp_path):
    sim, _ = _run(tmp_path, "spread")
    heard = {name for name, a in sim.agents.items() if name != "Isabella Rodriguez"
             and any("valentine" in n.description.lower() for n in a.memory.of_type("observation", "dialogue"))}
    assert heard, "no one heard about the party"


def test_invitees_show_up_at_the_party_in_stub(tmp_path):
    sim, events = _run(tmp_path, "party", hours=48)
    # Commitments are deduplicated: one per agent for the party, however often it came up.
    for agent in sim.agents.values():
        keys = [(n.meta.get("date"), n.meta.get("time"), n.meta.get("place"))
                for n in agent.memory.of_type("plan", "commitment")]
        assert len(keys) == len(set(keys)), agent.name
    snap = next(e for e in events if e["type"] == "state_snapshot" and e["game_time"] == "2023-02-14T18:00:00")
    at_party = {n for n, a in snap["data"]["agents"].items() if a["place"] == "Hobbs Cafe" and not a["in_transit"]}
    committed = {a.name for a in sim.agents.values() if a.memory.of_type("plan", "commitment")}
    assert "Isabella Rodriguez" in at_party
    assert at_party & (committed - {"Isabella Rodriguez"}), (at_party, committed)


def test_deterministic(tmp_path):
    def fingerprint(events):
        return [(e["type"], e["agent"], e["tick"], json.dumps(e["data"], sort_keys=True))
                for e in events if e["type"] not in ("llm_call", "embed_call")]

    _, a = _run(tmp_path, "a", hours=10)
    _, b = _run(tmp_path, "b", hours=10)
    assert fingerprint(a) == fingerprint(b)


def test_conversations_claim_both_agents(tmp_path):
    sim, events = _run(tmp_path, "claims", hours=18)
    busy: dict[int, list[str]] = {}
    for e in events:
        if e["type"] == "conversation_started":
            for p in e["data"]["participants"]:
                assert p not in busy.get(e["tick"], []), "agent in two conversations in one tick"
                busy.setdefault(e["tick"], []).append(p)


def test_budget_stop_aborts_the_tick_cleanly(tmp_path):
    scenario = load_scenario("party")
    cfg = scenario_config(scenario, overrides={"run": {"stub": True}, "budget": {"max_usd": 0.15}})
    path = tmp_path / "budget.jsonl"
    with EventLog(path, "budget") as log:
        sim = build_simulation(scenario, cfg, log, embed_cache_path=None)
        with pytest.raises(BudgetExhausted):
            sim.run(sim.clock.tick_at(scenario.start + timedelta(hours=24)))
        stopped_at = sim.tick
        spent = sim.mind.llm.budget.spent
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert spent <= 0.15
    aborted = [e for e in events if e["tick"] == stopped_at]
    assert aborted and all(e["type"] in ("llm_call", "embed_call", "budget_exhausted", "budget_warning") for e in aborted)
    assert all(e["data"].get("aborted_tick") for e in aborted)


def test_party_prefix_is_long_enough_to_cache_on_sonnet(tmp_path):
    sim, _ = _run(tmp_path, "prefix", hours=0)
    sonnet = sim.mind.config.models["claude-sonnet-5"]
    for agent in sim.agents.values():
        prefix = sim.mind.prefix(agent)
        assert len(prefix.text) / sonnet.chars_per_token >= sonnet.min_cache_tokens, agent.name
