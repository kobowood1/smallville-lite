"""Checkpoint round trips and resume equivalence."""

from __future__ import annotations

import json

from smallville_lite.sim.checkpoint import (apply_checkpoint, latest_checkpoint, list_checkpoints, load_checkpoint,
                                            save_checkpoint, snapshot)
from smallville_lite.sim.events import EventLog
from smallville_lite.sim.scenario import build_simulation, load_scenario, scenario_config, wire_simulation

IGNORED = {"llm_call", "embed_call", "run_resumed"}


def _cfg(scenario, **ov):
    return scenario_config(scenario, overrides={"run": {"stub": True}, "budget": {"max_usd": 100.0}, **ov})


def _fingerprint(path, from_tick):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e["type"] in IGNORED or e["tick"] is None or e["tick"] < from_tick:
            continue
        out.append((e["type"], e["agent"], e["tick"], e["game_time"], json.dumps(e["data"], sort_keys=True)))
    return out


def test_round_trip_restores_everything(tmp_path):
    scenario = load_scenario("party")
    cfg = _cfg(scenario)
    with EventLog(tmp_path / "a.jsonl", "a") as log:
        sim = build_simulation(scenario, cfg, log, embed_cache_path=None)
        sim.run(40)
        path = save_checkpoint(sim, tmp_path / "ck")
        before = snapshot(sim)
    with EventLog(tmp_path / "b.jsonl", "b") as log:
        restored = wire_simulation(scenario, cfg, log, embed_cache_path=None)
        apply_checkpoint(restored, load_checkpoint(path))
        after = snapshot(restored)
    before["log"] = after["log"] = None
    assert after == before


def test_resume_is_equivalent_to_an_uninterrupted_run(tmp_path):
    scenario = load_scenario("party")
    cfg = _cfg(scenario)
    straight = tmp_path / "straight.jsonl"
    with EventLog(straight, "s") as log:
        sim = build_simulation(scenario, cfg, log, embed_cache_path=None)
        sim.run(60)

    split = tmp_path / "split.jsonl"
    with EventLog(split, "p") as log:
        sim = build_simulation(scenario, cfg, log, embed_cache_path=None)
        sim.run(30)
        ckpt = save_checkpoint(sim, tmp_path / "ck")
        seq = log.seq
    with EventLog(split, "p", segment=1, start_seq=seq) as log:
        sim = wire_simulation(scenario, cfg, log, embed_cache_path=None, call_id_prefix="s1-c")
        apply_checkpoint(sim, load_checkpoint(ckpt))
        assert sim.tick == 30
        sim.run(60)

    assert _fingerprint(split, 30) == _fingerprint(straight, 30)
    assert len(_fingerprint(straight, 30)) > 100


def test_retention_keeps_recent_and_periodic(tmp_path):
    scenario = load_scenario("party")
    with EventLog(tmp_path / "e.jsonl", "r") as log:
        sim = build_simulation(scenario, _cfg(scenario), log, embed_cache_path=None)
        sim.on_commit = lambda s: save_checkpoint(s, tmp_path / "ck", keep=3, keep_every=6)
        sim.run(20)
    ticks = [t for t, _ in list_checkpoints(tmp_path / "ck")]
    assert ticks == [6, 12, 18, 19, 20]
    assert latest_checkpoint(tmp_path / "ck").name == "tick_000020.json"
