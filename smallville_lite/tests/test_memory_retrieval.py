"""Memory stream and retrieval scoring (paper §4.1)."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from smallville_lite.memory import MemoryNode, NewMemory, Weights, min_max, score_memories

from helpers import START, Rig


def _node(i: int, hours_ago: float, importance: int) -> MemoryNode:
    t = START - timedelta(hours=hours_ago)
    return MemoryNode(id=f"n{i}", agent="x", type="observation", subtype="perception", description=f"m{i}",
                      created=t, last_accessed=t, importance=importance, embedding_key=str(i))


def test_recency_is_exponential_in_game_hours_since_last_access():
    nodes = [_node(0, 0, 5), _node(1, 10, 5), _node(2, 100, 5)]
    vecs = np.eye(3, dtype=np.float32)
    only_recency = Weights(recency=1, importance=0, relevance=0, decay=0.995)
    scored = {s.node.id: s for s in score_memories(nodes, vecs, np.ones(3), START, only_recency)}
    raw = np.array([0.995 ** 0, 0.995 ** 10, 0.995 ** 100])
    expected = (raw - raw.min()) / (raw.max() - raw.min())
    assert [scored[f"n{i}"].recency for i in range(3)] == pytest.approx(list(expected))
    # Newer is better (the reference code inverted this; ARCHITECTURE D2).
    assert scored["n0"].score > scored["n1"].score > scored["n2"].score


def test_min_max_and_all_equal_case():
    assert list(min_max(np.array([2.0, 4.0, 6.0]))) == [0.0, 0.5, 1.0]
    assert list(min_max(np.array([3.0, 3.0]))) == [0.5, 0.5]


def test_weights_are_configurable_and_combined():
    nodes = [_node(0, 0, 1), _node(1, 50, 10)]
    vecs = np.array([[1, 0], [0, 1]], dtype=np.float32)
    query = np.array([0.0, 1.0])
    default = score_memories(nodes, vecs, query, START, Weights())
    # n0: recency 1, importance 0, relevance 0 -> 1 ; n1: recency 0, importance 1, relevance 1 -> 2
    assert [s.node.id for s in default] == ["n1", "n0"]
    assert [s.score for s in default] == pytest.approx([2.0, 1.0])
    recency_heavy = score_memories(nodes, vecs, query, START, Weights(recency=3))
    assert recency_heavy[0].node.id == "n0"


def test_retrieve_updates_last_accessed_and_logs_components():
    rig = Rig()
    now = rig.at(8)
    nodes = rig.seed("Isabella", ["Isabella is planning a Valentine's Day party at Hobbs Cafe",
                                  "Isabella wiped the counter", "Isabella read the newspaper"],
                     when=now - timedelta(hours=2))
    later = rig.at(12)
    top = rig.mind.retrieve(rig.agents["Isabella"], "Valentine's Day party", k=1, purpose="test")
    assert top[0].node.id == nodes[0].id
    assert nodes[0].last_accessed == later
    assert nodes[1].last_accessed == now - timedelta(hours=2)     # not returned, not touched
    (event,) = [e for e in rig.events("memory_accessed") if e["data"]["purpose"] == "test"]
    assert set(event["data"]["results"][0]) == {"node_id", "score", "recency", "importance", "relevance"}


def test_retrieve_filters_and_never_sees_the_future():
    rig = Rig()
    rig.seed("Sam", ["Sam gardens in the park"], when=rig.at(8))
    rig.seed("Sam", ["Sam will run for mayor"], when=rig.at(10), type="plan")
    rig.at(9)
    assert [m.node.description for m in rig.mind.retrieve(rig.agents["Sam"], "mayor", k=5)] == ["Sam gardens in the park"]
    rig.at(11)
    assert rig.mind.retrieve(rig.agents["Sam"], "mayor", k=5, types=["plan"])[0].node.type == "plan"


def test_stream_accumulator_depth_and_evidence():
    rig = Rig()
    mem = rig.agents["Klaus"].memory
    now = rig.at(9)
    obs = mem.add_many(now, [NewMemory("observation", "perception", "a", 4), NewMemory("observation", "perception", "b", 6)])
    mem.add(now, NewMemory("plan", "day_plan", "plan something", 9))
    assert mem.importance_since_reflection == 10          # plans do not count
    r1 = mem.add(now, NewMemory("reflection", "insight", "i1", 5, evidence=[obs[0].id]))
    r2 = mem.add(now, NewMemory("reflection", "insight", "i2", 5, evidence=[r1.id, obs[1].id]))
    assert (r1.depth, r2.depth) == (1, 2)                 # reflection trees (paper Fig. 7)
    assert mem.importance_since_reflection == 10          # reflections do not count either
    with pytest.raises(ValueError):
        mem.add(now, NewMemory("reflection", "insight", "bad", 5, evidence=["nope:1"]))


def test_stream_state_round_trip():
    rig = Rig()
    mem = rig.agents["Maria"].memory
    rig.seed("Maria", ["Maria studies physics", "Maria likes the cafe"])
    restored = type(mem)("Maria Lopez", rig.embedder)
    restored.load_state(mem.state())
    assert [n.to_dict() for n in restored.nodes] == [n.to_dict() for n in mem.nodes]
    assert np.array_equal(restored.vector(mem.nodes[0].id), mem.vector(mem.nodes[0].id))
