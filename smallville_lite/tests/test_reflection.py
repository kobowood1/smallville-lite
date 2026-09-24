"""Reflection (paper §4.2)."""

from __future__ import annotations

from smallville_lite.cognition.reflect import reflect, should_reflect
from smallville_lite.memory import NewMemory

from helpers import Rig, script


def _fill(rig: Rig, first: str, n: int, importance: int) -> None:
    rig.agents[first].memory.add_many(rig.mind.now, [
        NewMemory("observation", "perception", f"{first} noticed thing number {i} at the cafe", importance)
        for i in range(n)
    ])


def test_trigger_is_accumulated_observation_importance():
    rig = Rig(overrides={"reflection": {"threshold": 40}})
    rig.at(9)
    _fill(rig, "Klaus", 4, 9)
    assert not should_reflect(rig.mind, rig.agents["Klaus"])    # 36 < 40
    _fill(rig, "Klaus", 1, 4)
    assert should_reflect(rig.mind, rig.agents["Klaus"])        # 40


def test_questions_retrieval_insights_with_evidence():
    rig = Rig(overrides={"reflection": {"threshold": 10, "recent_records": 6}})
    klaus = rig.agents["Klaus"]
    now = rig.at(9)
    rig.seed("Klaus", [f"Klaus spent hour {i} on his gentrification research" for i in range(8)], importance=5)
    rig.mind.summaries.get(klaus.name)
    nodes = reflect(rig.mind, klaus, now)

    (qreq,) = rig.requests("reflect_questions")
    assert "most salient high-level questions" in qreq.messages[0]["content"]      # paper wording
    assert len(qreq.hints["statements"]) == 6                                     # the N most recent records
    assert len(rig.requests("reflect_insights")) == 3                             # one per question
    assert nodes and all(n.type == "reflection" and n.subtype == "insight" for n in nodes)
    for n in nodes:
        assert n.evidence and all(e in klaus.memory.by_id for e in n.evidence)
        assert n.depth == 1 + max(klaus.memory.get(e).depth for e in n.evidence)
    assert klaus.memory.importance_since_reflection == 0
    (event,) = rig.events("reflection")
    assert len(event["data"]["questions"]) == 3
    # The cached summary (prompt-cache prefix) is regenerated after a reflection.
    assert rig.mind.summaries.get(klaus.name).version == 2


def test_invalid_citations_are_dropped():
    insights = {"insights": [
        {"insight": "Klaus is devoted to research", "evidence": [99, 1]},
        {"insight": "Klaus is hallucinating", "evidence": [42]},
    ]}
    rig = Rig(overrides={"reflection": {"questions": 1}},
              responders={"reflect_questions": script({"questions": ["What does Klaus care about?"]}),
                          "reflect_insights": script(insights)})
    rig.seed("Klaus", ["Klaus works at the library", "Klaus reads about housing"], importance=5)
    nodes = reflect(rig.mind, rig.agents["Klaus"], rig.at(10))
    assert [n.description for n in nodes] == ["Klaus is devoted to research"]
    assert len(nodes[0].evidence) == 1
