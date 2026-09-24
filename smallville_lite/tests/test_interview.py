"""Interview mode (paper §6, App. B)."""

from __future__ import annotations

from smallville_lite.cognition.interview import interview, load_presets

from helpers import Rig


def test_interview_is_retrieval_grounded_and_not_remembered():
    rig = Rig()
    klaus = rig.agents["Klaus"]
    nodes = rig.seed("Klaus", ["Isabella Rodriguez invited Klaus to a Valentine's Day party at Hobbs Cafe",
                               "Klaus reads at the library"], importance=6)
    before = len(klaus.memory)
    result = interview(rig.mind, klaus, "Was there a Valentine's day party?", rig.at(20))
    assert result.cited_node_ids == [nodes[0].id]
    assert "Valentine" in result.answer
    assert len(klaus.memory) == before
    assert any(e["data"]["purpose"] == "interview" for e in rig.events("memory_accessed"))
    (event,) = rig.events("interview")
    assert event["data"]["question"] == "Was there a Valentine's day party?"


def test_interview_can_be_remembered():
    rig = Rig()
    klaus = rig.agents["Klaus"]
    rig.seed("Klaus", ["Klaus reads at the library"])
    interview(rig.mind, klaus, "What's your occupation?", rig.at(20), remember=True)
    assert klaus.memory.nodes[-1].subtype == "interview"


def test_appendix_b_presets():
    presets = load_presets()
    assert list(presets) == ["self_knowledge", "memory", "plans", "reactions", "reflections"]
    assert sum(len(q) for q in presets.values()) == 25
    assert "Give an introduction of yourself." in presets["self_knowledge"]
