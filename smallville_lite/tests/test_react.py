"""Perceive -> continue-or-react (paper §4.3.1)."""

from __future__ import annotations

from smallville_lite.cognition.perceive import Observation, observe, perceive
from smallville_lite.cognition.react import decide
from smallville_lite.world import ActionState, Position, WorldState

from helpers import Rig, script

NO = {"react": False, "observation_index": 0, "kind": "none", "reaction": "", "reason": "busy"}


def _state(rig: Rig) -> WorldState:
    state = WorldState.initial(rig.world)
    for name, place in (("Isabella Rodriguez", "Hobbs Cafe"), ("Maria Lopez", "Hobbs Cafe"), ("Sam Moore", "Johnson Park")):
        state.positions[name] = Position(place=place)
    state.actions["Maria Lopez"] = ActionState("a1", "Maria Lopez is studying for her physics exam", "studying")
    state.actions["Isabella Rodriguez"] = ActionState("a2", "Isabella Rodriguez is brewing coffee", "brewing coffee")
    return state


def test_observe_sees_only_the_current_place():
    rig = Rig()
    obs = observe(_state(rig), "Isabella Rodriguez")
    assert [(o.kind, o.subject) for o in obs] == [("self", "Isabella Rodriguez"), ("agent", "Maria Lopez")]
    assert observe(_state(rig), "Sam Moore") == []        # alone in the park: nothing to see


def test_perceive_batches_importance_and_respects_retention():
    rig = Rig()
    isa = rig.agents["Isabella"]
    obs = observe(_state(rig), "Isabella Rodriguez")
    first = perceive(rig.mind, isa, obs, rig.at(9))
    assert len(first) == 2 and len(rig.requests("importance")) == 1           # one call for the batch
    assert perceive(rig.mind, isa, obs, rig.at(9, 10)) == []                  # nothing new
    assert len(rig.requests("importance")) == 1


def test_attention_bandwidth_caps_observations():
    rig = Rig(overrides={"perception": {"att_bandwidth": 1}})
    obs = observe(_state(rig), "Isabella Rodriguez")
    got = perceive(rig.mind, rig.agents["Isabella"], obs, rig.at(9))
    assert [o.kind for o, _ in got] == ["agent"]                              # other agents first


def test_no_salient_observation_means_no_call():
    rig = Rig()
    got = perceive(rig.mind, rig.agents["Sam"], observe(_state(rig), "Sam Moore"), rig.at(9))
    assert decide(rig.mind, rig.agents["Sam"], got, rig.mind.now) is None
    assert rig.requests("react") == []


def test_talk_decision_uses_paper_prompt_and_two_queries():
    talk = {"react": True, "observation_index": 1, "kind": "talk", "reaction": "invite Maria to the party", "reason": "friend"}
    rig = Rig(responders={"react": script(talk)})
    isa = rig.agents["Isabella"]
    rig.seed("Isabella", ["Isabella is planning a Valentine's Day party at Hobbs Cafe"], importance=8)
    got = perceive(rig.mind, isa, observe(_state(rig), "Isabella Rodriguez"), rig.at(9))
    reaction = decide(rig.mind, isa, got, rig.mind.now)
    assert reaction.kind == "talk" and reaction.observation.subject == "Maria Lopez"
    (req,) = rig.requests("react")
    assert "Should Isabella Rodriguez react to one of these observations" in req.messages[0]["content"]
    queries = [e["data"]["query"] for e in rig.events("memory_accessed") if e["data"]["purpose"] == "react"]
    assert queries == ["What is Isabella Rodriguez's relationship with Maria Lopez?",
                       "Maria Lopez is studying for her physics exam"]
    assert rig.events("react_decision")[0]["data"]["react"] is True


def test_objects_can_trigger_a_plan_change():
    burning = {"react": True, "observation_index": 1, "kind": "change_plan",
               "reaction": "turn off the stove and remake breakfast", "reason": "the stove is burning"}
    rig = Rig(responders={"react": script(burning)})
    state = WorldState.initial(rig.world)
    state.positions["Sam Moore"] = Position(place="Moore House")
    state.set_object_state("Moore House/kitchen/stove", "burning", None)
    obs = observe(state, "Sam Moore")
    assert [o for o in obs if o.kind == "object" and o.salient]
    got = perceive(rig.mind, rig.agents["Sam"], obs, rig.at(8))
    reaction = decide(rig.mind, rig.agents["Sam"], got, rig.mind.now)
    assert reaction.kind == "change_plan" and "stove" in reaction.reaction


def test_cannot_talk_to_an_object():
    bad = {"react": True, "observation_index": 1, "kind": "talk", "reaction": "talk to the stove", "reason": "?"}
    rig = Rig(responders={"react": script(bad, NO)})
    obs = [Observation("Moore House/kitchen/stove", "The stove at Moore House is burning", "object", True)]
    got = perceive(rig.mind, rig.agents["Sam"], obs, rig.at(8))
    assert decide(rig.mind, rig.agents["Sam"], got, rig.mind.now) is None
    assert len(rig.requests("react")) == 2                                     # one repair
