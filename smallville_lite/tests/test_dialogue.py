"""Turn-by-turn dialogue (paper §4.3.2)."""

from __future__ import annotations

from smallville_lite.cognition.converse import converse

from helpers import Rig


def _utterances(end_at: int | None):
    def responder(req, rng):
        turn = req.hints["turn"]
        return {"utterance": f"{req.hints['speaker'].split()[0]} line {turn}",
                "end_conversation": end_at is not None and turn >= end_at}
    return responder


def _rig(**kw) -> Rig:
    rig = Rig(**kw)
    rig.seed("Isabella", ["Isabella is planning a Valentine's Day party at Hobbs Cafe on 2023-02-14 at 5pm",
                          "Isabella thinks Maria Lopez is a loyal friend"], importance=7)
    rig.seed("Maria", ["Maria Lopez and Isabella Rodriguez have been good friends for about a year",
                       "Maria is preparing for a physics exam"], importance=5)
    return rig


def test_turn_by_turn_conditioning():
    rig = _rig(responders={"utterance": _utterances(end_at=3)})
    isa, maria = rig.agents["Isabella"], rig.agents["Maria"]
    conv = converse(rig.mind, isa, maria, "Hobbs Cafe", "invite Maria to the party", rig.at(10), "conv-1")
    assert [s.split()[0] for s, _ in conv.transcript] == ["Isabella", "Maria", "Isabella", "Maria"]
    reqs = rig.requests("utterance")
    for i, req in enumerate(reqs):
        assert len(req.hints["transcript"]) == i                   # conditioned on the dialogue so far
        assert req.hints["speaker"] == ("Isabella Rodriguez" if i % 2 == 0 else "Maria Lopez")
    # Each utterance retrieves the *speaker's* memories for what the listener just said.
    dialogue_queries = [(e["agent"], e["data"]["query"]) for e in rig.events("memory_accessed")
                        if e["data"]["purpose"] == "dialogue" and "said" in e["data"]["query"]]
    assert dialogue_queries[0] == ("Maria Lopez", "Isabella Rodriguez said: Isabella line 0")
    # The relationship summary is computed once per participant, not every turn.
    assert len(rig.requests("relationship")) == 2
    assert conv.ended_by == "Maria Lopez"


def test_either_side_can_end_and_turns_are_capped():
    rig = _rig(responders={"utterance": _utterances(end_at=1)})
    conv = converse(rig.mind, rig.agents["Isabella"], rig.agents["Maria"], "Hobbs Cafe", "say hi", rig.at(10), "c")
    assert len(conv.transcript) == 2 and conv.ended_by == "Maria Lopez"
    rig = _rig(overrides={"dialogue": {"max_turns": 4}}, responders={"utterance": _utterances(end_at=None)})
    conv = converse(rig.mind, rig.agents["Isabella"], rig.agents["Maria"], "Hobbs Cafe", "chat", rig.at(10), "c")
    assert len(conv.transcript) == 4 and conv.ended_by == "max_turns"


def test_after_the_conversation_memories_and_commitments():
    rig = _rig()   # stub brain: the speaker repeats the party news; hearing it creates a commitment
    isa, maria = rig.agents["Isabella"], rig.agents["Maria"]
    maria.scratch.action = None
    conv = converse(rig.mind, isa, maria, "Hobbs Cafe", "invite Maria to the party", rig.at(10), "conv-9")
    for agent in (isa, maria):
        (dialogue,) = agent.memory.of_type("observation", "dialogue")
        assert dialogue.meta["conv_id"] == "conv-9" and dialogue.meta["transcript"]
        assert "party" in dialogue.description.lower()
        (note,) = agent.memory.of_type("reflection", "conversation_note")
        assert note.evidence == [dialogue.id]
    (commit,) = maria.memory.of_type("plan", "commitment")
    assert commit.meta["date"] == "2023-02-14" and commit.meta["time"] == "17:00" and commit.meta["place"] == "Hobbs Cafe"
    assert conv.replan["Maria Lopez"] is False                      # the party is tomorrow
    types = [e["type"] for e in rig.sink.events if e["type"] in ("conversation_started", "utterance", "conversation_ended")]
    assert types[0] == "conversation_started" and types[-1] == "conversation_ended"
    assert types.count("utterance") == len(conv.transcript)


def test_repeated_commitment_is_not_duplicated_or_replanned():
    rig = _rig()
    isa, maria = rig.agents["Isabella"], rig.agents["Maria"]
    converse(rig.mind, isa, maria, "Hobbs Cafe", "invite Maria to the party", rig.at(10), "c1")
    # Next day, the party (today) comes up again: Maria already holds that commitment.
    conv = converse(rig.mind, isa, maria, "Hobbs Cafe", "remind Maria about the party", rig.at(9, day=1), "c2")
    assert len(maria.memory.of_type("plan", "commitment")) == 1
    assert conv.replan["Maria Lopez"] is False


def test_utterance_prompt_contains_status_relationship_and_memories():
    rig = _rig(responders={"utterance": _utterances(end_at=0)})
    isa, maria = rig.agents["Isabella"], rig.agents["Maria"]
    converse(rig.mind, isa, maria, "Hobbs Cafe", "invite Maria to the party", rig.at(10), "c")
    (req,) = rig.requests("utterance")
    text = req.messages[0]["content"]
    assert "invite Maria to the party" in text
    assert "What Isabella Rodriguez thinks of Maria Lopez" in text
    assert "Valentine's Day party" in text                         # retrieved memory
    assert "cache_control" in req.system[-1]                       # speaker's cached summary prefix
