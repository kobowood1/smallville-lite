"""Turn-by-turn dialogue (paper §4.3.2; DESIGN §5.7).

Each utterance is generated for its speaker from: the speaker's cached summary (prefix), the
speaker's relationship summary of the listener (computed once per conversation), memories
retrieved for what the listener just said, and the transcript so far. Either speaker can end.
Afterwards both participants store a dialogue observation and a conversation note, and any
commitments become plan nodes so later planning can honour them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

from ..clock import fmt_datetime, parse_hhmm
from ..llm import LLMFailure
from ..llm.prompts import load_prompt
from ..llm.schemas import ConversationNoteOut, DialogueSummary, RelationshipOut, UtteranceOut
from ..memory import NewMemory, numbered
from .importance import score_importance
from .mind import Mind

if TYPE_CHECKING:
    from ..agent import Agent


@dataclass
class Conversation:
    conv_id: str
    participants: tuple[str, str]
    place: str
    start: datetime
    transcript: list[tuple[str, str]] = field(default_factory=list)
    ended_by: str = "max_turns"
    topic_label: str = ""
    summary: str = ""
    dialogue_nodes: dict[str, str] = field(default_factory=dict)
    replan: dict[str, bool] = field(default_factory=dict)
    commitments: dict[str, list[str]] = field(default_factory=dict)

    @property
    def minutes(self) -> int:
        return len(self.transcript)


def converse(mind: Mind, initiator: "Agent", target: "Agent", place: str, reason: str, now: datetime, conv_id: str) -> Conversation:
    s = mind.sim
    conv = Conversation(conv_id=conv_id, participants=(initiator.name, target.name), place=place, start=now)
    mind.sink.emit("conversation_started", {"conv_id": conv_id, "participants": list(conv.participants),
                                            "initiator": initiator.name, "place": place, "reason": reason},
                   agent=initiator.name, game_time=now.isoformat())
    relation = {initiator.name: _relationship(mind, initiator, target), target.name: _relationship(mind, target, initiator)}

    speaker, listener = initiator, target
    for turn in range(s.dialogue_max_turns):
        last = conv.transcript[-1][1] if conv.transcript else None
        query = f"{listener.name} said: {last}" if last else f"{reason} ({listener.name})"
        retrieved = mind.retrieve(speaker, query, k=s.dialogue_retrieve_k, purpose="dialogue",
                                  exclude_subtypes=("hour_plan",))
        try:
            out = _utterance(mind, speaker, listener, conv, relation[speaker.name], retrieved, reason, turn, now)
        except LLMFailure as exc:
            mind.fallback("utterance", speaker, exc.reason, "end conversation")
            conv.ended_by = "error"
            break
        text = out.utterance.strip()
        conv.transcript.append((speaker.name, text))
        mind.sink.emit("utterance", {
            "conv_id": conv_id, "turn": turn, "speaker": speaker.name, "listener": listener.name, "text": text,
            "end_conversation": out.end_conversation, "retrieved_node_ids": [m.node.id for m in retrieved],
        }, agent=speaker.name, game_time=(now + timedelta(minutes=turn * s.minutes_per_utterance)).isoformat())
        if out.end_conversation:
            conv.ended_by = speaker.name
            break
        speaker, listener = listener, speaker

    if conv.transcript:
        _after(mind, initiator, target, conv, now)
    mind.sink.emit("conversation_ended", {
        "conv_id": conv_id, "turns": len(conv.transcript), "ended_by": conv.ended_by,
        "topic_label": conv.topic_label, "summary": conv.summary,
        "duration_minutes": max(1, len(conv.transcript) * s.minutes_per_utterance),
    }, agent=initiator.name, game_time=now.isoformat())
    return conv


def _relationship(mind: Mind, me: "Agent", other: "Agent") -> str:
    retrieved = mind.retrieve(me, f"What does {me.name} know and feel about {other.name}?",
                              k=mind.sim.relationship_k, purpose="dialogue", exclude_subtypes=("hour_plan",))
    prompt = load_prompt("relationship")
    try:
        return mind.llm.call(
            "relationship", output=RelationshipOut, agent=me.name, prompt_id=prompt.ref,
            user=prompt.render(me=me.name, other=other.name, memories=numbered(retrieved)),
            hints={"self": me.name, "other": other.name, "memories": [m.node.description for m in retrieved]},
        ).value.summary
    except LLMFailure as exc:
        text = f"{me.name} has no particular impression of {other.name}."
        mind.fallback("relationship", me, exc.reason, text)
        return text


def _utterance(mind, speaker, listener, conv, relationship, retrieved, reason, turn, now) -> UtteranceOut:
    prompt = load_prompt("utterance")
    act = speaker.scratch.action
    other = listener.scratch.action
    transcript = "\n".join(f"{who}: {text}" for who, text in conv.transcript) or "(the conversation has not started yet)"
    if turn == 0:
        situation = f"{speaker.name} decided to start a conversation with {listener.name} because: {reason}"
    elif turn == 1:
        situation = f"{conv.transcript[0][0]} has started a conversation with {speaker.name}."
    else:
        situation = f"{speaker.name} and {listener.name} are in the middle of a conversation."
    return mind.llm.call(
        "utterance", output=UtteranceOut, agent=speaker.name, prefix=mind.prefix(speaker), prompt_id=prompt.ref,
        user=prompt.render(
            speaker=speaker.name, listener=listener.name, first=speaker.identity.first_name, now=fmt_datetime(now),
            place=conv.place,
            status=f"{speaker.name} was {act.activity}" if act else f"{speaker.name} was idle",
            other_status=f"{listener.name} was {other.activity}" if other else f"{listener.name} is here",
            situation=situation, relationship=relationship, memories=numbered(retrieved), transcript=transcript,
            max_turns=mind.sim.dialogue_max_turns, turn=turn + 1,
        ),
        hints={"speaker": speaker.name, "listener": listener.name, "turn": turn, "reason": reason,
               "memories": [m.node.description for m in retrieved], "transcript": list(conv.transcript)},
    ).value


def _after(mind: Mind, a: "Agent", b: "Agent", conv: Conversation, now: datetime) -> None:
    transcript = "\n".join(f"{who}: {text}" for who, text in conv.transcript)
    prompt = load_prompt("summarize_dialogue")
    try:
        summ = mind.llm.call(
            "summarize_dialogue", output=DialogueSummary, agent=a.name, prompt_id=prompt.ref,
            user=prompt.render(a=a.name, b=b.name, place=conv.place, transcript=transcript),
            hints={"transcript": list(conv.transcript)},
        ).value
        conv.topic_label, conv.summary = summ.topic_label, summ.summary
    except LLMFailure as exc:
        conv.topic_label, conv.summary = "a chat", " ".join(t for _, t in conv.transcript)[:400]
        mind.fallback("summarize_dialogue", a, exc.reason, conv.summary)

    for me, other in ((a, b), (b, a)):
        dialogue = NewMemory(
            "observation", "dialogue", f"Conversation with {other.name} at {conv.place}: {conv.summary}",
            importance=0, meta={"conv_id": conv.conv_id, "with": other.name, "place": conv.place,
                                "transcript": [list(t) for t in conv.transcript]},
        )
        note, commitments, replan = _note(mind, me, other, conv, transcript, now)
        texts = [dialogue.description] + ([note] if note else []) + [c[0] for c in commitments]
        scores = score_importance(mind, me, texts)
        dialogue.importance = scores[0]
        (dnode,) = me.memory.add_many(now, [dialogue])
        conv.dialogue_nodes[me.name] = dnode.id
        extra: list[NewMemory] = []
        idx = 1
        if note:
            extra.append(NewMemory("reflection", "conversation_note", note, importance=scores[idx],
                                   evidence=[dnode.id], meta={"conv_id": conv.conv_id}))
            idx += 1
        for (text, meta), sc in zip(commitments, scores[idx:]):
            extra.append(NewMemory("plan", "commitment", text, importance=sc, evidence=[dnode.id],
                                   meta={**meta, "conv_id": conv.conv_id}))
        me.memory.add_many(now, extra)
        conv.replan[me.name] = replan
        conv.commitments[me.name] = [c[0] for c in commitments]


def _note(mind: Mind, me: "Agent", other: "Agent", conv: Conversation, transcript: str, now: datetime):
    prompt = load_prompt("conversation_note")
    existing = [n.description for n in me.memory.of_type("plan", "commitment")
                if n.meta.get("date") is None or n.meta["date"] >= now.date().isoformat()]
    try:
        out = mind.llm.call(
            "conversation_note", output=ConversationNoteOut, agent=me.name, prefix=mind.prefix(me),
            prompt_id=prompt.ref,
            user=prompt.render(me=me.name, other=other.name, first=me.identity.first_name, now=fmt_datetime(now),
                               today=now.date().isoformat(), transcript=transcript,
                               existing="\n".join(f"- {c}" for c in existing) or "(none)"),
            hints={"agent": me.name, "other": other.name, "transcript": list(conv.transcript),
                   "today": now.date().isoformat()},
        ).value
    except LLMFailure as exc:
        mind.fallback("conversation_note", me, exc.reason, "no note")
        return None, [], False
    commitments = []
    today = now.date()
    # A commitment the agent already holds (same date, time and place) is not new: hearing about the
    # party again must not create another plan node or force another replan.
    known = {(n.meta.get("date"), n.meta.get("time"), n.meta.get("place"))
             for n in me.memory.of_type("plan", "commitment")}
    for c in out.commitments:
        meta = {"place": c.place or None}
        when = ""
        try:
            d = date.fromisoformat(c.date) if c.date else None
        except ValueError:
            d = None
        if d is not None:
            meta["date"] = d.isoformat()
            when = f" on {d.isoformat()}"
            if c.time:
                try:
                    meta["time"] = parse_hhmm(c.time, d).strftime("%H:%M")
                    when += f" at {meta['time']}"
                except ValueError:
                    pass
        where = f" at {c.place}" if c.place else ""
        key = (meta.get("date"), meta.get("time"), meta.get("place"))
        if key[0] is not None and key in known:
            continue
        known.add(key)
        commitments.append((f"{me.name} committed to: {c.what}{when}{where}", meta))
    replan = out.replan_today or any(m.get("date") == today.isoformat() for _, m in commitments)
    return out.note.strip() or None, commitments, replan
