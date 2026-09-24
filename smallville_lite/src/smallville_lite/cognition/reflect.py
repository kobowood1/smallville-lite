"""Reflection (paper §4.2; DESIGN §5.6).

Trigger: the sum of importance of observations since the last reflection reaches the threshold.
Flow: 100 most recent records -> 3 questions -> retrieve per question -> insights citing
numbered statements -> reflection nodes with evidence pointers (building reflection trees).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from ..llm import LLMFailure
from ..llm.prompts import load_prompt
from ..llm.schemas import InsightsOut, QuestionsOut
from ..memory import MemoryNode, NewMemory, numbered
from .importance import score_importance
from .mind import Mind

if TYPE_CHECKING:
    from ..agent import Agent


def should_reflect(mind: Mind, agent: "Agent") -> bool:
    return agent.memory.importance_since_reflection >= mind.sim.reflection_threshold


def reflect(mind: Mind, agent: "Agent", now: datetime) -> list[MemoryNode]:
    s = mind.sim
    trigger_sum = agent.memory.importance_since_reflection
    recent = agent.memory.recent(s.reflection_recent_records)
    questions = _questions(mind, agent, recent)
    insights: list[tuple[str, list[str]]] = []
    for q in questions:
        retrieved = mind.retrieve(agent, q, k=s.reflection_retrieve_k, purpose="reflect")
        if retrieved:
            insights.extend(_insights(mind, agent, q, [m.node for m in retrieved]))
    # Deduplicate identical insight texts produced for different questions.
    unique: dict[str, list[str]] = {}
    for text, evidence in insights:
        unique.setdefault(text.strip(), evidence)
    new_nodes: list[MemoryNode] = []
    if unique:
        texts = list(unique)
        scores = score_importance(mind, agent, texts)
        new_nodes = agent.memory.add_many(now, [
            NewMemory("reflection", "insight", t, importance=sc, evidence=unique[t]) for t, sc in zip(texts, scores)
        ])
    agent.memory.reset_reflection_accumulator()
    mind.sink.emit("reflection", {
        "trigger_sum": trigger_sum, "questions": questions,
        "insights": [{"node_id": n.id, "text": n.description, "evidence": n.evidence, "depth": n.depth} for n in new_nodes],
    }, agent=agent.name, game_time=now.isoformat())
    mind.summaries.notify(agent.name, "reflection")
    return new_nodes


def _questions(mind: Mind, agent: "Agent", recent: list[MemoryNode]) -> list[str]:
    n = mind.sim.reflection_questions
    prompt = load_prompt("reflect_questions")

    def check(v: QuestionsOut) -> None:
        if len(v.questions) != n:
            raise ValueError(f"give exactly {n} questions")

    try:
        return mind.llm.call(
            "reflect_questions", output=QuestionsOut, agent=agent.name, prefix=mind.prefix(agent),
            prompt_id=prompt.ref, validate=check,
            user=prompt.render(statements="\n".join(f"- {m.description}" for m in recent), n=n),
            hints={"n": n, "statements": [m.description for m in recent], "agent": agent.name},
        ).value.questions
    except LLMFailure as exc:
        fallback = [f"What matters most to {agent.name} right now?"]
        mind.fallback("reflect_questions", agent, exc.reason, fallback)
        return fallback


def _insights(mind: Mind, agent: "Agent", question: str, nodes: list[MemoryNode]) -> list[tuple[str, list[str]]]:
    n = mind.sim.insights_per_question
    prompt = load_prompt("reflect_insights")
    try:
        out = mind.llm.call(
            "reflect_insights", output=InsightsOut, agent=agent.name, prefix=mind.prefix(agent),
            prompt_id=prompt.ref,
            user=prompt.render(name=agent.name, question=question, statements=numbered(nodes), n=n),
            hints={"n": n, "n_statements": len(nodes), "statements": [x.description for x in nodes]},
        ).value
    except LLMFailure as exc:
        mind.fallback("reflect_insights", agent, exc.reason, "no insights")
        return []
    results = []
    for ins in out.insights[:n]:
        # Citations must point into the numbered list; invalid ones are dropped (DESIGN §5.6).
        evidence = list(dict.fromkeys(nodes[i - 1].id for i in ins.evidence if 1 <= i <= len(nodes)))
        if evidence and ins.insight.strip():
            results.append((ins.insight.strip(), evidence))
    return results
