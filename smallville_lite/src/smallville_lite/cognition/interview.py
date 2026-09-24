"""Interview mode (paper §6, App. B): ask an agent a question, answered with retrieval."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..clock import fmt_datetime
from ..llm import LLMFailure
from ..llm.prompts import load_prompt
from ..llm.schemas import InterviewAnswer
from ..memory import NewMemory, numbered
from .mind import Mind

if TYPE_CHECKING:
    from ..agent import Agent

PRESETS_PATH = Path(__file__).resolve().parents[3] / "scenarios" / "_shared" / "interview_presets.toml"


@dataclass
class InterviewResult:
    agent: str
    question: str
    answer: str
    cited_node_ids: list[str]
    ok: bool = True


def interview(mind: Mind, agent: "Agent", question: str, now: datetime, *, remember: bool | None = None,
              source: str = "cli") -> InterviewResult:
    # Interviews observe the agent without changing it: no last-accessed updates (paper §6 evaluation setting).
    retrieved = mind.retrieve(agent, question, k=mind.sim.interview_retrieve_k, purpose="interview",
                              update_access=False)
    prompt = load_prompt("interview")
    try:
        out = mind.llm.call(
            "interview", output=InterviewAnswer, agent=agent.name, prefix=mind.prefix(agent), prompt_id=prompt.ref,
            user=prompt.render(name=agent.name, first=agent.identity.first_name, now=fmt_datetime(now),
                               memories=numbered(retrieved), question=question),
            hints={"question": question, "memories": [m.node.description for m in retrieved]},
        ).value
        cited = [retrieved[i - 1].node.id for i in out.cited if 1 <= i <= len(retrieved)]
        result = InterviewResult(agent.name, question, out.answer.strip(), list(dict.fromkeys(cited)))
    except LLMFailure as exc:
        result = InterviewResult(agent.name, question, "(no answer: the model call failed)", [], ok=False)
        mind.fallback("interview", agent, exc.reason, result.answer)
    mind.sink.emit("interview", {"question": question, "answer": result.answer, "cited_node_ids": result.cited_node_ids,
                                 "source": source}, agent=agent.name, game_time=now.isoformat())
    if remember if remember is not None else mind.sim.interview_remember:
        agent.memory.add(now, NewMemory("observation", "interview",
                                        f"An interviewer asked {agent.name}: \"{question}\" and {agent.name} answered: \"{result.answer}\"",
                                        importance=3))
    return result


def load_presets(path: str | Path = PRESETS_PATH) -> dict[str, list[str]]:
    """App. B questions by category. Bracketed names like [Name] are filled by the caller."""
    with Path(path).open("rb") as fh:
        raw = tomllib.load(fh)
    return {cat: list(qs) for cat, qs in raw["categories"].items()}
