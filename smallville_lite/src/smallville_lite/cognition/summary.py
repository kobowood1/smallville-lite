"""App. A cached agent summary: three retrievals, one summarization call (DESIGN §5.3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..llm import LLMFailure
from ..llm.prompts import load_prompt
from ..llm.schemas import AgentSummaryOut
from ..memory import numbered
from .mind import Mind

if TYPE_CHECKING:
    from ..agent import Agent


def regenerate_summary(mind: Mind, agent: "Agent") -> str:
    name = agent.name
    k = 10
    core = mind.retrieve(agent, f"{name}'s core characteristics", k=k, purpose="summary")
    occupation = mind.retrieve(agent, f"{name}'s current daily occupation", k=k, purpose="summary")
    progress = mind.retrieve(agent, f"{name}'s feeling about their recent progress in life", k=k, purpose="summary")
    prompt = load_prompt("summary")
    header = f"Name: {name} (age {agent.identity.age})\nInnate traits: {agent.identity.traits}"
    try:
        # No prefix: the summary *is* the prefix, so it cannot depend on itself.
        out = mind.llm.call(
            "summary", output=AgentSummaryOut, agent=name, prompt_id=prompt.ref,
            user=prompt.render(
                name=name, header=header,
                core=numbered(core), occupation=numbered(occupation), progress=numbered(progress),
            ),
            hints={"agent": name, "bio": agent.identity.bio,
                   "memories": [m.node.description for m in core + occupation + progress]},
        ).value
        body = f"{out.core}\n{out.occupation}\n{out.recent_progress}"
    except LLMFailure as exc:
        body = "; ".join(agent.identity.bio)
        mind.fallback("summary", agent, exc.reason, body)
    return f"{header}\n{body}"
