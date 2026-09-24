"""Continue-or-react decision (paper §4.3.1; DESIGN §5.5).

One Haiku call per agent per tick, only when a new observation is salient. Context comes from
the paper's two retrieval queries per observation ("relationship with X", "X is <status>"),
passed as raw memories rather than summarized (DESIGN §10, L1).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from ..clock import fmt_datetime
from ..llm import LLMFailure
from ..llm.prompts import load_prompt
from ..llm.schemas import ReactDecision
from ..memory import MemoryNode, numbered
from .mind import Mind
from .perceive import Observation

if TYPE_CHECKING:
    from ..agent import Agent

MAX_OBSERVATIONS = 3


@dataclass
class Reaction:
    kind: str                 # talk | change_plan
    reaction: str
    reason: str
    observation: Observation
    node: MemoryNode


def decide(mind: Mind, agent: "Agent", perceived: list[tuple[Observation, MemoryNode]], now: datetime) -> Reaction | None:
    salient = [(o, n) for o, n in perceived if o.salient][:MAX_OBSERVATIONS]
    if not salient:
        return None
    k = mind.sim.react_context_k
    blocks, seen, memories = [], set(), []
    for i, (ob, node) in enumerate(salient, 1):
        who = ob.subject if ob.kind == "agent" else ob.subject.split("/")[-1]
        ctx = mind.retrieve(agent, f"What is {agent.name}'s relationship with {who}?", k=k, purpose="react",
                            exclude_subtypes=("hour_plan",))
        ctx += mind.retrieve(agent, ob.text, k=k, purpose="react", exclude_subtypes=("hour_plan",))
        uniq = [m for m in ctx if m.node.id not in seen and m.node.id != node.id]
        seen.update(m.node.id for m in uniq)
        memories.extend(m.node.description for m in uniq)
        blocks.append(f"Observation {i}: {ob.text}\nRelevant memories:\n{numbered(uniq)}")

    action = agent.scratch.action
    prompt = load_prompt("react")
    n = len(salient)

    def check(v: ReactDecision) -> None:
        if v.react and v.kind == "none":
            raise ValueError("react is true, so kind must be 'talk' or 'change_plan'")
        if v.react and not 1 <= v.observation_index <= n:
            raise ValueError(f"observation_index must be between 1 and {n}")
        if v.react and v.kind == "talk" and salient[v.observation_index - 1][0].kind != "agent":
            raise ValueError("'talk' is only possible when reacting to another person")

    try:
        out = mind.llm.call(
            "react", output=ReactDecision, agent=agent.name, prefix=mind.prefix(agent), prompt_id=prompt.ref,
            validate=check,
            user=prompt.render(
                first=agent.identity.first_name, name=agent.name, now=fmt_datetime(now),
                status=f"{agent.name} is {action.activity} at {action.place}" if action else f"{agent.name} is idle",
                observations="\n\n".join(blocks),
            ),
            hints={"agent": agent.name, "observations": [{"text": o.text, "kind": o.kind, "subject": o.subject}
                                                          for o, _ in salient], "memories": memories},
        ).value
    except LLMFailure as exc:
        mind.fallback("react", agent, exc.reason, "continue")
        return None

    reacted = salient[out.observation_index - 1] if out.react else salient[0]
    mind.sink.emit("react_decision", {
        "observation_node_id": reacted[1].id, "react": out.react, "kind": out.kind if out.react else "none",
        "reaction": out.reaction, "reason": out.reason,
    }, agent=agent.name, game_time=now.isoformat())
    if not out.react:
        return None
    return Reaction(kind=out.kind, reaction=out.reaction, reason=out.reason, observation=reacted[0], node=reacted[1])
