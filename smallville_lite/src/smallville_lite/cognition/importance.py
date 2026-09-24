"""Importance scoring (paper §4.1), batched: one Haiku call scores many memories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from ..llm import LLMFailure
from ..llm.prompts import load_prompt
from ..llm.schemas import ImportanceScores
from .mind import Mind

if TYPE_CHECKING:
    from ..agent import Agent

FALLBACK_IMPORTANCE = 3


def score_importance(mind: Mind, agent: "Agent", texts: Sequence[str]) -> list[int]:
    if not texts:
        return []
    n = len(texts)
    prompt = load_prompt("importance")
    statements = "\n".join(f"{i}. {t}" for i, t in enumerate(texts, 1))

    def check(v: ImportanceScores) -> None:
        if len(v.scores) != n:
            raise ValueError(f"expected exactly {n} scores, got {len(v.scores)}")
        bad = [s for s in v.scores if not 1 <= s <= 10]
        if bad:
            raise ValueError(f"scores must be integers from 1 to 10, got {bad}")

    try:
        result = mind.llm.call(
            "importance", output=ImportanceScores, agent=agent.name, prompt_id=prompt.ref, validate=check,
            user=prompt.render(name=agent.name, traits=agent.identity.traits, n=n, statements=statements),
            hints={"texts": list(texts)},
        )
        return list(result.value.scores)
    except LLMFailure as exc:
        scores = [FALLBACK_IMPORTANCE] * n
        mind.fallback("importance", agent, exc.reason, scores)
        return scores
