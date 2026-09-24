"""Pydantic output models, one per LLM task (DESIGN §9).

Structured outputs do not enforce numeric bounds (minimum/maximum); the SDK moves them into the
field description, and ``LLMClient`` enforces them on validation (with a repair retry).
Place and object choices are built per call as ``Literal`` enums from the agent's known world,
so an invalid location cannot come back (see ``hour_blocks_model`` / ``detail_model``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Ping(_Out):
    pong: str
    echo: str


# -- importance / labels (Haiku) -------------------------------------------

class ImportanceScores(_Out):
    scores: list[int] = Field(description="One integer 1-10 per memory, in the same order.")


class ActionLabel(_Out):
    emoji: str = Field(description="One or two emoji that depict the action.")
    label: str = Field(description="A short label of at most four words.")


class ActionLabels(_Out):
    labels: list[ActionLabel]


# -- planning (Sonnet) -------------------------------------------------------

class DayPlanItem(_Out):
    time: str = Field(description="Start time, 24-hour HH:MM.")
    description: str


class DayPlanOut(_Out):
    wake_time: str = Field(description="24-hour HH:MM.")
    sleep_time: str = Field(description="24-hour HH:MM (use 24:00 for midnight).")
    items: list[DayPlanItem]


class RecapOut(_Out):
    recap: str


@lru_cache(maxsize=256)
def hour_blocks_model(places: tuple[str, ...]) -> type[BaseModel]:
    """``{"blocks": [{start, end, activity, place}]}`` with ``place`` restricted to ``places``."""
    block = create_model(
        "HourBlockOut",
        __config__=ConfigDict(extra="forbid"),
        start=(str, Field(description="24-hour HH:MM")),
        end=(str, Field(description="24-hour HH:MM (24:00 for midnight)")),
        activity=(str, ...),
        place=(Literal[places], ...),  # type: ignore[valid-type]
    )
    return create_model("HourBlocksOut", __config__=ConfigDict(extra="forbid"), blocks=(list[block], ...))  # type: ignore[valid-type]


@lru_cache(maxsize=256)
def detail_model(objects: tuple[str, ...]) -> type[BaseModel]:
    """``{"steps": [{minutes, activity, object, object_state}]}``; ``object`` from ``objects`` or 'none'."""
    choices = tuple(dict.fromkeys(objects + ("none",)))
    step = create_model(
        "DetailStepOut",
        __config__=ConfigDict(extra="forbid"),
        minutes=(Literal[5, 10, 15], ...),
        activity=(str, ...),
        object=(Literal[choices], ...),  # type: ignore[valid-type]
        object_state=(str, Field(description="What the object is doing while used, e.g. 'brewing coffee'; '' if none.")),
    )
    return create_model("DetailOut", __config__=ConfigDict(extra="forbid"), steps=(list[step], ...))  # type: ignore[valid-type]


# -- reacting / dialogue ------------------------------------------------------

class ReactDecision(_Out):
    react: bool
    observation_index: int = Field(description="1-based index of the observation reacted to; 0 if not reacting.")
    kind: Literal["none", "talk", "change_plan"]
    reaction: str = Field(description="What the agent does, e.g. 'ask Maria about her exam'. Empty if none.")
    reason: str


class RelationshipOut(_Out):
    summary: str


class UtteranceOut(_Out):
    utterance: str
    end_conversation: bool


class DialogueSummary(_Out):
    topic_label: str = Field(description="At most six words.")
    summary: str = Field(description="1-3 sentences. Keep any plans, invitations, news, dates, times and places verbatim.")


class Commitment(_Out):
    what: str
    date: str = Field(description="YYYY-MM-DD, or '' if no date was agreed.")
    time: str = Field(description="24-hour HH:MM, or '' if no time was agreed.")
    place: str = Field(description="Place name, or '' if none.")


class ConversationNoteOut(_Out):
    note: str = Field(description="What the agent takes away from the conversation, in one or two sentences.")
    commitments: list[Commitment]
    replan_today: bool


# -- reflection / summary / interview -----------------------------------------

class QuestionsOut(_Out):
    questions: list[str]


class Insight(_Out):
    insight: str
    evidence: list[int] = Field(description="Statement numbers that support the insight.")


class InsightsOut(_Out):
    insights: list[Insight]


class AgentSummaryOut(_Out):
    core: str
    occupation: str
    recent_progress: str


class InterviewAnswer(_Out):
    answer: str
    cited: list[int] = Field(description="Numbers of the memories the answer relies on.")


class JudgeOut(_Out):
    knows: bool
    reason: str
