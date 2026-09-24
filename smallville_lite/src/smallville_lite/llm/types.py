"""Shared types for the model layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

FailureReason = Literal["refusal", "max_tokens", "invalid", "api_error"]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0            # uncached input
    output_tokens: int = 0           # includes thinking tokens
    cache_read_tokens: int = 0
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0

    @property
    def cache_write_tokens(self) -> int:
        return self.cache_write_5m_tokens + self.cache_write_1h_tokens

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


@dataclass
class BackendRequest:
    """Everything a backend needs to make (or fake) one Messages API call."""

    task: str
    model: str
    system: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    max_tokens: int
    output_model: type[BaseModel]
    output_schema: dict[str, Any]
    thinking: dict[str, Any] | None = None
    effort: str | None = None
    temperature: float | None = None
    attempt: int = 1
    # Structured context for the STUB backend only; never sent to the API.
    hints: dict[str, Any] | None = None


@dataclass
class BackendResponse:
    text: str
    stop_reason: str | None
    usage: Usage
    request_id: str | None = None
    latency_ms: float = 0.0


@dataclass
class LLMResult(Generic[T]):
    value: T
    task: str
    model: str
    cost_usd: float
    attempts: int
    call_ids: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


class BudgetExhausted(RuntimeError):
    """Raised *before* a call whose worst-case cost would exceed the run budget."""

    def __init__(self, spent: float, limit: float, estimate: float, context: str | None = None) -> None:
        self.spent, self.limit, self.estimate, self.context = spent, limit, estimate, context
        super().__init__(
            f"budget exhausted: spent ${spent:.4f} of ${limit:.2f}; next call "
            f"({context or 'unknown'}) could cost up to ${estimate:.4f}"
        )


class LLMFailure(RuntimeError):
    """A call that could not produce a valid result. Callers apply a logged fallback."""

    def __init__(self, reason: FailureReason, task: str, detail: str = "", call_ids: list[str] | None = None) -> None:
        self.reason, self.task, self.detail = reason, task, detail
        self.call_ids = call_ids or []
        super().__init__(f"{task}: {reason}{': ' + detail if detail else ''}")
