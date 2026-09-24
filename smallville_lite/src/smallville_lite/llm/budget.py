"""Hard per-run budget, checked before every paid call (DESIGN §6.4)."""

from __future__ import annotations

from ..sim.events import EventSink, NullSink
from .types import BudgetExhausted


class BudgetGuard:
    """Tracks spend against ``max_usd``.

    ``check(estimate)`` raises :class:`BudgetExhausted` *before* a call whose worst-case cost
    would push spend over the limit, so the limit is never exceeded by a completed call.
    ``charge(actual)`` records the real cost afterwards. Calls run sequentially, so there is
    no reservation bookkeeping. ``spent`` is persisted in checkpoints so a resumed run keeps
    counting from where it stopped.
    """

    def __init__(
        self,
        max_usd: float,
        *,
        warn_fraction: float = 0.8,
        spent: float = 0.0,
        sink: EventSink | None = None,
    ) -> None:
        self.max_usd = max_usd
        self.warn_fraction = warn_fraction
        self.spent = spent
        self.sink = sink or NullSink()
        self._warned = spent >= max_usd * warn_fraction
        self.exhausted = False

    @property
    def remaining(self) -> float:
        return max(0.0, self.max_usd - self.spent)

    def check(self, estimate: float, context: str | None = None) -> None:
        if self.spent + estimate > self.max_usd:
            self.exhausted = True
            self.sink.emit(
                "budget_exhausted",
                {"spent_usd": round(self.spent, 6), "limit_usd": self.max_usd,
                 "next_estimate_usd": round(estimate, 6), "context": context},
            )
            raise BudgetExhausted(self.spent, self.max_usd, estimate, context)

    def charge(self, cost: float) -> None:
        self.spent += cost
        if not self._warned and self.spent >= self.max_usd * self.warn_fraction:
            self._warned = True
            self.sink.emit("budget_warning", {"spent_usd": round(self.spent, 6), "limit_usd": self.max_usd})

    def state(self) -> dict[str, float]:
        return {"spent_usd": self.spent, "max_usd": self.max_usd}
