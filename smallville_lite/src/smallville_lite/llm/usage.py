"""Per-run usage summary: calls, tokens, cost, latency by task and by model (DESIGN §6.7).

Built from ``llm_call`` / ``embed_call`` event data, so the same numbers can be recomputed from
``events.jsonl`` by the report or the web viewer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Bucket:
    calls: int = 0
    failed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    latencies: list[float] = field(default_factory=list)

    def add(self, d: dict[str, Any]) -> None:
        self.calls += 1
        self.failed += 0 if d.get("ok", True) else 1
        self.input_tokens += d.get("input_tokens", 0)
        self.output_tokens += d.get("output_tokens", 0)
        self.cache_read_tokens += d.get("cache_read_tokens", 0)
        self.cache_write_tokens += d.get("cache_write_tokens", 0)
        self.cost_usd += d.get("cost_usd", 0.0)
        if d.get("latency_ms"):
            self.latencies.append(float(d["latency_ms"]))

    def pct(self, q: float) -> float | None:
        if not self.latencies:
            return None
        xs = sorted(self.latencies)
        return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]

    @property
    def cache_hit_rate(self) -> float:
        cacheable = self.cache_read_tokens + self.cache_write_tokens + self.input_tokens
        return self.cache_read_tokens / cacheable if cacheable else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls, "failed": self.failed,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens, "cache_write_tokens": self.cache_write_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "p50_latency_ms": self.pct(0.5), "p95_latency_ms": self.pct(0.95),
        }


@dataclass
class EmbedBucket:
    calls: int = 0
    texts: int = 0
    cached: int = 0
    tokens: int = 0
    cost_usd: float = 0.0

    def add(self, d: dict[str, Any]) -> None:
        self.calls += 1
        self.texts += d.get("n_texts", 0)
        self.cached += d.get("n_cached", 0)
        self.tokens += d.get("tokens", 0)
        self.cost_usd += d.get("cost_usd", 0.0)


class UsageSummary:
    def __init__(self) -> None:
        self.by_task: dict[str, Bucket] = {}
        self.by_model: dict[str, Bucket] = {}
        self.total = Bucket()
        self.embeddings: dict[str, EmbedBucket] = {}

    def add_llm_call(self, d: dict[str, Any]) -> None:
        self.by_task.setdefault(d["task"], Bucket()).add(d)
        self.by_model.setdefault(d["model"], Bucket()).add(d)
        self.total.add(d)

    def add_embed_call(self, d: dict[str, Any]) -> None:
        self.embeddings.setdefault(f"{d['provider']}/{d['model']}", EmbedBucket()).add(d)

    @property
    def cost_usd(self) -> float:
        return self.total.cost_usd + sum(b.cost_usd for b in self.embeddings.values())

    @classmethod
    def from_events(cls, events: Iterable[dict[str, Any]]) -> "UsageSummary":
        s = cls()
        for e in events:
            if e["type"] == "llm_call":
                s.add_llm_call(e["data"])
            elif e["type"] == "embed_call":
                s.add_embed_call(e["data"])
        return s

    @classmethod
    def from_log(cls, path: str | Path) -> "UsageSummary":
        from ..sim.events import read_events

        return cls.from_events(read_events(path))

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total.as_dict(),
            "total_cost_usd": round(self.cost_usd, 6),
            "by_task": {k: v.as_dict() for k, v in sorted(self.by_task.items())},
            "by_model": {k: v.as_dict() for k, v in sorted(self.by_model.items())},
            "embeddings": {k: vars(v) for k, v in sorted(self.embeddings.items())},
        }

    def render(self) -> str:
        lines = ["LLM usage", _header()]
        for label, groups in (("by model", self.by_model), ("by task", self.by_task)):
            lines.append(f"-- {label}")
            for name, b in sorted(groups.items(), key=lambda kv: -kv[1].cost_usd):
                lines.append(_row(name, b))
        lines.append(_row("TOTAL", self.total))
        if self.embeddings:
            lines.append("Embeddings")
            for name, e in sorted(self.embeddings.items()):
                lines.append(
                    f"  {name:<32} calls={e.calls} texts={e.texts} cached={e.cached} "
                    f"tokens={e.tokens} cost=${e.cost_usd:.4f}"
                )
        lines.append(f"Total cost: ${self.cost_usd:.4f}")
        return "\n".join(lines)


def _header() -> str:
    return (f"  {'':<28}{'calls':>6}{'fail':>5}{'in':>9}{'out':>8}{'c.read':>9}{'c.write':>9}"
            f"{'hit%':>6}{'cost $':>10}{'p50ms':>8}{'p95ms':>8}")


def _row(name: str, b: Bucket) -> str:
    p50, p95 = b.pct(0.5), b.pct(0.95)
    return (f"  {name[:28]:<28}{b.calls:>6}{b.failed:>5}{b.input_tokens:>9}{b.output_tokens:>8}"
            f"{b.cache_read_tokens:>9}{b.cache_write_tokens:>9}{b.cache_hit_rate * 100:>5.0f}%"
            f"{b.cost_usd:>10.4f}{(f'{p50:.0f}' if p50 is not None else '-'):>8}"
            f"{(f'{p95:.0f}' if p95 is not None else '-'):>8}")
