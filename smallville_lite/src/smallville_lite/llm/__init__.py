"""Model layer: routing, structured output, prompt caching, budget, usage, stub mode."""

from __future__ import annotations

from ..config import Config
from ..sim.events import EventSink
from .budget import BudgetGuard
from .client import LLMClient
from .prefix import PromptPrefix, SummaryCache
from .stub import FakeBackend, FakeReply
from .types import BudgetExhausted, LLMFailure, LLMResult, Usage


def build_llm(
    config: Config,
    *,
    sink: EventSink | None = None,
    budget: BudgetGuard | None = None,
    call_id_prefix: str = "c",
) -> LLMClient:
    """Stub mode gets the deterministic FakeBackend; otherwise the real Anthropic backend."""
    if config.stub:
        backend = FakeBackend(
            seed=config.seed,
            min_cache_tokens={m.id: m.min_cache_tokens for m in config.models.values()},
            chars_per_token={m.id: m.chars_per_token for m in config.models.values()},
        )
        return LLMClient(config, backend, budget=budget, sink=sink, call_id_prefix=call_id_prefix, log_prompts=True)
    from .backend_anthropic import AnthropicBackend

    backend = AnthropicBackend(max_retries=config.llm.max_retries, timeout_s=config.llm.request_timeout_s)
    return LLMClient(config, backend, budget=budget, sink=sink, call_id_prefix=call_id_prefix)


__all__ = [
    "BudgetExhausted", "BudgetGuard", "FakeBackend", "FakeReply", "LLMClient", "LLMFailure",
    "LLMResult", "PromptPrefix", "SummaryCache", "Usage", "build_llm",
]
