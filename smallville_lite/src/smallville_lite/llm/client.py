"""The single LLM entry point (DESIGN §6).

Every model call in the project goes through :meth:`LLMClient.call`, which:

1. routes the task to a model and settings from ``models.toml``;
2. builds the request: cached system prefix (rules + agent summary) and volatile user content,
   with a JSON-schema output format derived from a Pydantic model;
3. checks the budget with a worst-case estimate *before* sending;
4. validates the response (schema, then an optional semantic validator), with one repair
   retry on invalid output and one larger-``max_tokens`` retry on truncation;
5. prices the actual usage, charges the budget, and emits one ``llm_call`` event per attempt.

It raises :class:`BudgetExhausted` (stop the run cleanly) or :class:`LLMFailure` (caller
applies a logged fallback). It never returns unvalidated data.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

import anthropic
from pydantic import BaseModel, ValidationError

from ..config import Config
from ..sim.events import EventSink, NullSink
from .budget import BudgetGuard
from .prefix import PromptPrefix
from .pricing import cost_usd, request_chars, worst_case_cost
from .types import BackendRequest, BackendResponse, BudgetExhausted, LLMFailure, LLMResult, T, Usage
from .usage import UsageSummary

Validator = Callable[[Any], None]  # raise ValueError with a message the model can act on


class Backend(Protocol):
    def complete(self, req: BackendRequest) -> BackendResponse: ...


class LLMClient:
    def __init__(
        self,
        config: Config,
        backend: Backend,
        *,
        budget: BudgetGuard | None = None,
        sink: EventSink | None = None,
        call_id_prefix: str = "c",
        log_prompts: bool | None = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.sink = sink or NullSink()
        self.budget = budget or BudgetGuard(config.budget.max_usd, warn_fraction=config.budget.warn_fraction, sink=self.sink)
        self.usage = UsageSummary()
        self.log_prompts = config.llm.log_prompts if log_prompts is None else log_prompts
        self._prefix = call_id_prefix
        self._counter = 0
        self._schemas: dict[type[BaseModel], dict[str, Any]] = {}

    # ------------------------------------------------------------------
    def call(
        self,
        task: str,
        *,
        output: type[T],
        user: str,
        prefix: PromptPrefix | None = None,
        agent: str | None = None,
        validate: Validator | None = None,
        prompt_id: str | None = None,
        max_tokens: int | None = None,
        hints: dict[str, Any] | None = None,
    ) -> LLMResult[T]:
        spec = self.config.task(task)
        model = self.config.models[spec.model]
        system = prefix.system_blocks(self.config.llm.cache_ttl) if prefix else []
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        schema = self._schema_for(output)
        limit = max_tokens or spec.max_tokens
        repairs_left = self.config.llm.repair_attempts
        bumped = False
        call_ids: list[str] = []
        total_cost = 0.0
        total_usage = Usage()
        attempt = 0

        while True:
            attempt += 1
            req = BackendRequest(
                task=task,
                model=spec.model,
                system=system,
                messages=list(messages),
                max_tokens=limit,
                output_model=output,
                output_schema=schema,
                thinking=_thinking_param(spec.thinking),
                effort=spec.effort,
                temperature=spec.temperature,
                attempt=attempt,
                hints=hints,
            )
            estimate = worst_case_cost(model, request_chars(system, req.messages, schema), limit, self.config.llm.chars_per_token)
            self.budget.check(estimate, context=f"{task}{' for ' + agent if agent else ''}")

            call_id = self._next_call_id()
            call_ids.append(call_id)
            try:
                resp = self.backend.complete(req)
            except anthropic.APIError as exc:
                self._log(call_id, req, agent, prompt_id, ok=False, error=f"{type(exc).__name__}: {exc}")
                raise LLMFailure("api_error", task, str(exc), call_ids) from exc

            cost = cost_usd(model, resp.usage)
            self.budget.charge(cost)
            total_cost += cost
            total_usage = _add_usage(total_usage, resp.usage)

            if resp.stop_reason == "refusal":
                self._log(call_id, req, agent, prompt_id, resp=resp, cost=cost, ok=False, error="refusal")
                raise LLMFailure("refusal", task, "model declined the request", call_ids)

            if resp.stop_reason == "max_tokens":
                self._log(call_id, req, agent, prompt_id, resp=resp, cost=cost, ok=False, error="max_tokens")
                if not bumped and limit < spec.max_tokens_cap:
                    bumped = True
                    limit = min(spec.max_tokens_cap, limit * 2)
                    continue
                raise LLMFailure("max_tokens", task, f"output truncated at {limit} tokens", call_ids)

            try:
                value = output.model_validate_json(resp.text)
                if validate is not None:
                    validate(value)
            except (ValidationError, ValueError) as exc:
                detail = _short_error(exc)
                self._log(call_id, req, agent, prompt_id, resp=resp, cost=cost, ok=False, error=f"invalid: {detail}")
                if repairs_left > 0:
                    repairs_left -= 1
                    messages = messages + [
                        {"role": "assistant", "content": resp.text or "{}"},
                        {"role": "user", "content": (
                            f"That JSON failed validation: {detail}\n"
                            "Return a corrected JSON object that fixes this and follows every earlier instruction."
                        )},
                    ]
                    continue
                raise LLMFailure("invalid", task, detail, call_ids) from exc

            self._log(call_id, req, agent, prompt_id, resp=resp, cost=cost, ok=True)
            return LLMResult(
                value=value, task=task, model=spec.model, cost_usd=total_cost,
                attempts=attempt, call_ids=call_ids, usage=total_usage,
            )

    # ------------------------------------------------------------------
    def _schema_for(self, output: type[BaseModel]) -> dict[str, Any]:
        schema = self._schemas.get(output)
        if schema is None:
            schema = anthropic.transform_schema(output)
            self._schemas[output] = schema
        return schema

    def _next_call_id(self) -> str:
        self._counter += 1
        return f"{self._prefix}{self._counter:06d}"

    def _log(
        self,
        call_id: str,
        req: BackendRequest,
        agent: str | None,
        prompt_id: str | None,
        *,
        resp: BackendResponse | None = None,
        cost: float = 0.0,
        ok: bool,
        error: str | None = None,
    ) -> None:
        usage = resp.usage if resp else Usage()
        data: dict[str, Any] = {
            "call_id": call_id,
            "task": req.task,
            "model": req.model,
            "agent": agent,
            "prompt_id": prompt_id,
            "attempt": req.attempt,
            "ok": ok,
            "stop_reason": resp.stop_reason if resp else None,
            "error": error,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
            "cost_usd": round(cost, 8),
            "latency_ms": round(resp.latency_ms, 1) if resp else 0.0,
            "request_id": resp.request_id if resp else None,
        }
        if self.log_prompts:
            data["prompt"] = {"system": [b["text"] for b in req.system], "messages": req.messages}
            data["response"] = resp.text if resp else None
        self.usage.add_llm_call(data)
        self.sink.emit("llm_call", data, agent=agent)


def _thinking_param(mode: str) -> dict[str, Any] | None:
    if mode == "adaptive":
        return {"type": "adaptive"}
    if mode == "disabled":
        return {"type": "disabled"}
    return None


def _add_usage(a: Usage, b: Usage) -> Usage:
    return Usage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cache_read_tokens=a.cache_read_tokens + b.cache_read_tokens,
        cache_write_5m_tokens=a.cache_write_5m_tokens + b.cache_write_5m_tokens,
        cache_write_1h_tokens=a.cache_write_1h_tokens + b.cache_write_1h_tokens,
    )


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = []
        for err in exc.errors()[:5]:
            loc = ".".join(str(x) for x in err["loc"]) or "(root)"
            parts.append(f"{loc}: {err['msg']}")
        return "; ".join(parts)
    return str(exc)[:500]


__all__ = ["LLMClient", "BudgetExhausted", "LLMFailure", "Validator"]
