from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, Field

from smallville_lite.llm import BudgetExhausted, BudgetGuard, FakeBackend, FakeReply, LLMClient, LLMFailure, PromptPrefix
from smallville_lite.llm.schemas import Ping
from smallville_lite.llm.usage import UsageSummary
from smallville_lite.sim.events import ListSink

LONG_RULES = "Town rules and directory. " * 300          # ~7.5k chars: over Sonnet's 1024-token minimum
VERY_LONG_RULES = "Town rules and directory. " * 800     # ~20k chars: over Haiku's 4096-token minimum


class Scores(BaseModel):
    scores: list[int] = Field(min_length=1, max_length=5)
    mood: Literal["calm", "excited"]


class Rated(BaseModel):
    rating: int = Field(ge=1, le=10)


def test_structured_output_is_validated_and_logged(llm, sink):
    result = llm.call("importance", output=Scores, user="rate these", agent="Sam Moore", prompt_id="importance@1")
    assert isinstance(result.value, Scores)
    assert result.model == "claude-haiku-4-5-20251001"
    (event,) = sink.of_type("llm_call")
    d = event["data"]
    assert event["agent"] == "Sam Moore"
    assert d["ok"] and d["task"] == "importance" and d["prompt_id"] == "importance@1"
    assert d["input_tokens"] > 0 and d["output_tokens"] > 0 and d["cost_usd"] > 0
    assert d["prompt"]["messages"][0]["content"] == "rate these"


def test_stub_is_deterministic(cfg):
    def run() -> list[dict]:
        client = LLMClient(cfg, FakeBackend(seed=cfg.seed))
        return [client.call("utterance", output=Scores, user=f"turn {i}").value.model_dump() for i in range(3)]

    assert run() == run()


def test_request_shape_follows_task_settings(llm, fake):
    llm.call("utterance", output=Ping, user="hi", prefix=PromptPrefix("rules", "summary"))
    llm.call("plan_day", output=Ping, user="plan")
    llm.call("importance", output=Ping, user="rate")
    utter, plan, imp = fake.requests
    assert utter.thinking == {"type": "disabled"} and utter.temperature is None and utter.effort is None
    assert plan.thinking == {"type": "adaptive"} and plan.effort == "low"
    assert imp.thinking is None and imp.temperature == 0.0 and imp.effort is None
    # The cache breakpoint sits on the last prefix block (the agent summary).
    assert "cache_control" not in utter.system[0]
    assert utter.system[1]["cache_control"] == {"type": "ephemeral"}
    assert utter.output_schema["additionalProperties"] is False


def test_prompt_caching_respects_model_minimums(llm):
    sonnet_prefix = PromptPrefix(LONG_RULES, "Agent summary v1")
    first = llm.call("utterance", output=Ping, user="a", prefix=sonnet_prefix)
    second = llm.call("utterance", output=Ping, user="b", prefix=sonnet_prefix)
    assert first.usage.cache_write_tokens > 0 and first.usage.cache_read_tokens == 0
    assert second.usage.cache_read_tokens == first.usage.cache_write_tokens
    assert second.cost_usd < first.cost_usd

    # Same prefix on Haiku: below its 4096-token minimum, so no caching at all.
    h1 = llm.call("react", output=Ping, user="a", prefix=sonnet_prefix)
    h2 = llm.call("react", output=Ping, user="b", prefix=sonnet_prefix)
    assert h1.usage.cache_write_tokens == h2.usage.cache_read_tokens == 0
    big = PromptPrefix(VERY_LONG_RULES, "Agent summary v1")
    llm.call("react", output=Ping, user="a", prefix=big)
    assert llm.call("react", output=Ping, user="b", prefix=big).usage.cache_read_tokens > 0

    # A regenerated summary is a new prefix: one fresh write, then reads again.
    new_prefix = PromptPrefix(LONG_RULES, "Agent summary v2")
    assert llm.call("utterance", output=Ping, user="c", prefix=new_prefix).usage.cache_write_tokens > 0


def test_invalid_output_gets_one_repair_retry(cfg, sink):
    replies = iter(['{"rating": 42}', '{"rating": 7}'])
    fake = FakeBackend(responders={"importance": lambda req, rng: FakeReply(next(replies))})
    client = LLMClient(cfg, fake, sink=sink)
    result = client.call("importance", output=Rated, user="rate")
    assert result.value.rating == 7 and result.attempts == 2
    calls = [e["data"] for e in sink.of_type("llm_call")]
    assert [c["ok"] for c in calls] == [False, True]
    assert "rating" in calls[0]["error"]
    # The repair turn shows the model its bad output and the validation error.
    repair = fake.requests[1].messages
    assert repair[1] == {"role": "assistant", "content": '{"rating": 42}'}
    assert "failed validation" in repair[2]["content"]


def test_semantic_validator_and_final_failure(cfg, sink):
    fake = FakeBackend(responders={"importance": lambda req, rng: {"rating": 3}})
    client = LLMClient(cfg, fake, sink=sink)

    def must_be_high(v: Rated) -> None:
        if v.rating < 5:
            raise ValueError("rating must be at least 5 for this memory")

    with pytest.raises(LLMFailure) as exc:
        client.call("importance", output=Rated, user="rate", validate=must_be_high)
    assert exc.value.reason == "invalid" and len(exc.value.call_ids) == 2


def test_truncation_retries_with_larger_limit(cfg, sink):
    replies = iter([FakeReply('{"pong": "po', "max_tokens"), FakeReply('{"pong": "pong", "echo": "x"}')])
    fake = FakeBackend(responders={"utterance": lambda req, rng: next(replies)})
    client = LLMClient(cfg, fake, sink=sink)
    result = client.call("utterance", output=Ping, user="hi")
    assert result.attempts == 2
    assert fake.requests[1].max_tokens == 2 * fake.requests[0].max_tokens


def test_refusal_is_not_retried(cfg, sink):
    fake = FakeBackend(responders={"utterance": lambda req, rng: FakeReply("", "refusal")})
    client = LLMClient(cfg, fake, sink=sink)
    with pytest.raises(LLMFailure) as exc:
        client.call("utterance", output=Ping, user="hi")
    assert exc.value.reason == "refusal" and len(fake.requests) == 1


def test_budget_stop_sends_nothing_further(cfg, sink):
    fake = FakeBackend()
    client = LLMClient(cfg, fake, sink=sink, budget=BudgetGuard(0.02, sink=sink))
    with pytest.raises(BudgetExhausted):
        for i in range(1000):
            client.call("utterance", output=Ping, user=f"turn {i}")
    sent = len(fake.requests)
    assert client.budget.spent <= 0.02
    assert len(sink.of_type("llm_call")) == sent  # every sent request was logged and charged
    with pytest.raises(BudgetExhausted):
        client.call("utterance", output=Ping, user="one more")
    assert len(fake.requests) == sent


def test_usage_summary_by_task_and_model(llm, sink):
    llm.call("importance", output=Ping, user="a")
    llm.call("importance", output=Ping, user="b")
    llm.call("utterance", output=Ping, user="c")
    live = llm.usage.as_dict()
    assert live["by_task"]["importance"]["calls"] == 2
    assert live["by_model"]["claude-sonnet-5"]["calls"] == 1
    # The same summary can be rebuilt from the event stream alone.
    rebuilt = UsageSummary.from_events(sink.events).as_dict()
    assert rebuilt == live
    assert "TOTAL" in llm.usage.render()
