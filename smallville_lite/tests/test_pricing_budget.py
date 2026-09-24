from __future__ import annotations

import pytest

from smallville_lite.llm.budget import BudgetGuard
from smallville_lite.llm.pricing import cost_usd, worst_case_cost
from smallville_lite.llm.types import BudgetExhausted, Usage
from smallville_lite.sim.events import ListSink


def test_cost_matches_price_table(cfg):
    sonnet = cfg.models["claude-sonnet-5"]
    u = Usage(input_tokens=1000, output_tokens=200, cache_read_tokens=5000, cache_write_5m_tokens=2000)
    expected = (1000 * 2.00 + 200 * 10.00 + 5000 * 0.20 + 2000 * 2.50) / 1e6
    assert cost_usd(sonnet, u) == pytest.approx(expected)

    haiku = cfg.models["claude-haiku-4-5-20251001"]
    u = Usage(input_tokens=1_000_000, output_tokens=1_000_000, cache_write_1h_tokens=1_000_000)
    assert cost_usd(haiku, u) == pytest.approx(1.00 + 5.00 + 2.00)


def test_worst_case_bounds_any_real_outcome(cfg):
    sonnet = cfg.models["claude-sonnet-5"]
    est = worst_case_cost(sonnet, input_chars=3000, max_tokens=400, chars_per_token=3.0)
    # 1000 input tokens billed at the priciest input rate, plus the full output budget.
    actual_max = cost_usd(sonnet, Usage(cache_write_1h_tokens=1000, output_tokens=400))
    assert est >= actual_max


def test_budget_blocks_before_overspend_and_warns_once():
    sink = ListSink()
    guard = BudgetGuard(1.0, warn_fraction=0.5, sink=sink)
    guard.check(0.4)
    guard.charge(0.4)
    assert not sink.of_type("budget_warning")
    guard.check(0.2)
    guard.charge(0.2)
    guard.charge(0.1)
    assert len(sink.of_type("budget_warning")) == 1
    with pytest.raises(BudgetExhausted) as exc:
        guard.check(0.31, context="plan_day for Sam")
    assert exc.value.spent == pytest.approx(0.7)
    assert guard.exhausted
    event = sink.of_type("budget_exhausted")[0]["data"]
    assert event["context"] == "plan_day for Sam" and event["limit_usd"] == 1.0


def test_resumed_budget_keeps_counting():
    guard = BudgetGuard(1.0, spent=0.95)
    with pytest.raises(BudgetExhausted):
        guard.check(0.06)
    guard.check(0.05)
