from __future__ import annotations

import pytest

from evoshift.errors import BudgetExceeded
from evoshift.runtime.budget import BudgetLedger
from evoshift.schemas import LLMUsage


def test_reservations_prevent_concurrent_oversubscription() -> None:
    ledger = BudgetLedger(max_requests=2, max_total_tokens=10, max_cost_usd=1.0)
    first = ledger.reserve(estimated_tokens=6, estimated_cost_usd=0.4)
    snapshot = ledger.snapshot()
    assert snapshot.reserved_requests == 1
    assert snapshot.reserved_tokens == 6

    with pytest.raises(BudgetExceeded, match="tokens"):
        ledger.reserve(estimated_tokens=5, estimated_cost_usd=0.1)

    ledger.release(first)
    assert ledger.snapshot().reserved_requests == 0
    second = ledger.reserve(estimated_tokens=5, estimated_cost_usd=0.1)
    ledger.reconcile(
        second,
        LLMUsage(input_tokens=3, output_tokens=2, total_tokens=5, cost_usd=0.08),
    )
    consumed = ledger.snapshot()
    assert consumed.requests == 1
    assert consumed.total_tokens == 5
    assert consumed.cost_usd == pytest.approx(0.08)


def test_request_and_cost_limits_are_enforced() -> None:
    ledger = BudgetLedger(max_requests=1, max_total_tokens=100, max_cost_usd=0.25)
    reservation = ledger.reserve(estimated_tokens=10, estimated_cost_usd=0.25)
    ledger.reconcile(reservation, LLMUsage(total_tokens=10, cost_usd=0.2))

    with pytest.raises(BudgetExceeded, match="requests"):
        ledger.reserve(estimated_tokens=1)

    cost_limited = BudgetLedger(max_requests=10, max_total_tokens=100, max_cost_usd=0.25)
    with pytest.raises(BudgetExceeded, match="cost"):
        cost_limited.reserve(estimated_tokens=1, estimated_cost_usd=0.251)


def test_abort_counts_failed_logical_request_but_release_does_not() -> None:
    ledger = BudgetLedger(max_requests=3, max_total_tokens=100, max_cost_usd=1.0)
    unattempted = ledger.reserve(estimated_tokens=10)
    ledger.release(unattempted)
    assert ledger.snapshot().requests == 0

    attempted = ledger.reserve(estimated_tokens=10)
    ledger.abort(attempted)
    snapshot = ledger.snapshot()
    assert snapshot.requests == 1
    assert snapshot.total_tokens == 0
    assert snapshot.reserved_requests == 0


def test_actual_usage_can_exceed_reservation_and_blocks_future_calls() -> None:
    ledger = BudgetLedger(max_requests=5, max_total_tokens=10, max_cost_usd=1.0)
    reservation = ledger.reserve(estimated_tokens=5)
    ledger.reconcile(reservation, LLMUsage(input_tokens=7, output_tokens=5, total_tokens=12))

    assert ledger.over_budget is True
    with pytest.raises(BudgetExceeded, match="tokens"):
        ledger.reserve(estimated_tokens=0)


def test_reservation_cannot_be_closed_twice() -> None:
    ledger = BudgetLedger(max_requests=2, max_total_tokens=100, max_cost_usd=1.0)
    reservation = ledger.reserve(estimated_tokens=1)
    ledger.release(reservation)
    with pytest.raises(ValueError, match="already closed"):
        ledger.release(reservation)
