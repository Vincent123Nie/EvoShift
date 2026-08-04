from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass

from evoshift.config import BudgetConfig
from evoshift.errors import BudgetExceeded
from evoshift.schemas import LLMUsage


@dataclass(frozen=True)
class BudgetReservation:
    reservation_id: str
    estimated_tokens: int
    estimated_cost_usd: float


@dataclass(frozen=True)
class BudgetSnapshot:
    requests: int
    total_tokens: int
    cost_usd: float
    reserved_requests: int
    reserved_tokens: int
    reserved_cost_usd: float
    max_requests: int
    max_total_tokens: int
    max_cost_usd: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class BudgetLedger:
    """Thread-safe reserve/reconcile ledger for logical LLM requests.

    A reservation makes concurrent calls account for their worst-case token
    envelope before they are sent. Retries sharing an idempotency key count as
    one logical request. ``reconcile`` records actual usage; ``abort`` records
    a failed logical request without fabricating token usage.
    """

    def __init__(
        self,
        max_requests: int = 1000,
        max_total_tokens: int = 2_000_000,
        max_cost_usd: float = 50.0,
    ) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be at least 1")
        if max_total_tokens < 1:
            raise ValueError("max_total_tokens must be at least 1")
        if max_cost_usd < 0:
            raise ValueError("max_cost_usd must be non-negative")
        self.max_requests = max_requests
        self.max_total_tokens = max_total_tokens
        self.max_cost_usd = float(max_cost_usd)
        self._requests = 0
        self._total_tokens = 0
        self._cost_usd = 0.0
        self._reservations: dict[str, BudgetReservation] = {}
        self._lock = threading.RLock()

    @classmethod
    def from_config(cls, config: BudgetConfig) -> BudgetLedger:
        return cls(
            max_requests=config.max_requests,
            max_total_tokens=config.max_total_tokens,
            max_cost_usd=config.max_cost_usd,
        )

    def reserve(
        self,
        *,
        estimated_tokens: int,
        estimated_cost_usd: float = 0.0,
    ) -> BudgetReservation:
        if estimated_tokens < 0:
            raise ValueError("estimated_tokens must be non-negative")
        if estimated_cost_usd < 0:
            raise ValueError("estimated_cost_usd must be non-negative")
        with self._lock:
            reserved_tokens = sum(item.estimated_tokens for item in self._reservations.values())
            reserved_cost = sum(item.estimated_cost_usd for item in self._reservations.values())
            projected_requests = self._requests + len(self._reservations) + 1
            projected_tokens = self._total_tokens + reserved_tokens + estimated_tokens
            projected_cost = self._cost_usd + reserved_cost + estimated_cost_usd
            failures: list[str] = []
            if projected_requests > self.max_requests:
                failures.append(f"requests {projected_requests}>{self.max_requests}")
            if projected_tokens > self.max_total_tokens:
                failures.append(f"tokens {projected_tokens}>{self.max_total_tokens}")
            if projected_cost > self.max_cost_usd + 1e-12:
                failures.append(f"cost ${projected_cost:.6f}>${self.max_cost_usd:.6f}")
            if failures:
                raise BudgetExceeded("LLM budget would be exceeded: " + ", ".join(failures))
            reservation = BudgetReservation(
                reservation_id=uuid.uuid4().hex,
                estimated_tokens=estimated_tokens,
                estimated_cost_usd=float(estimated_cost_usd),
            )
            self._reservations[reservation.reservation_id] = reservation
            return reservation

    def reconcile(self, reservation: BudgetReservation, usage: LLMUsage) -> BudgetSnapshot:
        with self._lock:
            self._pop_reservation(reservation)
            total_tokens = usage.total_tokens or (usage.input_tokens + usage.output_tokens)
            self._requests += 1
            self._total_tokens += total_tokens
            self._cost_usd += usage.cost_usd
            return self._snapshot_unlocked()

    def release(self, reservation: BudgetReservation) -> BudgetSnapshot:
        """Release a reservation when no external request was attempted."""

        with self._lock:
            self._pop_reservation(reservation)
            return self._snapshot_unlocked()

    def abort(
        self,
        reservation: BudgetReservation,
        *,
        count_request: bool = True,
    ) -> BudgetSnapshot:
        """Close a failed request reservation without invented token usage."""

        with self._lock:
            self._pop_reservation(reservation)
            if count_request:
                self._requests += 1
            return self._snapshot_unlocked()

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return self._snapshot_unlocked()

    @property
    def over_budget(self) -> bool:
        snapshot = self.snapshot()
        return (
            snapshot.requests > snapshot.max_requests
            or snapshot.total_tokens > snapshot.max_total_tokens
            or snapshot.cost_usd > snapshot.max_cost_usd + 1e-12
        )

    def _pop_reservation(self, reservation: BudgetReservation) -> None:
        current = self._reservations.pop(reservation.reservation_id, None)
        if current is None or current != reservation:
            raise ValueError("unknown or already closed budget reservation")

    def _snapshot_unlocked(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            requests=self._requests,
            total_tokens=self._total_tokens,
            cost_usd=self._cost_usd,
            reserved_requests=len(self._reservations),
            reserved_tokens=sum(item.estimated_tokens for item in self._reservations.values()),
            reserved_cost_usd=sum(item.estimated_cost_usd for item in self._reservations.values()),
            max_requests=self.max_requests,
            max_total_tokens=self.max_total_tokens,
            max_cost_usd=self.max_cost_usd,
        )
