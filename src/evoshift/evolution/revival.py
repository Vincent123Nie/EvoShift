from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from evoshift.config import EvolutionConfig
from evoshift.schemas import MemoryItem


@dataclass(frozen=True)
class PendingRevivalCanary:
    source: str
    context: str
    memory: MemoryItem
    registered_index: int
    expires_after_index: int
    feedback_off: float
    feedback_on: float

    @property
    def observable_key(self) -> tuple[str, str]:
        return self.source, self.context

    @property
    def memory_key(self) -> tuple[str, int]:
        return self.memory.memory_id, self.memory.version

    @property
    def delta(self) -> float:
        return self.feedback_on - self.feedback_off


class DormantMemoryRevival:
    """Bounded two-observation revival of previously verified retired memories."""

    def __init__(self, config: EvolutionConfig):
        self.min_trust = config.dormant_revival_min_trust
        self.ordinary_trust = config.min_feedback_trust_for_active_audit
        self.delta_threshold = config.dormant_revival_delta
        self.max_age = config.dormant_revival_max_age
        self.min_retired_age = config.dormant_revival_min_retired_age
        self.status_indexed = config.dormant_revival_status_index_enabled
        self._pending: dict[tuple[str, str], PendingRevivalCanary] = {}
        self.probes = 0
        self.registrations = 0
        self.interventions = 0
        self.confirmations = 0
        self.cancellations = 0
        self.expirations = 0
        self.invalidations = 0

    def trust_is_probe_eligible(self, trust: float) -> bool:
        return self.min_trust <= trust < self.ordinary_trust

    def retired_long_enough(self, *, retired_index: int, episode_index: int) -> bool:
        return episode_index - retired_index >= self.min_retired_age

    def qualifies(self, delta: float) -> bool:
        return float(delta) >= self.delta_threshold

    def note_probe(self) -> None:
        self.probes += 1

    def register(
        self,
        *,
        source: str,
        context: str,
        memory: MemoryItem,
        episode_index: int,
        feedback_off: float,
        feedback_on: float,
    ) -> PendingRevivalCanary | None:
        if not self.qualifies(float(feedback_on) - float(feedback_off)):
            return None
        key = (source, context)
        if key in self._pending:
            return None
        pending = PendingRevivalCanary(
            source=source,
            context=context,
            memory=memory,
            registered_index=episode_index,
            expires_after_index=episode_index + self.max_age,
            feedback_off=float(feedback_off),
            feedback_on=float(feedback_on),
        )
        self._pending[key] = pending
        self.registrations += 1
        return pending

    def match(
        self,
        *,
        source: str,
        context: str,
        episode_index: int,
        retired_latest_versions: Iterable[tuple[str, int]],
    ) -> PendingRevivalCanary | None:
        key = (source, context)
        pending = self._pending.get(key)
        if pending is None or episode_index <= pending.registered_index:
            return None
        if pending.memory_key not in set(retired_latest_versions):
            self._pending.pop(key, None)
            self.cancellations += 1
            self.invalidations += 1
            return None
        self._pending.pop(key, None)
        self.interventions += 1
        return pending

    def resolve(self, pending: PendingRevivalCanary, *, confirmed: bool) -> None:
        del pending
        if confirmed:
            self.confirmations += 1
        else:
            self.cancellations += 1

    def expire(self, episode_index: int) -> tuple[PendingRevivalCanary, ...]:
        expired = tuple(
            pending
            for pending in self._pending.values()
            if episode_index > pending.expires_after_index
        )
        for pending in expired:
            self._pending.pop(pending.observable_key, None)
        self.expirations += len(expired)
        return expired

    def expire_all(self) -> tuple[PendingRevivalCanary, ...]:
        expired = tuple(self._pending.values())
        self._pending.clear()
        self.expirations += len(expired)
        return expired

    def snapshot(self) -> dict[str, int | float | bool]:
        return {
            "enabled": True,
            "min_trust": self.min_trust,
            "ordinary_trust": self.ordinary_trust,
            "delta_threshold": self.delta_threshold,
            "max_age": self.max_age,
            "min_retired_age": self.min_retired_age,
            "status_indexed": self.status_indexed,
            "probes": self.probes,
            "registrations": self.registrations,
            "interventions": self.interventions,
            "confirmations": self.confirmations,
            "cancellations": self.cancellations,
            "expirations": self.expirations,
            "invalidations": self.invalidations,
            "pending": len(self._pending),
        }


__all__ = ["DormantMemoryRevival", "PendingRevivalCanary"]
