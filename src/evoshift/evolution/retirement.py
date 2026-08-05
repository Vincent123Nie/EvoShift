from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from evoshift.config import EvolutionConfig
from evoshift.schemas import MemoryItem


@dataclass(frozen=True)
class PendingRetirement:
    source: str
    context: str
    memory: MemoryItem
    restored_predecessors: tuple[MemoryItem, ...]
    registered_index: int
    expires_after_index: int
    mechanism: str
    phase_index: int
    oracle_stale: bool
    early_retirement: bool

    @property
    def observable_key(self) -> tuple[str, str]:
        return self.source, self.context

    @property
    def memory_key(self) -> tuple[str, int]:
        return self.memory.memory_id, self.memory.version


class RetirementProbation:
    """Two-phase, reversible confirmation for destructive memory retirement."""

    def __init__(self, config: EvolutionConfig):
        self.delta_threshold = config.dormant_revival_delta
        self.ordinary_trust = config.min_feedback_trust_for_active_audit
        self.max_age = config.active_audit_retirement_probation_max_age
        self._pending: dict[tuple[str, str], PendingRetirement] = {}
        self.registrations = 0
        self.interventions = 0
        self.confirmations = 0
        self.cancellations = 0
        self.deferrals = 0
        self.expirations = 0
        self.invalidations = 0

    def qualifies(self, delta: float) -> bool:
        return float(delta) >= self.delta_threshold

    def register(
        self,
        *,
        source: str,
        context: str,
        memory: MemoryItem,
        restored_predecessors: Iterable[MemoryItem],
        episode_index: int,
        mechanism: str,
        phase_index: int,
        oracle_stale: bool,
        early_retirement: bool,
    ) -> PendingRetirement | None:
        key = (source, context)
        if key in self._pending:
            return None
        pending = PendingRetirement(
            source=source,
            context=context,
            memory=memory,
            restored_predecessors=tuple(restored_predecessors),
            registered_index=episode_index,
            expires_after_index=episode_index + self.max_age,
            mechanism=mechanism,
            phase_index=phase_index,
            oracle_stale=oracle_stale,
            early_retirement=early_retirement,
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
        retired_memory_versions: Iterable[tuple[str, int]],
    ) -> PendingRetirement | None:
        key = (source, context)
        pending = self._pending.get(key)
        if pending is None or episode_index <= pending.registered_index:
            return None
        if pending.memory_key not in set(retired_memory_versions):
            self._pending.pop(key, None)
            self.cancellations += 1
            self.invalidations += 1
            return None
        self._pending.pop(key, None)
        self.interventions += 1
        return pending

    def resolve(self, pending: PendingRetirement, *, confirmed: bool) -> None:
        del pending
        if confirmed:
            self.confirmations += 1
        else:
            self.cancellations += 1

    def defer(self, pending: PendingRetirement) -> None:
        """Return an inconclusive intervention to its original bounded window."""

        if pending.observable_key in self._pending:
            raise ValueError("retirement transaction context is already pending")
        self._pending[pending.observable_key] = pending
        self.deferrals += 1

    def pending_memory_keys(self) -> set[tuple[str, int]]:
        """Return exact versions currently in a reversible retirement transaction."""

        return {pending.memory_key for pending in self._pending.values()}

    def expire(self, episode_index: int) -> tuple[PendingRetirement, ...]:
        expired = tuple(
            pending
            for pending in self._pending.values()
            if episode_index > pending.expires_after_index
        )
        for pending in expired:
            self._pending.pop(pending.observable_key, None)
        self.expirations += len(expired)
        return expired

    def expire_all(self) -> tuple[PendingRetirement, ...]:
        expired = tuple(self._pending.values())
        self._pending.clear()
        self.expirations += len(expired)
        return expired

    def snapshot(self) -> dict[str, int | float | bool]:
        return {
            "enabled": True,
            "delta_threshold": self.delta_threshold,
            "ordinary_trust": self.ordinary_trust,
            "max_age": self.max_age,
            "registrations": self.registrations,
            "interventions": self.interventions,
            "confirmations": self.confirmations,
            "cancellations": self.cancellations,
            "deferrals": self.deferrals,
            "expirations": self.expirations,
            "invalidations": self.invalidations,
            "pending": len(self._pending),
        }


__all__ = ["PendingRetirement", "RetirementProbation"]
