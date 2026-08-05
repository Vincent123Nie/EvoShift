from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from evoshift.config import EvolutionConfig
from evoshift.evolution.active_audit import ActiveAuditDecision


@dataclass(frozen=True)
class PendingCausalCanary:
    source: str
    context: str
    observation: ActiveAuditDecision
    registered_index: int
    expires_after_index: int

    @property
    def observable_key(self) -> tuple[str, str]:
        return self.source, self.context

    @property
    def memory_key(self) -> tuple[str, int]:
        memory = self.observation.memory_before
        return memory.memory_id, memory.version


class CausalCircuitBreaker:
    """Bounded reversible interventions for low-trust causal contradictions."""

    def __init__(self, config: EvolutionConfig):
        self.min_trust = config.active_audit_circuit_breaker_min_trust
        self.ordinary_trust = config.min_feedback_trust_for_active_audit
        self.delta_threshold = config.active_audit_circuit_breaker_delta
        self.max_age = config.active_audit_circuit_breaker_max_age
        self._pending: dict[tuple[str, str], PendingCausalCanary] = {}
        self.probes = 0
        self.registrations = 0
        self.interventions = 0
        self.confirmations = 0
        self.cancellations = 0
        self.expirations = 0
        self.invalidations = 0

    def trust_is_probe_eligible(self, trust: float) -> bool:
        return self.min_trust <= trust < self.ordinary_trust

    def note_probe(self) -> None:
        self.probes += 1

    def qualifies(self, delta: float) -> bool:
        return float(delta) <= self.delta_threshold

    def register(
        self,
        *,
        source: str,
        context: str,
        observation: ActiveAuditDecision,
    ) -> PendingCausalCanary | None:
        if not self.qualifies(observation.delta):
            return None
        key = (source, context)
        if key in self._pending:
            return None
        pending = PendingCausalCanary(
            source=source,
            context=context,
            observation=observation,
            registered_index=observation.episode_index,
            expires_after_index=observation.episode_index + self.max_age,
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
        active_memory_versions: Iterable[tuple[str, int]],
    ) -> PendingCausalCanary | None:
        key = (source, context)
        pending = self._pending.get(key)
        if pending is None or episode_index <= pending.registered_index:
            return None
        if pending.memory_key not in set(active_memory_versions):
            self._pending.pop(key, None)
            self.cancellations += 1
            self.invalidations += 1
            return None
        self._pending.pop(key, None)
        self.interventions += 1
        return pending

    def resolve(self, pending: PendingCausalCanary, *, confirmed: bool) -> None:
        del pending
        if confirmed:
            self.confirmations += 1
        else:
            self.cancellations += 1

    def expire(self, episode_index: int) -> tuple[PendingCausalCanary, ...]:
        expired = tuple(
            pending
            for pending in self._pending.values()
            if episode_index > pending.expires_after_index
        )
        for pending in expired:
            self._pending.pop(pending.observable_key, None)
        self.expirations += len(expired)
        return expired

    def expire_all(self) -> tuple[PendingCausalCanary, ...]:
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
            "probes": self.probes,
            "registrations": self.registrations,
            "interventions": self.interventions,
            "confirmations": self.confirmations,
            "cancellations": self.cancellations,
            "expirations": self.expirations,
            "invalidations": self.invalidations,
            "pending": len(self._pending),
        }


__all__ = ["CausalCircuitBreaker", "PendingCausalCanary"]
