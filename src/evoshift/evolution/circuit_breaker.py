from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from evoshift.config import EvolutionConfig
from evoshift.evolution.active_audit import ActiveAuditDecision, ActiveMemoryAuditor
from evoshift.schemas import MemoryItem


@dataclass(frozen=True)
class PendingCausalCanary:
    source: str
    context: str
    observation: ActiveAuditDecision
    control_memory: MemoryItem | None
    registered_index: int
    expires_after_index: int

    @property
    def observable_key(self) -> tuple[str, str]:
        return self.source, self.context

    @property
    def memory_key(self) -> tuple[str, int]:
        memory = self.observation.memory_before
        return memory.memory_id, memory.version

    @property
    def control_memory_key(self) -> tuple[str, int] | None:
        if self.control_memory is None:
            return None
        return self.control_memory.memory_id, self.control_memory.version


class CausalCircuitBreaker:
    """Bounded reversible interventions for low-trust causal contradictions."""

    def __init__(self, config: EvolutionConfig):
        self.min_trust = config.active_audit_circuit_breaker_min_trust
        self.ordinary_trust = config.min_feedback_trust_for_active_audit
        self.delta_threshold = config.active_audit_circuit_breaker_delta
        self.max_age = config.active_audit_circuit_breaker_max_age
        self.lineage_control_enabled = config.active_audit_lineage_control_enabled
        self._pending: dict[tuple[str, str], PendingCausalCanary] = {}
        self._invalidated: list[tuple[PendingCausalCanary, str]] = []
        self.probes = 0
        self.lineage_probes = 0
        self.registrations = 0
        self.lineage_registrations = 0
        self.interventions = 0
        self.lineage_interventions = 0
        self.confirmations = 0
        self.cancellations = 0
        self.expirations = 0
        self.invalidations = 0

    def trust_is_probe_eligible(self, trust: float) -> bool:
        return self.min_trust <= trust < self.ordinary_trust

    def note_probe(self, *, used_lineage_control: bool = False) -> None:
        self.probes += 1
        self.lineage_probes += int(used_lineage_control)

    def qualifies(self, delta: float) -> bool:
        return float(delta) <= self.delta_threshold

    def register(
        self,
        *,
        source: str,
        context: str,
        observation: ActiveAuditDecision,
        control_memory: MemoryItem | None = None,
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
            control_memory=control_memory,
            registered_index=observation.episode_index,
            expires_after_index=observation.episode_index + self.max_age,
        )
        self._pending[key] = pending
        self.registrations += 1
        self.lineage_registrations += int(control_memory is not None)
        return pending

    def match(
        self,
        *,
        source: str,
        context: str,
        episode_index: int,
        active_memories: Iterable[MemoryItem],
        lineage_control_versions: Iterable[tuple[str, int]] = (),
    ) -> PendingCausalCanary | None:
        key = (source, context)
        pending = self._pending.get(key)
        if pending is None or episode_index <= pending.registered_index:
            return None
        active_by_key = {(memory.memory_id, memory.version): memory for memory in active_memories}
        active_memory = active_by_key.get(pending.memory_key)
        if active_memory is None:
            self._invalidate(key, pending, "audited memory version is no longer active")
            return None
        if not ActiveMemoryAuditor.observation_is_present(
            active_memory,
            pending.observation,
        ):
            self._invalidate(
                key,
                pending,
                "provisional causal observation is no longer persisted",
            )
            return None
        if pending.control_memory_key is not None and pending.control_memory_key not in set(
            lineage_control_versions
        ):
            self._invalidate(key, pending, "lineage control version is no longer superseded")
            return None
        self._pending.pop(key, None)
        self.interventions += 1
        self.lineage_interventions += int(pending.control_memory is not None)
        return pending

    def _invalidate(
        self,
        key: tuple[str, str],
        pending: PendingCausalCanary,
        reason: str,
    ) -> None:
        self._pending.pop(key, None)
        self._invalidated.append((pending, reason))
        self.cancellations += 1
        self.invalidations += 1

    def drain_invalidated(self) -> tuple[tuple[PendingCausalCanary, str], ...]:
        invalidated = tuple(self._invalidated)
        self._invalidated.clear()
        return invalidated

    def invalidate_memory_versions(
        self,
        memory_versions: Iterable[tuple[str, int]],
        *,
        reason: str,
    ) -> None:
        versions = set(memory_versions)
        for key, pending in tuple(self._pending.items()):
            if pending.memory_key in versions:
                self._invalidate(key, pending, reason)

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
            "lineage_control_enabled": self.lineage_control_enabled,
            "min_trust": self.min_trust,
            "ordinary_trust": self.ordinary_trust,
            "delta_threshold": self.delta_threshold,
            "max_age": self.max_age,
            "probes": self.probes,
            "lineage_probes": self.lineage_probes,
            "registrations": self.registrations,
            "lineage_registrations": self.lineage_registrations,
            "interventions": self.interventions,
            "lineage_interventions": self.lineage_interventions,
            "confirmations": self.confirmations,
            "cancellations": self.cancellations,
            "expirations": self.expirations,
            "invalidations": self.invalidations,
            "invalidated_awaiting_revert": len(self._invalidated),
            "pending": len(self._pending),
        }


__all__ = ["CausalCircuitBreaker", "PendingCausalCanary"]
