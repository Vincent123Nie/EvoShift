from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Literal

from evoshift.config import EvolutionConfig
from evoshift.schemas import MemoryItem


@dataclass(frozen=True)
class PendingRetirement:
    source: str
    context: str
    memory: MemoryItem
    active_snapshot: MemoryItem
    restored_predecessors: tuple[MemoryItem, ...]
    registered_index: int
    expires_after_index: int
    mechanism: str
    phase_index: int
    oracle_stale: bool
    early_retirement: bool
    confirmation_indices: tuple[int, ...] = ()
    veto_indices: tuple[int, ...] = ()

    @property
    def observable_key(self) -> tuple[str, str]:
        return self.source, self.context

    @property
    def memory_key(self) -> tuple[str, int]:
        return self.memory.memory_id, self.memory.version

    @property
    def transaction_key(self) -> tuple[str, str, str, int]:
        return self.source, self.context, self.memory.memory_id, self.memory.version

    @property
    def last_evidence_index(self) -> int | None:
        indices = self.confirmation_indices + self.veto_indices
        return max(indices) if indices else None


RetirementEvidenceOutcome = Literal["confirm", "veto", "defer"]


@dataclass(frozen=True)
class RetirementEvidenceDecision:
    pending: PendingRetirement
    outcome: RetirementEvidenceOutcome
    reason: str
    evidence_recorded: bool = False


class RetirementProbation:
    """Two-phase, reversible confirmation for destructive memory retirement."""

    def __init__(self, config: EvolutionConfig):
        self.delta_threshold = config.dormant_revival_delta
        self.ordinary_trust = config.min_feedback_trust_for_active_audit
        self.max_age = config.active_audit_retirement_probation_max_age
        self.min_confirmations = config.active_audit_retirement_probation_min_confirmations
        self.min_evidence_span = config.active_audit_retirement_probation_min_evidence_span
        self._pending: dict[tuple[str, str, str, int], PendingRetirement] = {}
        self._invalidated: list[tuple[PendingRetirement, str]] = []
        self.registrations = 0
        self.interventions = 0
        self.confirmations = 0
        self.cancellations = 0
        self.deferrals = 0
        self.evidence_observations = 0
        self.temporal_deferrals = 0
        self.low_trust_deferrals = 0
        self.unapplied_deferrals = 0
        self.neutral_deferrals = 0
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
        active_snapshot: MemoryItem,
        restored_predecessors: Iterable[MemoryItem],
        episode_index: int,
        mechanism: str,
        phase_index: int,
        oracle_stale: bool,
        early_retirement: bool,
    ) -> PendingRetirement | None:
        pending = PendingRetirement(
            source=source,
            context=context,
            memory=memory,
            active_snapshot=active_snapshot,
            restored_predecessors=tuple(restored_predecessors),
            registered_index=episode_index,
            expires_after_index=episode_index + self.max_age,
            mechanism=mechanism,
            phase_index=phase_index,
            oracle_stale=oracle_stale,
            early_retirement=early_retirement,
        )
        existing = self._pending.get(pending.transaction_key)
        if existing is not None:
            return existing
        self._pending[pending.transaction_key] = pending
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
        retired_versions = set(retired_memory_versions)
        candidates: list[PendingRetirement] = []
        for transaction_key, pending in tuple(self._pending.items()):
            if pending.observable_key != (source, context):
                continue
            if pending.memory_key not in retired_versions:
                self._pending.pop(transaction_key, None)
                self.cancellations += 1
                self.invalidations += 1
                self._invalidated.append(
                    (pending, "exact retired memory version is no longer available")
                )
                continue
            if episode_index > pending.registered_index:
                candidates.append(pending)
        if not candidates:
            return None
        pending = min(
            candidates,
            key=lambda item: (item.registered_index, item.memory.memory_id, item.memory.version),
        )
        self._pending.pop(pending.transaction_key, None)
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

        if pending.transaction_key in self._pending:
            raise ValueError("retirement transaction is already pending")
        self._pending[pending.transaction_key] = pending
        self.deferrals += 1

    def observe(
        self,
        pending: PendingRetirement,
        *,
        episode_index: int,
        trust: float,
        old_memory_applied: bool,
        delta: float,
    ) -> RetirementEvidenceDecision:
        """Record one learner-visible paired observation and advance the transaction."""

        if trust < self.ordinary_trust:
            self.low_trust_deferrals += 1
            self.defer(pending)
            return RetirementEvidenceDecision(
                pending,
                "defer",
                "feedback trust below audit threshold",
            )
        if not old_memory_applied:
            self.unapplied_deferrals += 1
            self.defer(pending)
            return RetirementEvidenceDecision(
                pending,
                "defer",
                "forced old memory was not explicitly applied",
            )
        direction: RetirementEvidenceOutcome
        if self.qualifies(delta):
            direction = "confirm"
        elif self.qualifies(-delta):
            direction = "veto"
        else:
            self.neutral_deferrals += 1
            self.defer(pending)
            return RetirementEvidenceDecision(
                pending,
                "defer",
                "trusted paired effect was inconclusive",
            )
        last_index = pending.last_evidence_index
        if last_index is not None and episode_index - last_index < self.min_evidence_span:
            self.temporal_deferrals += 1
            self.defer(pending)
            return RetirementEvidenceDecision(
                pending,
                "defer",
                "decisive evidence was too close to the previous observation",
            )
        self.evidence_observations += 1
        if direction == "confirm":
            updated = replace(
                pending,
                confirmation_indices=(*pending.confirmation_indices, episode_index),
            )
            if len(updated.confirmation_indices) >= self.min_confirmations:
                self.resolve(updated, confirmed=True)
                return RetirementEvidenceDecision(
                    updated,
                    "confirm",
                    "temporally separated current-over-old evidence reached threshold",
                    evidence_recorded=True,
                )
        else:
            updated = replace(
                pending,
                veto_indices=(*pending.veto_indices, episode_index),
            )
            if len(updated.veto_indices) >= self.min_confirmations:
                self.resolve(updated, confirmed=False)
                return RetirementEvidenceDecision(
                    updated,
                    "veto",
                    "temporally separated old-over-current evidence reached threshold",
                    evidence_recorded=True,
                )
        self.defer(updated)
        return RetirementEvidenceDecision(
            updated,
            "defer",
            "decisive evidence recorded; awaiting sequential threshold",
            evidence_recorded=True,
        )

    def pending_memory_keys(self) -> set[tuple[str, int]]:
        """Return exact versions currently in a reversible retirement transaction."""

        return {pending.memory_key for pending in self._pending.values()}

    def pending_items(self) -> tuple[PendingRetirement, ...]:
        """Return a stable snapshot for post-hoc path metrics only."""

        return tuple(self._pending.values())

    def drain_invalidated(self) -> tuple[tuple[PendingRetirement, str], ...]:
        invalidated = tuple(self._invalidated)
        self._invalidated.clear()
        return invalidated

    def expire(self, episode_index: int) -> tuple[PendingRetirement, ...]:
        expired = tuple(
            pending
            for pending in self._pending.values()
            if episode_index > pending.expires_after_index
        )
        for pending in expired:
            self._pending.pop(pending.transaction_key, None)
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
            "min_confirmations": self.min_confirmations,
            "min_evidence_span": self.min_evidence_span,
            "registrations": self.registrations,
            "interventions": self.interventions,
            "confirmations": self.confirmations,
            "cancellations": self.cancellations,
            "deferrals": self.deferrals,
            "evidence_observations": self.evidence_observations,
            "temporal_deferrals": self.temporal_deferrals,
            "low_trust_deferrals": self.low_trust_deferrals,
            "unapplied_deferrals": self.unapplied_deferrals,
            "neutral_deferrals": self.neutral_deferrals,
            "expirations": self.expirations,
            "invalidations": self.invalidations,
            "invalidated_awaiting_resolution": len(self._invalidated),
            "pending": len(self._pending),
        }


__all__ = [
    "PendingRetirement",
    "RetirementEvidenceDecision",
    "RetirementProbation",
]
