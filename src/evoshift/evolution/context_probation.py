from __future__ import annotations

from dataclasses import dataclass

from evoshift.config import EvolutionConfig
from evoshift.schemas import MemoryItem

ObservableContext = tuple[str, str]


@dataclass(frozen=True)
class ContextProbationLease:
    """A bounded learner-visible lease for one probationary memory/context."""

    memory: MemoryItem
    source: str
    context: str
    signal: str
    registered_context_observations: int
    registered_index: int
    expires_after_index: int
    max_uses: int
    uses: int = 0

    @property
    def memory_key(self) -> tuple[str, int]:
        return self.memory.memory_id, self.memory.version

    @property
    def observable_key(self) -> ObservableContext:
        return self.source, self.context


class ContextLocalProbation:
    """Keep probationary exposure local to the context that produced it.

    This class owns no memory lifecycle transitions. It only issues short-lived
    leases; the caller supplies the paired predictions and lets future audit
    decide whether the memory may become active.
    """

    def __init__(self, config: EvolutionConfig):
        self.max_age = config.context_local_probation_fast_path_max_age
        self.max_uses = config.context_local_probation_fast_path_max_uses
        self.min_trust = config.context_local_probation_fast_path_min_trust
        self._leases: dict[ObservableContext, ContextProbationLease] = {}
        self.registrations = 0
        self.interventions = 0
        self.expirations = 0
        self.exhaustions = 0

    def register(
        self,
        memory: MemoryItem,
        *,
        source: str,
        context: str,
        signal: str,
        context_observations: int,
        episode_index: int,
    ) -> ContextProbationLease | None:
        key = (str(source).strip(), str(context).strip().casefold())
        normalized_signal = str(signal).strip().casefold()
        if not key[0] or not key[1] or not normalized_signal or key in self._leases:
            return None
        lease = ContextProbationLease(
            memory=memory,
            source=key[0],
            context=key[1],
            signal=normalized_signal,
            registered_context_observations=max(0, int(context_observations)),
            registered_index=episode_index,
            expires_after_index=episode_index + self.max_age,
            max_uses=self.max_uses,
        )
        self._leases[key] = lease
        self.registrations += 1
        return lease

    def match(
        self,
        *,
        source: str,
        context: str,
        signal: str,
        context_observations: int,
        episode_index: int,
    ) -> ContextProbationLease | None:
        key = (str(source).strip(), str(context).strip().casefold())
        lease = self._leases.get(key)
        if lease is None or episode_index <= lease.registered_index:
            return None
        if str(signal).strip().casefold() != lease.signal:
            return None
        if int(context_observations) <= lease.registered_context_observations:
            return None
        if episode_index > lease.expires_after_index:
            self._leases.pop(key, None)
            self.expirations += 1
            return None
        if lease.uses >= lease.max_uses:
            self._leases.pop(key, None)
            self.exhaustions += 1
            return None
        return lease

    def consume(self, lease: ContextProbationLease) -> ContextProbationLease:
        current = self._leases.get(lease.observable_key)
        if current is None or current.memory_key != lease.memory_key:
            raise ValueError("context probation lease is no longer pending")
        updated = ContextProbationLease(
            memory=current.memory,
            source=current.source,
            context=current.context,
            signal=current.signal,
            registered_context_observations=current.registered_context_observations,
            registered_index=current.registered_index,
            expires_after_index=current.expires_after_index,
            max_uses=current.max_uses,
            uses=current.uses + 1,
        )
        self.interventions += 1
        if updated.uses >= updated.max_uses:
            self._leases.pop(updated.observable_key, None)
            self.exhaustions += 1
        else:
            self._leases[updated.observable_key] = updated
        return updated

    def discard(self, memory_id: str) -> None:
        for key, lease in tuple(self._leases.items()):
            if lease.memory.memory_id == memory_id:
                self._leases.pop(key, None)

    def expire(self, episode_index: int) -> int:
        expired = [
            key for key, lease in self._leases.items() if episode_index > lease.expires_after_index
        ]
        for key in expired:
            self._leases.pop(key, None)
        self.expirations += len(expired)
        return len(expired)

    def expire_all(self) -> int:
        count = len(self._leases)
        self._leases.clear()
        self.expirations += count
        return count

    def pending(self) -> tuple[ContextProbationLease, ...]:
        return tuple(self._leases.values())

    def snapshot(self) -> dict[str, int | float | bool]:
        return {
            "enabled": True,
            "max_age": self.max_age,
            "max_uses": self.max_uses,
            "min_trust": self.min_trust,
            "registrations": self.registrations,
            "interventions": self.interventions,
            "expirations": self.expirations,
            "exhaustions": self.exhaustions,
            "pending": len(self._leases),
        }


class ContextLocalProvisionalLane(ContextLocalProbation):
    """Issue leases for a confirmed, source-stable context change.

    The lane deliberately keeps the candidate outside ``MemoryManager`` until
    future counterfactual audit confirms it.  It therefore cannot enter the
    ordinary global retriever or mutate the committed trust model.
    """

    def __init__(self, config: EvolutionConfig):
        super().__init__(config.model_copy(
            update={
                "context_local_probation_fast_path_max_age": (
                    config.context_local_provisional_lane_max_age
                ),
                "context_local_probation_fast_path_max_uses": (
                    config.context_local_provisional_lane_max_uses
                ),
                "context_local_probation_fast_path_min_trust": (
                    config.context_local_provisional_lane_min_trust
                ),
            }
        ))
        self.min_source_trust = config.context_local_provisional_lane_min_source_trust
        self.min_context_observations = (
            config.context_local_provisional_lane_min_context_observations
        )
        self.match_attempts = 0
        self.match_hits = 0
        self.discard_calls = 0
        self.discarded_leases = 0

    def discard(self, memory_id: str) -> None:
        before = len(self.pending())
        super().discard(memory_id)
        self.discard_calls += 1
        self.discarded_leases += int(len(self.pending()) < before)

    def match(
        self,
        *,
        source: str,
        context: str,
        signal: str,
        context_observations: int,
        episode_index: int,
    ) -> ContextProbationLease | None:
        self.match_attempts += 1
        lease = super().match(
            source=source,
            context=context,
            signal=signal,
            context_observations=context_observations,
            episode_index=episode_index,
        )
        self.match_hits += int(lease is not None)
        return lease

    def register_from_assessment(
        self,
        memory: MemoryItem,
        *,
        source: str,
        context: str,
        signal: str,
        source_posterior_mean: float,
        context_observations: int,
        pending_observations: int,
        reason: str,
        episode_index: int,
        registered_context_observations: int | None = None,
    ) -> ContextProbationLease | None:
        """Register only a confirmed change from a mature source.

        A pending contradiction remains quarantined because it may still be
        transient noise. The lane becomes eligible only after the feedback
        model commits the new signal.
        """

        if source_posterior_mean < self.min_source_trust:
            return None
        if context_observations < self.min_context_observations:
            return None
        if pending_observations != 0:
            return None
        if reason != "dynamic_confirmed_change":
            return None
        return self.register(
            memory,
            source=source,
            context=context,
            signal=signal,
            context_observations=(
                context_observations
                if registered_context_observations is None
                else registered_context_observations
            ),
            episode_index=episode_index,
        )

    def snapshot(self) -> dict[str, int | float | bool]:
        return {
            **super().snapshot(),
            "min_source_trust": self.min_source_trust,
            "min_context_observations": self.min_context_observations,
            "match_attempts": self.match_attempts,
            "match_hits": self.match_hits,
            "discard_calls": self.discard_calls,
            "discarded_leases": self.discarded_leases,
        }


__all__ = [
    "ContextLocalProbation",
    "ContextLocalProvisionalLane",
    "ContextProbationLease",
    "ObservableContext",
]
