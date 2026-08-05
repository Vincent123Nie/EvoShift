from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from evoshift.config import EvolutionConfig
from evoshift.schemas import MemoryItem, MemoryStatus, RetrievedMemory

MemoryVersionKey = tuple[str, int]


def dormant_candidate_keys(
    memories: Iterable[MemoryItem],
    retired_indices: Mapping[MemoryVersionKey, int],
    *,
    semantic_indexed: bool,
) -> set[MemoryVersionKey]:
    """Build the exact retired-version view used by recurrence probes.

    The legacy view keeps only the most recently retired card in each scope.
    The semantic view instead keeps the latest retired version of every memory
    ID, allowing retrieval relevance to decide between competing rules.
    """

    latest: dict[str, MemoryItem] = {}
    for item in memories:
        key = (item.memory_id, item.version)
        if item.status != MemoryStatus.RETIRED or key not in retired_indices:
            continue
        grouping_key = item.memory_id if semantic_indexed else item.scope
        previous = latest.get(grouping_key)
        if previous is None:
            latest[grouping_key] = item
            continue
        previous_key = (previous.memory_id, previous.version)
        if semantic_indexed:
            replace = item.version > previous.version or (
                item.version == previous.version
                and retired_indices[key] > retired_indices[previous_key]
            )
        else:
            replace = retired_indices[key] > retired_indices[previous_key]
        if replace:
            latest[grouping_key] = item
    return {(item.memory_id, item.version) for item in latest.values()}


def order_semantic_dormant_candidates(
    candidates: Sequence[RetrievedMemory],
    retired_indices: Mapping[MemoryVersionKey, int],
) -> list[RetrievedMemory]:
    """Order by retrieval score, using retirement recency only for exact ties."""

    return sorted(
        candidates,
        key=lambda result: (
            -result.final_score,
            -retired_indices.get((result.item.memory_id, result.item.version), -1),
            result.item.memory_id,
            -result.item.version,
        ),
    )


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
        self.semantic_indexed = config.dormant_revival_semantic_index_enabled
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
            "semantic_indexed": self.semantic_indexed,
            "probes": self.probes,
            "registrations": self.registrations,
            "interventions": self.interventions,
            "confirmations": self.confirmations,
            "cancellations": self.cancellations,
            "expirations": self.expirations,
            "invalidations": self.invalidations,
            "pending": len(self._pending),
        }


__all__ = [
    "DormantMemoryRevival",
    "PendingRevivalCanary",
    "dormant_candidate_keys",
    "order_semantic_dormant_candidates",
]
