from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence, Tuple

from evoshift.memory.retriever import (
    BM25MemoryRetriever,
    memory_document,
    render_memory_context,
    tokenize,
)
from evoshift.schemas import MemoryItem, MemoryStatus, PolicyGenome, RetrievedMemory
from evoshift.storage import SQLiteStore


def _similarity(left: MemoryItem, right: MemoryItem) -> float:
    left_tokens = set(tokenize(memory_document(left)))
    right_tokens = set(tokenize(memory_document(right)))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


@dataclass(frozen=True)
class MemoryActivation:
    active: MemoryItem
    superseded: tuple[MemoryItem, ...] = ()


@dataclass(frozen=True)
class MemoryOutcome:
    rolled_back: tuple[MemoryItem, ...] = ()
    reactivated: tuple[MemoryItem, ...] = ()
    retirement_protected: tuple[MemoryItem, ...] = ()
    # Keep the predecessor set paired with each posterior retirement.  A flat
    # ``reactivated`` tuple is convenient for legacy callers, but it is not
    # sufficient to undo one retirement transaction without touching an
    # unrelated lineage when several memories are updated in one episode.
    retirement_predecessors: tuple[tuple[MemoryItem, ...], ...] = ()


class MemoryManager:
    def __init__(
        self,
        store: SQLiteStore,
        retriever: Optional[BM25MemoryRetriever] = None,
        *,
        status_indexed_revival: bool = False,
    ):
        self.store = store
        self.retriever = retriever or BM25MemoryRetriever()
        self.status_indexed_revival = status_indexed_revival

    @staticmethod
    def stable_memory_id(trigger: str, directive: str) -> str:
        digest = hashlib.sha256(f"{trigger}\n{directive}".encode("utf-8")).hexdigest()[:16]
        return f"mem-{digest}"

    def active(self) -> List[MemoryItem]:
        return self.store.list_memories([MemoryStatus.ACTIVE, MemoryStatus.PROBATION])

    def retrieve(
        self, query: str, policy: PolicyGenome, *, domain: str = ""
    ) -> Tuple[List[RetrievedMemory], str, float]:
        active = self.active()
        selected = self.retriever.retrieve(query, active, policy, domain=domain)
        context = render_memory_context(selected, policy.memory_token_budget)
        novelty = self.retriever.novelty(query, active, policy, domain=domain)
        return selected, context, novelty

    def rank_memories(
        self,
        query: str,
        memories: Sequence[MemoryItem],
        policy: PolicyGenome,
        *,
        domain: str = "",
    ) -> List[RetrievedMemory]:
        return self.retriever.retrieve(query, memories, policy, domain=domain)

    def direct_superseded_predecessors(self, successor: MemoryItem) -> List[MemoryItem]:
        """Return exact, verified direct predecessors eligible as causal controls."""

        predecessor_ids = set(successor.supersedes_memory_ids)
        if not predecessor_ids:
            return []
        return sorted(
            (
                item
                for item in self.store.list_memories([MemoryStatus.SUPERSEDED])
                if item.memory_id in predecessor_ids and item.scope == successor.scope
            ),
            key=lambda item: (item.memory_id, -item.version),
        )

    def stage(self, candidate: MemoryItem, policy: PolicyGenome) -> MemoryItem:
        if candidate.confidence < policy.write_confidence_threshold:
            candidate = candidate.model_copy(update={"status": MemoryStatus.REJECTED})
            self.store.save_memory(candidate)
            return candidate
        all_items = self.store.list_memories()
        nearest = max(all_items, key=lambda item: _similarity(candidate, item), default=None)
        if (
            nearest is not None
            and _similarity(candidate, nearest) >= policy.dedup_similarity_threshold
        ):
            provenance = list(
                dict.fromkeys(nearest.provenance_episode_ids + candidate.provenance_episode_ids)
            )
            source_domains = list(dict.fromkeys(nearest.source_domains + candidate.source_domains))
            supersedes = list(
                dict.fromkeys(nearest.supersedes_memory_ids + candidate.supersedes_memory_ids)
            )
            merged = candidate.model_copy(
                update={
                    "memory_id": nearest.memory_id,
                    "version": nearest.version + 1,
                    "status": MemoryStatus.SHADOW,
                    "provenance_episode_ids": provenance,
                    "source_domains": source_domains,
                    "supersedes_memory_ids": supersedes,
                    "alpha": nearest.alpha,
                    "beta": nearest.beta,
                    "use_count": nearest.use_count,
                    "success_count": nearest.success_count,
                    "causal_audit_count": 0,
                    "causal_positive_count": 0,
                    "causal_negative_count": 0,
                    "causal_neutral_count": 0,
                    "causal_delta_sum": 0.0,
                    "causal_last_audit_index": None,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self.store.save_memory(merged)
            return merged
        if not candidate.memory_id:
            candidate = candidate.model_copy(
                update={"memory_id": self.stable_memory_id(candidate.trigger, candidate.directive)}
            )
        candidate = candidate.model_copy(update={"status": MemoryStatus.SHADOW})
        self.store.save_memory(candidate)
        return candidate

    def probation(
        self,
        candidate: MemoryItem,
        gain: float,
        lcb: float,
        regression_rate: float,
    ) -> MemoryItem:
        probationary = candidate.model_copy(
            update={
                "status": MemoryStatus.PROBATION,
                "validation_gain": gain,
                "validation_lcb": lcb,
                "regression_rate": regression_rate,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self.store.save_memory(probationary)
        return probationary

    def activate(
        self,
        candidate: MemoryItem,
        gain: float,
        lcb: float,
        regression_rate: float,
        *,
        apply_supersession: bool = True,
    ) -> MemoryActivation:
        superseded: list[MemoryItem] = []
        if apply_supersession:
            for memory_id in candidate.supersedes_memory_ids:
                if memory_id == candidate.memory_id:
                    continue
                previous = self._latest_status_memory(
                    memory_id,
                    [MemoryStatus.ACTIVE],
                )
                if previous is None or previous.status != MemoryStatus.ACTIVE:
                    continue
                replaced = previous.model_copy(
                    update={
                        "status": MemoryStatus.SUPERSEDED,
                        "updated_at": datetime.now(timezone.utc),
                    }
                )
                superseded.append(replaced)
        active = candidate.model_copy(
            update={
                "status": MemoryStatus.ACTIVE,
                "validation_gain": gain,
                "validation_lcb": lcb,
                "regression_rate": regression_rate,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self.store.save_memories_atomic([*superseded, active])
        return MemoryActivation(active=active, superseded=tuple(superseded))

    def reject(self, candidate: MemoryItem) -> MemoryItem:
        rejected = candidate.model_copy(
            update={"status": MemoryStatus.REJECTED, "updated_at": datetime.now(timezone.utc)}
        )
        self.store.save_memory(rejected)
        return rejected

    def record_outcome(
        self,
        memory_ids: Sequence[str],
        success: bool,
        policy: PolicyGenome,
        *,
        retirement_protected_versions: Iterable[tuple[str, int]] = (),
    ) -> MemoryOutcome:
        rolled_back: List[MemoryItem] = []
        reactivated: List[MemoryItem] = []
        retirement_protected: List[MemoryItem] = []
        retirement_predecessors: List[tuple[MemoryItem, ...]] = []
        protected_versions = set(retirement_protected_versions)
        for memory_id in memory_ids:
            current = self._latest_status_memory(
                memory_id,
                [MemoryStatus.ACTIVE, MemoryStatus.PROBATION],
            )
            if current is None or current.status not in {
                MemoryStatus.ACTIVE,
                MemoryStatus.PROBATION,
            }:
                continue
            updated = current.model_copy(
                update={
                    "alpha": current.alpha + (1.0 if success else 0.0),
                    "beta": current.beta + (0.0 if success else 1.0),
                    "use_count": current.use_count + 1,
                    "success_count": current.success_count + int(success),
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            if (
                current.status == MemoryStatus.ACTIVE
                and updated.use_count >= policy.rollback_min_uses
                and updated.posterior_utility < policy.rollback_utility_threshold
            ):
                if (current.memory_id, current.version) in protected_versions:
                    retirement_protected.append(updated)
                else:
                    updated = updated.model_copy(update={"status": MemoryStatus.RETIRED})
                    rolled_back.append(updated)
                    restored = tuple(self._restore_predecessors(updated))
                    retirement_predecessors.append(restored)
                    reactivated.extend(restored)
            self.store.save_memory(updated)
        return MemoryOutcome(
            tuple(rolled_back),
            tuple(reactivated),
            tuple(retirement_protected),
            tuple(retirement_predecessors),
        )

    def apply_active_audit(
        self,
        updated: MemoryItem,
        *,
        retire: bool,
        restore_predecessors: bool = False,
    ) -> MemoryOutcome:
        current = self.store.get_memory(updated.memory_id, version=updated.version)
        latest = self._latest_status_memory(
            updated.memory_id,
            [MemoryStatus.ACTIVE],
        )
        if (
            current is None
            or latest is None
            or latest.version != updated.version
            or current.status != MemoryStatus.ACTIVE
        ):
            return MemoryOutcome()
        if not retire:
            self.store.save_memory(updated)
            return MemoryOutcome()
        retired = updated.model_copy(
            update={
                "status": MemoryStatus.RETIRED,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        restored = self._restorable_predecessors(retired) if restore_predecessors else []
        self.store.save_memories_atomic([retired, *restored])
        return MemoryOutcome(
            rolled_back=(retired,),
            reactivated=tuple(restored),
            retirement_predecessors=(tuple(restored),),
        )

    def reactivate_retired(self, item: MemoryItem) -> Optional[MemoryItem]:
        current = self.store.get_memory(item.memory_id, version=item.version)
        latest = (
            self._latest_status_memory(item.memory_id, [MemoryStatus.RETIRED])
            if self.status_indexed_revival
            else self.store.get_memory(item.memory_id)
        )
        if (
            current is None
            or latest is None
            or latest.version != item.version
            or current.status != MemoryStatus.RETIRED
        ):
            return None
        reactivated = current.model_copy(
            update={
                "status": MemoryStatus.ACTIVE,
                "causal_audit_count": 0,
                "causal_positive_count": 0,
                "causal_negative_count": 0,
                "causal_neutral_count": 0,
                "causal_delta_sum": 0.0,
                "causal_last_audit_index": None,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self.store.save_memory(reactivated)
        return reactivated

    def rollback_retirement(
        self,
        retired: MemoryItem,
        restored_predecessors: Sequence[MemoryItem],
    ) -> Optional[MemoryItem]:
        """Undo one exact retirement transaction and its predecessor restores."""

        current = self.store.get_memory(retired.memory_id, version=retired.version)
        if current is None or current.status != MemoryStatus.RETIRED:
            return None
        active = current.model_copy(
            update={
                "status": MemoryStatus.ACTIVE,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        superseded: list[MemoryItem] = []
        for predecessor in restored_predecessors:
            current_predecessor = self.store.get_memory(
                predecessor.memory_id,
                version=predecessor.version,
            )
            if (
                current_predecessor is not None
                and current_predecessor.status == MemoryStatus.ACTIVE
            ):
                superseded.append(
                    current_predecessor.model_copy(
                        update={
                            "status": MemoryStatus.SUPERSEDED,
                            "updated_at": datetime.now(timezone.utc),
                        }
                    )
                )
        self.store.save_memories_atomic([*superseded, active])
        return active

    def _restore_predecessors(self, successor: MemoryItem) -> list[MemoryItem]:
        restored_items = self._restorable_predecessors(successor)
        self.store.save_memories_atomic(restored_items)
        return restored_items

    def _restorable_predecessors(self, successor: MemoryItem) -> list[MemoryItem]:
        restored_items: list[MemoryItem] = []
        for superseded_id in successor.supersedes_memory_ids:
            previous = self._latest_status_memory(
                superseded_id,
                [MemoryStatus.SUPERSEDED],
            )
            if previous is None or previous.status != MemoryStatus.SUPERSEDED:
                continue
            restored = previous.model_copy(
                update={
                    "status": MemoryStatus.ACTIVE,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            restored_items.append(restored)
        return restored_items

    def _latest_status_memory(
        self,
        memory_id: str,
        statuses: Sequence[MemoryStatus],
    ) -> Optional[MemoryItem]:
        return next(
            (item for item in self.store.list_memories(statuses) if item.memory_id == memory_id),
            None,
        )


__all__ = ["MemoryActivation", "MemoryManager", "MemoryOutcome"]
