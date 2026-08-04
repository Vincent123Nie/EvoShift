from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Tuple

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


class MemoryManager:
    def __init__(self, store: SQLiteStore, retriever: Optional[BM25MemoryRetriever] = None):
        self.store = store
        self.retriever = retriever or BM25MemoryRetriever()

    @staticmethod
    def stable_memory_id(trigger: str, directive: str) -> str:
        digest = hashlib.sha256(f"{trigger}\n{directive}".encode("utf-8")).hexdigest()[:16]
        return f"mem-{digest}"

    def active(self) -> List[MemoryItem]:
        return self.store.list_memories([MemoryStatus.ACTIVE])

    def retrieve(
        self, query: str, policy: PolicyGenome, *, domain: str = ""
    ) -> Tuple[List[RetrievedMemory], str, float]:
        active = self.active()
        selected = self.retriever.retrieve(query, active, policy, domain=domain)
        context = render_memory_context(selected, policy.memory_token_budget)
        novelty = self.retriever.novelty(query, active, policy, domain=domain)
        return selected, context, novelty

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
            merged = candidate.model_copy(
                update={
                    "memory_id": nearest.memory_id,
                    "version": nearest.version + 1,
                    "status": MemoryStatus.SHADOW,
                    "provenance_episode_ids": provenance,
                    "source_domains": source_domains,
                    "alpha": nearest.alpha,
                    "beta": nearest.beta,
                    "use_count": nearest.use_count,
                    "success_count": nearest.success_count,
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

    def activate(
        self,
        candidate: MemoryItem,
        gain: float,
        lcb: float,
        regression_rate: float,
    ) -> MemoryItem:
        active = candidate.model_copy(
            update={
                "status": MemoryStatus.ACTIVE,
                "validation_gain": gain,
                "validation_lcb": lcb,
                "regression_rate": regression_rate,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self.store.save_memory(active)
        return active

    def reject(self, candidate: MemoryItem) -> MemoryItem:
        rejected = candidate.model_copy(
            update={"status": MemoryStatus.REJECTED, "updated_at": datetime.now(timezone.utc)}
        )
        self.store.save_memory(rejected)
        return rejected

    def record_outcome(
        self, memory_ids: Sequence[str], success: bool, policy: PolicyGenome
    ) -> List[MemoryItem]:
        rolled_back: List[MemoryItem] = []
        for memory_id in memory_ids:
            current = self.store.get_memory(memory_id)
            if current is None or current.status != MemoryStatus.ACTIVE:
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
                updated.use_count >= policy.rollback_min_uses
                and updated.posterior_utility < policy.rollback_utility_threshold
            ):
                updated = updated.model_copy(update={"status": MemoryStatus.RETIRED})
                rolled_back.append(updated)
            self.store.save_memory(updated)
        return rolled_back
