from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from evoshift.config import EvolutionConfig
from evoshift.schemas import MemoryItem, MemoryStatus


@dataclass(frozen=True)
class ActiveAuditDecision:
    memory_before: MemoryItem
    memory_after: MemoryItem
    episode_index: int
    feedback_control: float
    feedback_candidate: float
    delta: float
    retire: bool
    reason: str

    @property
    def candidate_id(self) -> str:
        return f"{self.memory_before.memory_id}@v{self.memory_before.version}"


class ActiveMemoryAuditor:
    """Budgeted learner-visible causal monitoring for already-active memories."""

    def __init__(self, config: EvolutionConfig):
        self.max_per_episode = config.active_audit_max_per_episode
        self.min_observations = config.active_audit_min_observations
        self.min_negative_observations = config.active_audit_min_negative_observations
        self.retire_mean_delta = config.active_audit_retire_mean_delta
        self.early_retire_enabled = config.active_audit_early_retire_enabled
        self.early_retire_delta = config.active_audit_early_retire_delta
        self.cooldown_episodes = config.active_audit_cooldown_episodes

    def eligible(
        self,
        memories: Iterable[MemoryItem],
        applied_memory_ids: Iterable[str],
        *,
        episode_index: int,
    ) -> list[MemoryItem]:
        applied = set(applied_memory_ids)
        eligible = [
            memory
            for memory in memories
            if memory.memory_id in applied
            and memory.status == MemoryStatus.ACTIVE
            and (
                memory.causal_last_audit_index is None
                or episode_index - memory.causal_last_audit_index > self.cooldown_episodes
            )
        ]
        return sorted(
            eligible,
            key=lambda memory: (
                memory.causal_mean_delta,
                memory.posterior_utility,
                memory.causal_audit_count,
                memory.memory_id,
                memory.version,
            ),
        )

    def select(
        self,
        memories: Iterable[MemoryItem],
        applied_memory_ids: Iterable[str],
        *,
        episode_index: int,
    ) -> list[MemoryItem]:
        return self.eligible(
            memories,
            applied_memory_ids,
            episode_index=episode_index,
        )[: self.max_per_episode]

    def observe(
        self,
        memory: MemoryItem,
        *,
        episode_index: int,
        feedback_control: float,
        feedback_candidate: float,
        shift_detected: bool = False,
        retirement_enabled: bool = True,
    ) -> ActiveAuditDecision:
        if memory.status != MemoryStatus.ACTIVE:
            raise ValueError("active memory audit requires an active memory")
        delta = float(feedback_candidate) - float(feedback_control)
        positive = int(delta > 0.0)
        negative = int(delta < 0.0)
        neutral = int(delta == 0.0)
        updated = memory.model_copy(
            update={
                "causal_audit_count": memory.causal_audit_count + 1,
                "causal_positive_count": memory.causal_positive_count + positive,
                "causal_negative_count": memory.causal_negative_count + negative,
                "causal_neutral_count": memory.causal_neutral_count + neutral,
                "causal_delta_sum": memory.causal_delta_sum + delta,
                "causal_last_audit_index": episode_index,
            }
        )
        enough_evidence = (
            updated.causal_audit_count >= self.min_observations
            and updated.causal_negative_count >= self.min_negative_observations
        )
        early_retire = (
            self.early_retire_enabled and shift_detected and delta <= self.early_retire_delta
        )
        retirement_evidence = early_retire or (
            enough_evidence and updated.causal_mean_delta <= self.retire_mean_delta
        )
        retire = retirement_enabled and retirement_evidence
        if retire:
            if early_retire:
                reason = (
                    "early retire: shift-gated learner-visible leave-one-memory-out "
                    f"delta {delta:.3f} <= {self.early_retire_delta:.3f}"
                )
            else:
                reason = (
                    "retire: learner-visible leave-one-memory-out mean delta "
                    f"{updated.causal_mean_delta:.3f} <= {self.retire_mean_delta:.3f} "
                    f"after {updated.causal_audit_count} audits"
                )
        elif retirement_evidence:
            reason = "keep active: retirement disabled for reversible causal canary"
        elif not enough_evidence:
            reason = "keep active: insufficient learner-visible causal evidence"
        else:
            reason = (
                "keep active: learner-visible leave-one-memory-out mean delta "
                f"{updated.causal_mean_delta:.3f} > {self.retire_mean_delta:.3f}"
            )
        return ActiveAuditDecision(
            memory_before=memory,
            memory_after=updated,
            episode_index=episode_index,
            feedback_control=float(feedback_control),
            feedback_candidate=float(feedback_candidate),
            delta=delta,
            retire=retire,
            reason=reason,
        )

    @staticmethod
    def revert_observation(memory: MemoryItem, observation: ActiveAuditDecision) -> MemoryItem:
        """Remove one provisional audit observation while preserving later evidence."""

        before = observation.memory_before
        after = observation.memory_after
        if (memory.memory_id, memory.version) != (before.memory_id, before.version):
            raise ValueError("cannot revert a causal observation from another memory version")
        if memory.causal_audit_count < 1:
            raise ValueError("cannot revert a causal observation from an empty ledger")
        positive = after.causal_positive_count - before.causal_positive_count
        negative = after.causal_negative_count - before.causal_negative_count
        neutral = after.causal_neutral_count - before.causal_neutral_count
        if (positive, negative, neutral).count(1) != 1 or any(
            value not in {0, 1} for value in (positive, negative, neutral)
        ):
            raise ValueError("invalid provisional causal observation")
        audit_count = memory.causal_audit_count - 1
        last_index = memory.causal_last_audit_index
        if last_index == observation.episode_index:
            last_index = before.causal_last_audit_index
        if audit_count == 0:
            last_index = None
        return memory.model_copy(
            update={
                "causal_audit_count": audit_count,
                "causal_positive_count": memory.causal_positive_count - positive,
                "causal_negative_count": memory.causal_negative_count - negative,
                "causal_neutral_count": memory.causal_neutral_count - neutral,
                "causal_delta_sum": memory.causal_delta_sum - observation.delta,
                "causal_last_audit_index": last_index,
            }
        )


__all__ = ["ActiveAuditDecision", "ActiveMemoryAuditor"]
