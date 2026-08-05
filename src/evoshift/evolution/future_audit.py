from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Iterable, Literal

from evoshift.config import EvolutionConfig
from evoshift.evaluation import PromotionGate
from evoshift.schemas import LLMUsage, MemoryItem, PromotionDecision


def _cost_proxy(usage: LLMUsage) -> float:
    return usage.cost_usd if usage.cost_usd > 0.0 else float(usage.total_tokens)


@dataclass
class PendingFutureAudit:
    memory: MemoryItem
    evidence_signature: str
    start_index: int
    feedback_control_scores: list[float] = field(default_factory=list)
    feedback_candidate_scores: list[float] = field(default_factory=list)
    oracle_control_scores: list[float] = field(default_factory=list)
    oracle_candidate_scores: list[float] = field(default_factory=list)
    control_costs: list[float] = field(default_factory=list)
    candidate_costs: list[float] = field(default_factory=list)
    protected_mask: list[bool] = field(default_factory=list)
    observation_indices: list[int] = field(default_factory=list)

    @property
    def candidate_id(self) -> str:
        return f"{self.memory.memory_id}@v{self.memory.version}"

    @property
    def n(self) -> int:
        return len(self.feedback_candidate_scores)


@dataclass(frozen=True)
class FutureAuditOutcome:
    audit: PendingFutureAudit
    decision: PromotionDecision
    completion_reason: Literal["evidence", "stream_end"] = "evidence"


class FutureCounterfactualAuditor:
    """Confirm probationary memories on later disjoint paired interactions."""

    def __init__(self, config: EvolutionConfig):
        self.min_observations = config.future_audit_min_observations
        self.max_observations = config.future_audit_max_observations
        self.early_harm_observations = config.future_audit_early_harm_observations
        gate_config = config.model_copy(
            update={"min_validation_examples": config.future_audit_min_observations}
        )
        self.gate = PromotionGate(gate_config)
        self._pending: dict[str, PendingFutureAudit] = {}

    def register(
        self,
        memory: MemoryItem,
        *,
        evidence_signature: str,
        start_index: int,
    ) -> PendingFutureAudit:
        if memory.memory_id in self._pending:
            raise ValueError(f"future audit already pending for memory: {memory.memory_id}")
        audit = PendingFutureAudit(memory, evidence_signature, start_index)
        self._pending[memory.memory_id] = audit
        return audit

    def is_pending(self, memory_id: str) -> bool:
        return memory_id in self._pending

    def pending_for(self, applied_memory_ids: Iterable[str]) -> list[PendingFutureAudit]:
        return [
            self._pending[memory_id]
            for memory_id in dict.fromkeys(applied_memory_ids)
            if memory_id in self._pending
        ]

    def record(
        self,
        audit: PendingFutureAudit,
        *,
        index: int,
        feedback_control: float,
        feedback_candidate: float,
        oracle_control: float,
        oracle_candidate: float,
        control_usage: LLMUsage,
        candidate_usage: LLMUsage,
        protected: bool,
    ) -> FutureAuditOutcome | None:
        if index <= audit.start_index:
            return None
        audit.feedback_control_scores.append(feedback_control)
        audit.feedback_candidate_scores.append(feedback_candidate)
        audit.oracle_control_scores.append(oracle_control)
        audit.oracle_candidate_scores.append(oracle_candidate)
        audit.control_costs.append(_cost_proxy(control_usage))
        audit.candidate_costs.append(_cost_proxy(candidate_usage))
        audit.protected_mask.append(protected)
        audit.observation_indices.append(index)

        deltas = [
            candidate - control
            for control, candidate in zip(
                audit.feedback_control_scores,
                audit.feedback_candidate_scores,
            )
        ]
        if (
            self.early_harm_observations > 0
            and audit.n >= self.early_harm_observations
            and statistics.fmean(deltas) < 0.0
        ):
            return self._complete(
                audit,
                reason_override=(
                    "early rollback: trusted future counterfactual mean utility is negative"
                ),
            )
        if audit.n < self.min_observations:
            return None
        if audit.n == self.min_observations:
            decision = self._decide(audit)
            clearly_regressive = not decision.gate_checks.get(
                "regression_rate", True
            ) or not decision.gate_checks.get("protected_slice", True)
            if decision.promote or clearly_regressive or audit.n >= self.max_observations:
                return self._complete(audit, decision=decision)
            return None
        if audit.n >= self.max_observations:
            return self._complete(audit)
        return None

    def finalize(self) -> list[FutureAuditOutcome]:
        return [
            self._complete(
                audit,
                reason_override="expired: stream ended before future audit reached a decision",
                completion_reason="stream_end",
            )
            for audit in list(self._pending.values())
        ]

    def pending(self) -> tuple[PendingFutureAudit, ...]:
        return tuple(self._pending.values())

    def _decide(self, audit: PendingFutureAudit) -> PromotionDecision:
        return self.gate.decide(
            candidate_id=audit.candidate_id,
            candidate_type="memory_future_audit",
            control_scores=audit.feedback_control_scores,
            candidate_scores=audit.feedback_candidate_scores,
            control_costs=audit.control_costs,
            candidate_costs=audit.candidate_costs,
            protected_mask=audit.protected_mask,
        )

    def _complete(
        self,
        audit: PendingFutureAudit,
        *,
        decision: PromotionDecision | None = None,
        reason_override: str | None = None,
        completion_reason: Literal["evidence", "stream_end"] = "evidence",
    ) -> FutureAuditOutcome:
        decision = decision or self._decide(audit)
        if reason_override is not None:
            decision = decision.model_copy(update={"reason": reason_override})
        oracle_control = (
            statistics.fmean(audit.oracle_control_scores) if audit.oracle_control_scores else None
        )
        oracle_candidate = (
            statistics.fmean(audit.oracle_candidate_scores)
            if audit.oracle_candidate_scores
            else None
        )
        oracle_delta = (
            oracle_candidate - oracle_control
            if oracle_candidate is not None and oracle_control is not None
            else None
        )
        result = decision.result.model_copy(
            update={
                "oracle_control_mean": oracle_control,
                "oracle_candidate_mean": oracle_candidate,
                "oracle_mean_delta": oracle_delta,
                "observation_start_index": (
                    audit.observation_indices[0] if audit.observation_indices else None
                ),
                "observation_end_index": (
                    audit.observation_indices[-1] if audit.observation_indices else None
                ),
            }
        )
        completed = decision.model_copy(update={"result": result})
        self._pending.pop(audit.memory.memory_id, None)
        return FutureAuditOutcome(
            audit=audit,
            decision=completed,
            completion_reason=completion_reason,
        )


__all__ = [
    "FutureAuditOutcome",
    "FutureCounterfactualAuditor",
    "PendingFutureAudit",
]
