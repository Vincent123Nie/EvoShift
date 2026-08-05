import pytest

from evoshift.config import EvolutionConfig
from evoshift.evolution import FutureCounterfactualAuditor
from evoshift.schemas import LLMUsage, MemoryItem, MemoryStatus


def _config() -> EvolutionConfig:
    return EvolutionConfig(
        future_audit_enabled=True,
        future_audit_min_observations=2,
        future_audit_max_observations=4,
        future_audit_early_harm_observations=1,
        min_mean_gain=0.0,
        min_ci_lower_bound=0.0,
        max_regression_rate=0.0,
        max_protected_slice_regression=0.0,
        max_cost_increase_ratio=1.0,
    )


def _memory() -> MemoryItem:
    return MemoryItem(
        memory_id="candidate",
        status=MemoryStatus.PROBATION,
        trigger="refund days 8 to 14",
        directive="Approve through day 14.",
    )


def test_future_audit_confirms_realized_benefit_and_records_oracle_delta() -> None:
    auditor = FutureCounterfactualAuditor(_config())
    audit = auditor.register(_memory(), evidence_signature="sig", start_index=4)
    usage = LLMUsage(total_tokens=10)

    assert (
        auditor.record(
            audit,
            index=5,
            feedback_control=0.0,
            feedback_candidate=1.0,
            oracle_control=0.0,
            oracle_candidate=1.0,
            control_usage=usage,
            candidate_usage=usage,
            protected=False,
        )
        is None
    )
    outcome = auditor.record(
        audit,
        index=6,
        feedback_control=0.0,
        feedback_candidate=1.0,
        oracle_control=0.0,
        oracle_candidate=1.0,
        control_usage=usage,
        candidate_usage=usage,
        protected=False,
    )

    assert outcome is not None
    assert outcome.decision.promote is True
    assert outcome.decision.result.oracle_mean_delta == 1.0
    assert outcome.decision.result.observation_start_index == 5
    assert outcome.decision.result.observation_end_index == 6
    assert auditor.pending() == ()


def test_future_audit_rejects_harmful_candidate_after_one_trusted_pair() -> None:
    auditor = FutureCounterfactualAuditor(_config())
    audit = auditor.register(_memory(), evidence_signature="sig", start_index=4)
    usage = LLMUsage(total_tokens=10)
    outcome = auditor.record(
        audit,
        index=5,
        feedback_control=1.0,
        feedback_candidate=0.0,
        oracle_control=1.0,
        oracle_candidate=0.0,
        control_usage=usage,
        candidate_usage=usage,
        protected=False,
    )

    assert outcome is not None
    assert outcome.decision.promote is False
    assert outcome.decision.reason.startswith("early rollback")
    assert outcome.decision.gate_checks["minimum_examples"] is False
    assert outcome.decision.result.oracle_mean_delta == -1.0
    assert outcome.decision.result.observation_start_index == 5
    assert outcome.decision.result.observation_end_index == 5


def test_future_audit_accumulates_inconclusive_minimum_until_maximum() -> None:
    config = _config().model_copy(
        update={
            "future_audit_early_harm_observations": 0,
            "min_mean_gain": 0.6,
            "max_regression_rate": 1.0,
        }
    )
    auditor = FutureCounterfactualAuditor(config)
    audit = auditor.register(_memory(), evidence_signature="sig", start_index=4)
    usage = LLMUsage(total_tokens=10)

    for index in (5, 6, 7):
        assert (
            auditor.record(
                audit,
                index=index,
                feedback_control=0.0,
                feedback_candidate=0.5,
                oracle_control=0.0,
                oracle_candidate=0.5,
                control_usage=usage,
                candidate_usage=usage,
                protected=False,
            )
            is None
        )

    outcome = auditor.record(
        audit,
        index=8,
        feedback_control=0.0,
        feedback_candidate=0.5,
        oracle_control=0.0,
        oracle_candidate=0.5,
        control_usage=usage,
        candidate_usage=usage,
        protected=False,
    )
    assert outcome is not None
    assert outcome.decision.promote is False
    assert outcome.decision.result.n == 4


def test_future_audit_marks_incomplete_stream_end_as_expired() -> None:
    auditor = FutureCounterfactualAuditor(_config())
    audit = auditor.register(_memory(), evidence_signature="sig", start_index=4)
    usage = LLMUsage(total_tokens=10)
    assert (
        auditor.record(
            audit,
            index=5,
            feedback_control=0.0,
            feedback_candidate=1.0,
            oracle_control=0.0,
            oracle_candidate=1.0,
            control_usage=usage,
            candidate_usage=usage,
            protected=False,
        )
        is None
    )

    outcomes = auditor.finalize()

    assert len(outcomes) == 1
    assert outcomes[0].completion_reason == "stream_end"
    assert outcomes[0].decision.promote is False
    assert outcomes[0].decision.reason.startswith("expired")


def test_future_audit_rejects_a_second_pending_version_of_the_same_memory() -> None:
    auditor = FutureCounterfactualAuditor(_config())
    auditor.register(_memory(), evidence_signature="sig-v1", start_index=4)
    newer = _memory().model_copy(update={"version": 2})

    assert auditor.is_pending(newer.memory_id) is True
    with pytest.raises(ValueError, match="already pending"):
        auditor.register(newer, evidence_signature="sig-v2", start_index=5)
