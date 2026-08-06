import inspect

import pytest

from evoshift.config import EvolutionConfig
from evoshift.evolution import ActiveMemoryAuditor
from evoshift.schemas import MemoryItem, MemoryStatus


def _memory(memory_id: str = "active") -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        status=MemoryStatus.ACTIVE,
        trigger="refund after day seven",
        directive="Approve refunds through day fourteen.",
    )


def _auditor(**updates: object) -> ActiveMemoryAuditor:
    config = EvolutionConfig(
        active_audit_enabled=True,
        active_audit_min_observations=2,
        active_audit_min_negative_observations=2,
        active_audit_retire_mean_delta=-0.5,
    ).model_copy(update=updates)
    return ActiveMemoryAuditor(config)


def test_active_audit_retires_only_after_repeated_learner_visible_harm() -> None:
    auditor = _auditor()
    first = auditor.observe(
        _memory(),
        episode_index=10,
        feedback_control=1.0,
        feedback_candidate=0.0,
    )

    assert first.retire is False
    assert first.memory_after.causal_negative_count == 1
    assert first.memory_after.causal_mean_delta == -1.0

    second = auditor.observe(
        first.memory_after,
        episode_index=12,
        feedback_control=1.0,
        feedback_candidate=0.0,
    )

    assert second.retire is True
    assert second.reason.startswith("retire: learner-visible")
    assert second.memory_after.causal_audit_count == 2
    assert second.memory_after.causal_negative_count == 2
    assert second.memory_after.causal_delta_sum == -2.0


def test_active_audit_does_not_accept_oracle_evidence_or_retire_on_neutral_controls() -> None:
    assert all(
        "oracle" not in parameter
        for parameter in inspect.signature(ActiveMemoryAuditor.observe).parameters
    )
    auditor = _auditor()
    first = auditor.observe(
        _memory(),
        episode_index=3,
        feedback_control=0.0,
        feedback_candidate=0.0,
    )
    second = auditor.observe(
        first.memory_after,
        episode_index=4,
        feedback_control=0.0,
        feedback_candidate=0.0,
    )

    assert second.retire is False
    assert second.memory_after.causal_neutral_count == 2
    assert second.memory_after.causal_negative_count == 0


def test_shift_gated_sequential_retirement_requires_strong_harm() -> None:
    auditor = _auditor(
        active_audit_early_retire_enabled=True,
        active_audit_early_retire_delta=-0.75,
    )
    not_shift = auditor.observe(
        _memory("no-shift"),
        episode_index=10,
        feedback_control=1.0,
        feedback_candidate=0.0,
        shift_detected=False,
    )
    assert not_shift.retire is False

    early = auditor.observe(
        _memory("shift"),
        episode_index=10,
        feedback_control=1.0,
        feedback_candidate=0.0,
        shift_detected=True,
    )
    assert early.retire is True
    assert early.reason.startswith("early retire:")

    mild = auditor.observe(
        _memory("mild"),
        episode_index=10,
        feedback_control=1.0,
        feedback_candidate=0.4,
        shift_detected=True,
    )
    assert mild.retire is False


def test_provisional_causal_observation_cannot_retire_and_is_exactly_reversible() -> None:
    auditor = _auditor(
        active_audit_min_observations=1,
        active_audit_min_negative_observations=1,
        active_audit_early_retire_enabled=True,
    )
    original = _memory("reversible")
    provisional = auditor.observe(
        original,
        episode_index=10,
        feedback_control=1.0,
        feedback_candidate=0.0,
        shift_detected=True,
        retirement_enabled=False,
    )

    assert provisional.retire is False
    assert provisional.reason == "keep active: retirement disabled for reversible causal canary"
    assert provisional.memory_after.causal_negative_count == 1
    assert auditor.revert_observation(provisional.memory_after, provisional) == original


def test_reverting_provisional_observation_preserves_later_causal_evidence() -> None:
    auditor = _auditor()
    provisional = auditor.observe(
        _memory("interleaved"),
        episode_index=10,
        feedback_control=1.0,
        feedback_candidate=0.0,
        retirement_enabled=False,
    )
    later = auditor.observe(
        provisional.memory_after,
        episode_index=12,
        feedback_control=0.0,
        feedback_candidate=1.0,
    )

    reverted = auditor.revert_observation(later.memory_after, provisional)

    assert reverted.causal_audit_count == 1
    assert reverted.causal_positive_count == 1
    assert reverted.causal_negative_count == 0
    assert reverted.causal_delta_sum == 1.0
    assert reverted.causal_last_audit_index == 12


def test_active_audit_configuration_rejects_impossible_negative_evidence_gate() -> None:
    with pytest.raises(ValueError, match="negative_observations"):
        EvolutionConfig(
            active_audit_min_observations=1,
            active_audit_min_negative_observations=2,
        )


def test_circuit_breaker_configuration_requires_dynamic_trust_and_lower_floor() -> None:
    with pytest.raises(ValueError, match="requires active audit"):
        EvolutionConfig(active_audit_circuit_breaker_enabled=True)
    with pytest.raises(ValueError, match="requires dynamic feedback trust"):
        EvolutionConfig(
            active_audit_enabled=True,
            active_audit_circuit_breaker_enabled=True,
        )
    with pytest.raises(ValueError, match="trust floor"):
        EvolutionConfig(
            active_audit_enabled=True,
            dynamic_feedback_trust_enabled=True,
            active_audit_circuit_breaker_enabled=True,
            active_audit_circuit_breaker_min_trust=0.60,
            min_feedback_trust_for_active_audit=0.60,
        )
    with pytest.raises(ValueError, match="lineage control requires"):
        EvolutionConfig(active_audit_lineage_control_enabled=True)


def test_dormant_revival_configuration_requires_dynamic_trust_and_lower_floor() -> None:
    with pytest.raises(ValueError, match="requires dynamic feedback trust"):
        EvolutionConfig(dormant_revival_enabled=True)
    with pytest.raises(ValueError, match="requires dormant memory revival"):
        EvolutionConfig(dormant_revival_status_index_enabled=True)
    with pytest.raises(ValueError, match="revival trust floor"):
        EvolutionConfig(
            dynamic_feedback_trust_enabled=True,
            dormant_revival_enabled=True,
            dormant_revival_min_trust=0.60,
            min_feedback_trust_for_active_audit=0.60,
        )


def test_active_audit_selection_is_applied_active_exact_and_budgeted() -> None:
    auditor = _auditor(active_audit_max_per_episode=1, active_audit_cooldown_episodes=2)
    risky = _memory("risky").model_copy(
        update={
            "causal_audit_count": 1,
            "causal_negative_count": 1,
            "causal_delta_sum": -1.0,
            "causal_last_audit_index": 4,
        }
    )
    fresh = _memory("fresh")
    probation = _memory("probation").model_copy(update={"status": MemoryStatus.PROBATION})

    selected = auditor.select(
        [risky, fresh, probation],
        ["risky", "fresh", "probation"],
        episode_index=5,
    )

    assert [memory.memory_id for memory in selected] == ["fresh"]
