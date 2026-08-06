import pytest

from evoshift.config import EvolutionConfig


def test_retirement_context_bound_revival_requires_dormant_revival() -> None:
    with pytest.raises(
        ValueError,
        match="retirement-context-bound revival requires dormant memory revival",
    ):
        EvolutionConfig(dormant_revival_retirement_context_enabled=True)


def test_retirement_context_bound_revival_accepts_complete_dependency_chain() -> None:
    config = EvolutionConfig(
        dynamic_feedback_trust_enabled=True,
        dormant_revival_enabled=True,
        dormant_revival_retirement_context_enabled=True,
    )

    assert config.dormant_revival_retirement_context_enabled


def test_context_scoped_retirement_requires_probation() -> None:
    with pytest.raises(
        ValueError,
        match="context-scoped retirement probation requires retirement probation",
    ):
        EvolutionConfig(active_audit_retirement_probation_context_scoped=True)


def test_context_scoped_retirement_accepts_complete_dependency_chain() -> None:
    config = EvolutionConfig(
        active_audit_enabled=True,
        active_audit_retirement_probation_enabled=True,
        active_audit_retirement_probation_context_scoped=True,
    )

    assert config.active_audit_retirement_probation_context_scoped


def test_fast_retirement_confirmation_requires_probation() -> None:
    with pytest.raises(
        ValueError,
        match="fast retirement confirmation requires retirement probation",
    ):
        EvolutionConfig(active_audit_retirement_probation_fast_confirm_enabled=True)


def test_fast_revival_cooldown_is_a_nonnegative_evolution_setting() -> None:
    config = EvolutionConfig(
        active_audit_enabled=True,
        active_audit_retirement_probation_enabled=True,
        active_audit_retirement_probation_fast_revival_cooldown_episodes=13,
    )

    assert config.active_audit_retirement_probation_fast_revival_cooldown_episodes == 13


def test_fast_confirmation_trust_floor_is_bounded() -> None:
    config = EvolutionConfig(
        active_audit_enabled=True,
        active_audit_retirement_probation_enabled=True,
        active_audit_retirement_probation_fast_confirm_min_trust=0.94,
    )

    assert config.active_audit_retirement_probation_fast_confirm_min_trust == 0.94


def test_candidate_mature_feedback_gate_is_nonnegative() -> None:
    config = EvolutionConfig(candidate_min_mature_feedback_observations=1)

    assert config.candidate_min_mature_feedback_observations == 1


def test_weighted_candidate_evidence_settings_are_bounded() -> None:
    config = EvolutionConfig(
        candidate_min_weighted_mature_evidence=2.5,
        candidate_provisional_evidence_weight=0.5,
    )

    assert config.candidate_min_weighted_mature_evidence == 2.5
    assert config.candidate_provisional_evidence_weight == 0.5


def test_historical_replay_anchors_require_at_least_one_exact_quota_slot() -> None:
    with pytest.raises(ValueError, match="historical replay anchors require"):
        EvolutionConfig(
            validation_window=2,
            replay_historical_context_anchors_enabled=True,
            replay_historical_context_anchor_fraction=1.0 / 3.0,
        )


@pytest.mark.parametrize(
    "hidden_field",
    [
        "phase_index",
        "policy_version",
        "protected",
        "valid_memory_tags",
        "stale_memory_tags",
        "feedback_kind",
        "feedback_corrupted",
    ],
)
def test_dynamic_feedback_context_rejects_oracle_metadata_fields(hidden_field: str) -> None:
    with pytest.raises(ValueError, match="typed learner-visible feedback_context"):
        EvolutionConfig(dynamic_feedback_context_field=hidden_field)
