from pathlib import Path

import pytest
from pydantic import ValidationError

from evoshift.config import EvolutionConfig, load_config
from evoshift.sweep import expand_sweep, load_sweep_spec


def test_all_shipped_yaml_configs_parse() -> None:
    root = Path.cwd()
    paths = sorted((root / "configs").rglob("*.yaml"))

    assert len(paths) >= 68
    for path in paths:
        if "sweeps" in path.parts:
            assert expand_sweep(load_sweep_spec(path, root))
        else:
            load_config(path)


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


def test_context_local_probation_requires_dynamic_feedback_trust() -> None:
    with pytest.raises(
        ValidationError,
        match="context-local probation fast path requires dynamic feedback trust",
    ):
        EvolutionConfig(
            context_local_probation_fast_path_enabled=True,
            future_audit_enabled=True,
            paired_replay=True,
            dynamic_feedback_trust_enabled=False,
        )


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


def test_change_point_posterior_gate_requires_dynamic_feedback() -> None:
    with pytest.raises(ValidationError, match="requires dynamic feedback trust"):
        EvolutionConfig(dynamic_feedback_posterior_gate_enabled=True)


def test_change_point_posterior_gate_requires_separating_likelihoods() -> None:
    with pytest.raises(ValidationError, match="alternative repeat probability"):
        EvolutionConfig(
            dynamic_feedback_trust_enabled=True,
            dynamic_feedback_posterior_gate_enabled=True,
            dynamic_feedback_change_null_repeat_probability=0.80,
            dynamic_feedback_change_alternative_repeat_probability=0.20,
        )
