import pytest
from pydantic import ValidationError

from evoshift.config import EvolutionConfig, EvoShiftConfig
from evoshift.evolution import FeedbackTrustModel
from evoshift.schemas import Algorithm, BenchmarkSample


def test_feedback_trust_uses_only_observable_source_provenance() -> None:
    model = FeedbackTrustModel(
        EvolutionConfig(
            feedback_default_trust=0.5,
            feedback_source_trust={
                "verified": 0.95,
                "untrusted": 0.10,
            },
        )
    )
    clean = BenchmarkSample(
        sample_id="clean",
        prompt="task",
        reference="hidden-a",
        metadata={
            "feedback_source": "verified",
            "feedback_kind": "clean",
            "feedback_corrupted": False,
        },
    )
    corrupted = clean.model_copy(
        update={
            "sample_id": "corrupted",
            "reference": "hidden-b",
            "metadata": {
                "feedback_source": "verified",
                "feedback_kind": "attack",
                "feedback_corrupted": True,
            },
        }
    )

    assert model.assess(clean).trust == 0.95
    assert model.assess(corrupted).trust == 0.95
    assert (
        model.assess(clean.model_copy(update={"metadata": {"feedback_source": "untrusted"}})).trust
        == 0.10
    )


def test_feedback_trust_falls_back_for_unconfigured_sources() -> None:
    model = FeedbackTrustModel(EvolutionConfig(feedback_default_trust=0.42))
    sample = BenchmarkSample(
        sample_id="sample",
        prompt="task",
        reference="answer",
    )

    assessment = model.assess(sample)

    assert assessment.source == "unspecified"
    assert assessment.trust == 0.42
    assert assessment.reason == "default_source:unspecified"


def test_dynamic_feedback_trust_quarantines_isolated_conflict_and_accepts_persistence() -> None:
    model = FeedbackTrustModel(
        EvolutionConfig(
            feedback_default_trust=0.90,
            dynamic_feedback_trust_enabled=True,
            dynamic_feedback_min_consistent_observations=2,
            dynamic_feedback_cold_start_trust=0.40,
            dynamic_feedback_conflict_trust=0.10,
        )
    )

    def sample(sample_id: str, label: str) -> BenchmarkSample:
        return BenchmarkSample(
            sample_id=sample_id,
            prompt="Refund case: days 8 to 14",
            reference="hidden-oracle-not-used",
            metadata={
                "feedback_source": "shared_portal",
                "feedback_context": "refund:any:days_8_14",
                "feedback_reference": label,
                "feedback_kind": "hidden-test-annotation",
                "feedback_corrupted": label == "APPROVE",
            },
        )

    first = model.assess(sample("one", "DENY"))
    consensus = model.assess(sample("two", "DENY"))
    isolated = model.assess(sample("three", "APPROVE"))
    recovered = model.assess(sample("four", "DENY"))
    pending_shift = model.assess(sample("five", "APPROVE"))
    confirmed_shift = model.assess(sample("six", "APPROVE"))

    assert first.reason == "dynamic_cold_start"
    assert first.trust == 0.40
    assert consensus.reason == "dynamic_initial_consensus"
    assert consensus.trust > 0.60
    assert isolated.reason == "dynamic_pending_change"
    assert isolated.trust == 0.10
    assert recovered.reason == "dynamic_consistent"
    assert recovered.trust > 0.60
    assert pending_shift.reason == "dynamic_pending_change"
    assert confirmed_shift.reason == "dynamic_confirmed_change"
    assert confirmed_shift.trust > 0.60
    assert model.snapshot()["confirmed_context_changes"] == 1


def test_dynamic_feedback_trust_bounds_context_state() -> None:
    model = FeedbackTrustModel(
        EvolutionConfig(
            dynamic_feedback_trust_enabled=True,
            dynamic_feedback_max_contexts=2,
        )
    )
    for index in range(3):
        model.assess(
            BenchmarkSample(
                sample_id=str(index),
                prompt=f"task {index}",
                reference="hidden",
                metadata={
                    "feedback_source": "source",
                    "feedback_context": f"context-{index}",
                    "feedback_reference": "OK",
                },
            )
        )

    assert model.snapshot()["tracked_contexts"] == 2


def test_dynamic_feedback_trust_can_be_disabled_by_algorithm_control() -> None:
    model = FeedbackTrustModel(
        EvolutionConfig(
            feedback_default_trust=0.8,
            dynamic_feedback_trust_enabled=True,
        ),
        dynamic_enabled=False,
    )
    sample = BenchmarkSample(
        sample_id="sample",
        prompt="task",
        reference="hidden",
        metadata={
            "feedback_source": "source",
            "feedback_context": "context",
            "feedback_reference": "ALLOW",
        },
    )

    assessment = model.assess(sample)

    assert assessment.trust == 0.8
    assert assessment.reason == "default_source:source"
    assert model.snapshot()["dynamic_enabled"] is False


def test_shadow_candidate_config_enforces_verified_admission_invariants() -> None:
    with pytest.raises(ValidationError, match="requires future audit"):
        EvolutionConfig(shadow_candidate_enabled=True)
    with pytest.raises(ValidationError, match="requires paired replay"):
        EvolutionConfig(
            shadow_candidate_enabled=True,
            future_audit_enabled=True,
            paired_replay=False,
        )
    with pytest.raises(ValidationError, match="threshold must not exceed"):
        EvolutionConfig(
            shadow_candidate_enabled=True,
            future_audit_enabled=True,
            min_feedback_trust_for_shadow_candidate=0.80,
            min_feedback_trust_for_candidate=0.60,
        )
    with pytest.raises(ValidationError, match="replay trust"):
        EvolutionConfig(
            shadow_candidate_enabled=True,
            future_audit_enabled=True,
            min_feedback_trust_for_replay=0.50,
        )
    with pytest.raises(ValidationError, match="requires shadow candidate"):
        EvolutionConfig(shadow_eprocess_enabled=True)
    with pytest.raises(ValidationError, match="must exceed"):
        EvolutionConfig(
            shadow_candidate_enabled=True,
            future_audit_enabled=True,
            shadow_eprocess_enabled=True,
            shadow_eprocess_null_match_probability=0.75,
            shadow_eprocess_alternative_match_probability=0.25,
        )
    with pytest.raises(ValidationError, match="supported only by evoshift"):
        EvoShiftConfig(
            algorithm=Algorithm.REFLEXION,
            evolution=EvolutionConfig(
                shadow_candidate_enabled=True,
                future_audit_enabled=True,
            ),
        )
