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


def test_dynamic_feedback_change_requires_temporally_diverse_recurrence() -> None:
    model = FeedbackTrustModel(
        EvolutionConfig(
            feedback_default_trust=0.90,
            dynamic_feedback_trust_enabled=True,
            dynamic_feedback_min_consistent_observations=2,
            dynamic_feedback_change_min_span=2,
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
            },
        )

    first = model.assess(sample("one", "DENY"), episode_index=10)
    consensus = model.assess(sample("two", "DENY"), episode_index=11)
    pending = model.assess(sample("three", "APPROVE"), episode_index=84)
    adjacent = model.assess(sample("four", "APPROVE"), episode_index=85)
    confirmed = model.assess(sample("five", "APPROVE"), episode_index=86)

    assert first.reason == "dynamic_cold_start"
    assert consensus.reason == "dynamic_initial_consensus"
    assert pending.reason == "dynamic_pending_change"
    assert adjacent.reason == "dynamic_pending_change_span"
    assert adjacent.trust == 0.10
    assert confirmed.reason == "dynamic_confirmed_change"
    assert confirmed.trust > 0.60
    assert model.snapshot()["temporally_deferred_changes"] == 1
    assert model.snapshot()["confirmed_context_changes"] == 1


def test_dynamic_feedback_change_span_resets_after_pending_evidence_is_cancelled() -> None:
    model = FeedbackTrustModel(
        EvolutionConfig(
            dynamic_feedback_trust_enabled=True,
            dynamic_feedback_change_min_span=2,
        )
    )

    def sample(sample_id: str, label: str) -> BenchmarkSample:
        return BenchmarkSample(
            sample_id=sample_id,
            prompt="case",
            reference="hidden",
            metadata={
                "feedback_source": "portal",
                "feedback_context": "case",
                "feedback_reference": label,
            },
        )

    model.assess(sample("one", "DENY"), episode_index=0)
    model.assess(sample("two", "DENY"), episode_index=1)
    model.assess(sample("three", "APPROVE"), episode_index=10)
    deferred = model.assess(sample("four", "APPROVE"), episode_index=11)
    recovered = model.assess(sample("five", "DENY"), episode_index=12)
    model.assess(sample("six", "APPROVE"), episode_index=20)
    deferred_again = model.assess(sample("seven", "APPROVE"), episode_index=21)
    confirmed = model.assess(sample("eight", "APPROVE"), episode_index=22)

    assert deferred.reason == "dynamic_pending_change_span"
    assert recovered.reason == "dynamic_consistent"
    assert deferred_again.reason == "dynamic_pending_change_span"
    assert confirmed.reason == "dynamic_confirmed_change"
    assert model.snapshot()["temporally_deferred_changes"] == 2


def test_dynamic_feedback_change_span_requires_global_episode_index() -> None:
    model = FeedbackTrustModel(
        EvolutionConfig(
            dynamic_feedback_trust_enabled=True,
            dynamic_feedback_change_min_span=2,
        )
    )
    sample = BenchmarkSample(
        sample_id="sample",
        prompt="case",
        reference="hidden",
        metadata={
            "feedback_source": "portal",
            "feedback_context": "case",
            "feedback_reference": "DENY",
        },
    )

    with pytest.raises(ValueError, match="episode_index is required"):
        model.assess(sample)


def test_change_point_posterior_soft_crossing_is_non_destructive() -> None:
    model = FeedbackTrustModel(
        EvolutionConfig(
            feedback_default_trust=0.90,
            dynamic_feedback_trust_enabled=True,
            dynamic_feedback_change_point_enabled=True,
            dynamic_feedback_change_hazard=0.15,
            dynamic_feedback_change_epsilon_0=0.10,
            dynamic_feedback_change_epsilon_1=0.10,
            dynamic_feedback_change_threshold=0.60,
            dynamic_feedback_change_soft_trust=0.65,
        )
    )

    def sample(sample_id: str, label: str) -> BenchmarkSample:
        return BenchmarkSample(
            sample_id=sample_id,
            prompt="case",
            reference="hidden",
            metadata={
                "feedback_source": "portal",
                "feedback_context": "case",
                "feedback_reference": label,
            },
        )

    model.assess(sample("one", "DENY"))
    model.assess(sample("two", "DENY"))
    soft = model.assess(sample("three", "APPROVE"))

    assert soft.reason == "dynamic_pending_change"
    assert soft.trust == 0.10
    assert soft.adaptation_trust == 0.65
    assert soft.change_point_crossed is True
    assert soft.change_point_probability > 0.60
    assert soft.pending_observations == 1

    recovered = model.assess(sample("four", "DENY"))
    assert recovered.reason == "dynamic_consistent"
    assert recovered.change_point_probability == 0.0
    assert recovered.change_point_crossed is False
    assert recovered.adaptation_trust > 0.60


def test_change_point_posterior_confirms_only_existing_repeated_evidence() -> None:
    model = FeedbackTrustModel(
        EvolutionConfig(
            feedback_default_trust=0.90,
            dynamic_feedback_trust_enabled=True,
            dynamic_feedback_change_point_enabled=True,
            dynamic_feedback_change_hazard=0.15,
        )
    )

    def sample(sample_id: str, label: str) -> BenchmarkSample:
        return BenchmarkSample(
            sample_id=sample_id,
            prompt="case",
            reference="hidden",
            metadata={
                "feedback_source": "portal",
                "feedback_context": "case",
                "feedback_reference": label,
            },
        )

    model.assess(sample("one", "DENY"))
    model.assess(sample("two", "DENY"))
    first = model.assess(sample("three", "APPROVE"))
    second = model.assess(sample("four", "APPROVE"))

    assert first.change_point_crossed is True
    assert first.trust == 0.10
    assert second.reason == "dynamic_confirmed_change"
    assert second.change_point_probability == 0.0
    assert second.trust > 0.60
    assert model.snapshot()["posterior_crossings"] == 1


def test_change_point_posterior_does_not_accumulate_alternating_candidates() -> None:
    model = FeedbackTrustModel(
        EvolutionConfig(
            feedback_default_trust=0.90,
            dynamic_feedback_trust_enabled=True,
            dynamic_feedback_change_point_enabled=True,
            dynamic_feedback_change_hazard=0.15,
        )
    )

    def sample(sample_id: str, label: str) -> BenchmarkSample:
        return BenchmarkSample(
            sample_id=sample_id,
            prompt="case",
            reference="hidden",
            metadata={
                "feedback_source": "portal",
                "feedback_context": "case",
                "feedback_reference": label,
            },
        )

    model.assess(sample("one", "DENY"))
    model.assess(sample("two", "DENY"))
    first = model.assess(sample("three", "APPROVE"))
    alternating = model.assess(sample("four", "REVIEW"))
    recovered = model.assess(sample("five", "DENY"))

    assert first.change_point_crossed is True
    assert alternating.change_point_probability < first.change_point_probability
    assert alternating.change_point_crossed is False
    assert recovered.change_point_probability == 0.0
    assert recovered.change_point_crossed is False


def test_change_point_posterior_requires_dynamic_feedback_trust() -> None:
    with pytest.raises(ValidationError, match="requires dynamic feedback trust"):
        EvolutionConfig(dynamic_feedback_change_point_enabled=True)


def test_change_point_posterior_ignores_oracle_only_metadata() -> None:
    config = EvolutionConfig(
        feedback_default_trust=0.90,
        dynamic_feedback_trust_enabled=True,
        dynamic_feedback_change_point_enabled=True,
        dynamic_feedback_change_hazard=0.15,
    )
    control = FeedbackTrustModel(config)
    perturbed = FeedbackTrustModel(config)

    for index, label in enumerate(("DENY", "DENY", "APPROVE", "DENY")):
        observable = {
            "feedback_source": "portal",
            "feedback_context": "case",
            "feedback_reference": label,
        }
        control_sample = BenchmarkSample(
            sample_id=f"control-{index}",
            prompt="case",
            reference="hidden-control",
            phase="control-phase",
            metadata={
                **observable,
                "feedback_kind": "clean",
                "feedback_corrupted": False,
            },
        )
        perturbed_sample = BenchmarkSample(
            sample_id=f"perturbed-{index}",
            prompt="case",
            reference="hidden-perturbed",
            phase="perturbed-phase",
            metadata={
                **observable,
                "feedback_kind": "attack",
                "feedback_corrupted": True,
                "feedback_attack_goal": "premature_update",
                "policy_version": "oracle-only",
                "valid_memory_tags": ["hidden-valid"],
                "stale_memory_tags": ["hidden-stale"],
            },
        )

        control_assessment = control.assess(control_sample, episode_index=index)
        perturbed_assessment = perturbed.assess(perturbed_sample, episode_index=index)
        assert control_assessment == perturbed_assessment

    assert control.snapshot() == perturbed.snapshot()


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


def test_observable_key_is_normalized_and_does_not_consume_feedback() -> None:
    model = FeedbackTrustModel(EvolutionConfig(dynamic_feedback_trust_enabled=True))
    sample = BenchmarkSample(
        sample_id="sample",
        prompt="  Refund   Prompt ",
        reference="hidden",
        metadata={
            "feedback_source": " portal ",
            "feedback_context": " Refund:Premium:Days_15_30 ",
            "feedback_reference": "APPROVE",
        },
    )

    before = model.snapshot()
    key = model.observable_key(sample)
    after = model.snapshot()

    assert key == ("portal", "refund:premium:days_15_30")
    assert before == after


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
