import pytest
from pydantic import ValidationError

from evoshift.config import EvolutionConfig
from evoshift.evolution import FeedbackTrustModel
from evoshift.schemas import BenchmarkSample


def _dynamic_sample(
    sample_id: str,
    context: str,
    label: str,
    **hidden_metadata: object,
) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=sample_id,
        prompt=f"Policy case {context}",
        reference="hidden-oracle-not-used",
        metadata={
            "feedback_source": "shared_portal",
            "feedback_context": context,
            "feedback_reference": label,
            **hidden_metadata,
        },
    )


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


def test_change_point_mode_accepts_sparse_clean_context_with_capped_trust() -> None:
    model = FeedbackTrustModel(
        EvolutionConfig(
            feedback_default_trust=0.90,
            dynamic_feedback_trust_enabled=True,
            dynamic_feedback_trust_mode="change_point",
            dynamic_feedback_change_cold_start_trust=0.70,
        )
    )

    first = model.assess(_dynamic_sample("one", "refund:late", "DENY"))
    second = model.assess(_dynamic_sample("two", "exchange:opened", "ALLOW"))

    assert first.reason == "change_point_initial_commit"
    assert first.trust == 0.70
    assert second.trust == 0.70
    assert second.source_posterior_mean == first.source_posterior_mean
    assert second.change_probability == 0.0


def test_change_point_mode_quarantines_isolated_contradiction() -> None:
    model = FeedbackTrustModel(
        EvolutionConfig(
            feedback_default_trust=0.90,
            dynamic_feedback_trust_enabled=True,
            dynamic_feedback_trust_mode="change_point",
        )
    )
    model.assess(_dynamic_sample("one", "refund:late", "DENY"))

    contradiction = model.assess(_dynamic_sample("two", "refund:late", "ALLOW"))

    assert contradiction.reason == "change_point_pending_change"
    assert contradiction.trust == 0.10
    assert 0.0 < contradiction.change_probability < 0.80
    assert contradiction.source_regime == 0
    assert model.snapshot()["confirmed_source_changes"] == 0


def test_change_point_mode_recovery_lowers_posterior_and_penalizes_noise() -> None:
    model = FeedbackTrustModel(
        EvolutionConfig(
            feedback_default_trust=0.90,
            dynamic_feedback_trust_enabled=True,
            dynamic_feedback_trust_mode="change_point",
        )
    )
    model.assess(_dynamic_sample("one", "refund:late", "DENY"))
    contradiction = model.assess(_dynamic_sample("two", "refund:late", "ALLOW"))

    recovered = model.assess(_dynamic_sample("three", "refund:late", "DENY"))

    assert recovered.reason == "change_point_consistent"
    assert recovered.change_probability < contradiction.change_probability
    assert recovered.pending_observations == 0
    assert recovered.source_posterior_mean < contradiction.source_posterior_mean


def test_change_point_mode_shares_change_evidence_across_contexts() -> None:
    model = FeedbackTrustModel(
        EvolutionConfig(
            feedback_default_trust=0.90,
            dynamic_feedback_trust_enabled=True,
            dynamic_feedback_trust_mode="change_point",
        )
    )
    model.assess(_dynamic_sample("one", "refund:late", "DENY"))
    model.assess(_dynamic_sample("two", "exchange:opened", "DENY"))

    first = model.assess(_dynamic_sample("three", "refund:late", "ALLOW"))
    confirmed = model.assess(_dynamic_sample("four", "exchange:opened", "ALLOW"))

    assert first.reason == "change_point_pending_change"
    assert confirmed.reason == "change_point_confirmed_source_change"
    assert confirmed.trust > 0.60
    assert confirmed.source_regime == 1
    assert confirmed.grace_observations == 8
    snapshot = model.snapshot()
    assert snapshot["confirmed_context_changes"] == 2
    assert snapshot["confirmed_source_changes"] == 1


def test_change_point_mode_transfers_confirmed_regime_with_bounded_grace() -> None:
    model = FeedbackTrustModel(
        EvolutionConfig(
            feedback_default_trust=0.90,
            dynamic_feedback_trust_enabled=True,
            dynamic_feedback_trust_mode="change_point",
            dynamic_feedback_change_grace_observations=2,
        )
    )
    for index, context in enumerate(("refund:late", "exchange:opened", "warranty:used")):
        model.assess(_dynamic_sample(f"initial-{index}", context, "DENY"))
    model.assess(_dynamic_sample("change-one", "refund:late", "ALLOW"))
    confirmed = model.assess(_dynamic_sample("change-two", "exchange:opened", "ALLOW"))

    transferred = model.assess(_dynamic_sample("transfer", "warranty:used", "ALLOW"))
    consumed = model.assess(_dynamic_sample("stable", "refund:late", "ALLOW"))

    assert confirmed.grace_observations == 2
    assert transferred.reason == "change_point_regime_transfer"
    assert transferred.trust > 0.60
    assert transferred.grace_observations == 1
    assert consumed.grace_observations == 0
    assert model.snapshot()["confirmed_context_changes"] == 3


def test_change_point_mode_ignores_hidden_oracle_and_attack_annotations() -> None:
    config = EvolutionConfig(
        feedback_default_trust=0.90,
        dynamic_feedback_trust_enabled=True,
        dynamic_feedback_trust_mode="change_point",
    )
    clean_model = FeedbackTrustModel(config)
    adversarial_model = FeedbackTrustModel(config)
    observable_sequence = [
        ("refund:late", "DENY"),
        ("exchange:opened", "DENY"),
        ("refund:late", "ALLOW"),
        ("exchange:opened", "ALLOW"),
    ]

    clean = [
        clean_model.assess(
            _dynamic_sample(
                f"clean-{index}",
                context,
                label,
                feedback_kind="clean",
                feedback_corrupted=False,
                oracle_policy_version="v1",
            )
        )
        for index, (context, label) in enumerate(observable_sequence)
    ]
    adversarial = [
        adversarial_model.assess(
            _dynamic_sample(
                f"attack-{index}",
                context,
                label,
                feedback_kind="attack",
                feedback_corrupted=True,
                oracle_policy_version="v9",
                attack_goal="poison",
            )
        )
        for index, (context, label) in enumerate(observable_sequence)
    ]

    assert [item.trust for item in clean] == [item.trust for item in adversarial]
    assert [item.reason for item in clean] == [item.reason for item in adversarial]
    assert [item.change_probability for item in clean] == [
        item.change_probability for item in adversarial
    ]
    assert [item.source_regime for item in clean] == [item.source_regime for item in adversarial]


def test_change_point_config_rejects_invalid_mode_and_reliability_interval() -> None:
    with pytest.raises(ValidationError, match="unsupported dynamic_feedback_trust_mode"):
        EvolutionConfig(dynamic_feedback_trust_mode="oracle")
    with pytest.raises(ValidationError, match="reliability floor must be below"):
        EvolutionConfig(
            dynamic_feedback_reliability_floor=0.90,
            dynamic_feedback_reliability_ceiling=0.80,
        )
