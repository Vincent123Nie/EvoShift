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

    assert model.pre_predict(clean).trust == 0.95
    assert model.pre_predict(corrupted).trust == 0.95
    assert (
        model.pre_predict(
            clean.model_copy(update={"metadata": {"feedback_source": "untrusted"}})
        ).trust
        == 0.10
    )


def test_feedback_trust_falls_back_for_unconfigured_sources() -> None:
    model = FeedbackTrustModel(EvolutionConfig(feedback_default_trust=0.42))
    sample = BenchmarkSample(
        sample_id="sample",
        prompt="task",
        reference="answer",
    )

    assessment = model.pre_predict(sample)

    assert assessment.source == "unspecified"
    assert assessment.trust == 0.42
    assert assessment.reason == "pre_predict_default_source:unspecified"


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

    first = model.observe_feedback(sample("one", "DENY"))
    consensus = model.observe_feedback(sample("two", "DENY"))
    isolated = model.observe_feedback(sample("three", "APPROVE"))
    recovered = model.observe_feedback(sample("four", "DENY"))
    pending_shift = model.observe_feedback(sample("five", "APPROVE"))
    confirmed_shift = model.observe_feedback(sample("six", "APPROVE"))

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

    first = model.observe_feedback(sample("one", "DENY"), episode_index=10)
    consensus = model.observe_feedback(sample("two", "DENY"), episode_index=11)
    pending = model.observe_feedback(sample("three", "APPROVE"), episode_index=84)
    adjacent = model.observe_feedback(sample("four", "APPROVE"), episode_index=85)
    confirmed = model.observe_feedback(sample("five", "APPROVE"), episode_index=86)

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

    model.observe_feedback(sample("one", "DENY"), episode_index=0)
    model.observe_feedback(sample("two", "DENY"), episode_index=1)
    model.observe_feedback(sample("three", "APPROVE"), episode_index=10)
    deferred = model.observe_feedback(sample("four", "APPROVE"), episode_index=11)
    recovered = model.observe_feedback(sample("five", "DENY"), episode_index=12)
    model.observe_feedback(sample("six", "APPROVE"), episode_index=20)
    deferred_again = model.observe_feedback(sample("seven", "APPROVE"), episode_index=21)
    confirmed = model.observe_feedback(sample("eight", "APPROVE"), episode_index=22)

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
        model.observe_feedback(sample)


def test_dynamic_feedback_trust_bounds_context_state() -> None:
    model = FeedbackTrustModel(
        EvolutionConfig(
            dynamic_feedback_trust_enabled=True,
            dynamic_feedback_max_contexts=2,
        )
    )
    for index in range(3):
        model.observe_feedback(
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


def test_pre_predict_assessment_has_a_current_feedback_temporal_firewall() -> None:
    model = FeedbackTrustModel(
        EvolutionConfig(
            feedback_default_trust=0.90,
            dynamic_feedback_trust_enabled=True,
            dynamic_feedback_min_consistent_observations=2,
            dynamic_feedback_cold_start_trust=0.40,
            dynamic_feedback_conflict_trust=0.10,
        )
    )

    def sample(
        label: str,
        *,
        reference: str = "hidden-oracle-not-used",
        kind: str = "clean",
        corrupted: bool = False,
    ) -> BenchmarkSample:
        return BenchmarkSample(
            sample_id="same-current-request",
            prompt="Refund case: days 8 to 14",
            reference=reference,
            metadata={
                "feedback_source": "shared_portal",
                "feedback_context": "refund:any:days_8_14",
                "feedback_reference": label,
                "feedback_kind": kind,
                "feedback_corrupted": corrupted,
            },
        )

    before = model.snapshot()
    cold_deny = model.pre_predict(sample("DENY"))
    cold_approve = model.pre_predict(
        sample(
            "APPROVE",
            reference="different-hidden-oracle",
            kind="attack",
            corrupted=True,
        )
    )
    assert cold_deny == cold_approve
    assert cold_deny.signal == ""
    assert model.snapshot() == before

    model.observe_feedback(sample("DENY"), episode_index=0)
    model.observe_feedback(sample("DENY"), episode_index=1)
    stable_before = model.snapshot()
    stable_deny = model.pre_predict(sample("DENY"))
    stable_approve = model.pre_predict(
        sample(
            "APPROVE",
            reference="different-hidden-oracle",
            kind="attack",
            corrupted=True,
        )
    )
    assert stable_deny == stable_approve
    assert stable_deny.signal == "deny"
    assert stable_deny.reason == "dynamic_pre_predict_consistent"
    assert model.snapshot() == stable_before

    model.observe_feedback(sample("APPROVE"), episode_index=2)
    pending_before = model.snapshot()
    pending_deny = model.pre_predict(sample("DENY"))
    pending_approve = model.pre_predict(
        sample(
            "APPROVE",
            reference="different-hidden-oracle",
            kind="attack",
            corrupted=True,
        )
    )
    assert pending_deny == pending_approve
    assert pending_deny.signal == "approve"
    assert pending_deny.trust == 0.10
    assert pending_deny.reason == "dynamic_pre_predict_pending_change"
    assert model.snapshot() == pending_before
    assert model.pre_feedback_assessment(sample("unread")) == pending_deny


def test_assess_remains_a_compatible_post_score_observation_alias() -> None:
    config = EvolutionConfig(dynamic_feedback_trust_enabled=True)
    explicit = FeedbackTrustModel(config)
    legacy = FeedbackTrustModel(config)
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

    assert legacy.assess(sample) == explicit.observe_feedback(sample)
    assert legacy.snapshot() == explicit.snapshot()


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

    assessment = model.pre_predict(sample)

    assert assessment.trust == 0.8
    assert assessment.reason == "pre_predict_default_source:source"
    assert model.snapshot()["dynamic_enabled"] is False


def test_posterior_gate_uses_source_lower_bound_for_trusted_cold_start() -> None:
    model = FeedbackTrustModel(
        EvolutionConfig(
            feedback_default_trust=0.50,
            feedback_source_trust={"portal": 0.90},
            dynamic_feedback_trust_enabled=True,
            dynamic_feedback_posterior_gate_enabled=True,
            dynamic_feedback_cold_start_trust=0.40,
        )
    )

    trusted = BenchmarkSample(
        sample_id="trusted",
        prompt="case",
        reference="hidden",
        metadata={
            "feedback_source": "portal",
            "feedback_context": "rare-context",
            "feedback_reference": "ALLOW",
        },
    )
    unknown = trusted.model_copy(
        update={"sample_id": "unknown", "metadata": {"feedback_context": "other"}}
    )

    trusted_before = model.pre_predict(trusted)
    unknown_before = model.pre_predict(unknown)
    trusted_after = model.observe_feedback(trusted, episode_index=0)

    assert trusted_before.trust > 0.60
    assert trusted_before.reason == "dynamic_pre_predict_cold_start_posterior_lcb"
    assert trusted_after.trust > 0.60
    assert trusted_after.source_posterior_lower_bound > 0.60
    assert unknown_before.trust == 0.40
    assert unknown_before.reason == "dynamic_pre_predict_cold_start"


def test_posterior_gate_calibrates_change_confirmation_and_keeps_first_conflict_quarantined() -> (
    None
):
    model = FeedbackTrustModel(
        EvolutionConfig(
            feedback_default_trust=0.90,
            dynamic_feedback_trust_enabled=True,
            dynamic_feedback_posterior_gate_enabled=True,
            dynamic_feedback_min_consistent_observations=2,
            dynamic_feedback_change_prior_probability=0.20,
            dynamic_feedback_change_null_repeat_probability=0.20,
            dynamic_feedback_change_alternative_repeat_probability=0.80,
            dynamic_feedback_change_posterior_threshold=0.80,
        )
    )

    def sample(label: str, sample_id: str) -> BenchmarkSample:
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

    model.observe_feedback(sample("DENY", "one"), episode_index=0)
    model.observe_feedback(sample("DENY", "two"), episode_index=1)
    first_change = model.observe_feedback(sample("ALLOW", "three"), episode_index=2)
    second_change = model.observe_feedback(sample("ALLOW", "four"), episode_index=3)

    assert first_change.reason == "dynamic_pending_change"
    assert first_change.trust == 0.10
    assert first_change.change_posterior == pytest.approx(0.50)
    assert second_change.reason == "dynamic_confirmed_change_posterior"
    assert second_change.change_posterior == 0.0
    assert second_change.trust > 0.60
    assert model.snapshot()["posterior_confirmed_changes"] == 1


def test_posterior_gate_still_requires_temporal_span() -> None:
    model = FeedbackTrustModel(
        EvolutionConfig(
            feedback_default_trust=0.90,
            dynamic_feedback_trust_enabled=True,
            dynamic_feedback_posterior_gate_enabled=True,
            dynamic_feedback_min_consistent_observations=2,
            dynamic_feedback_change_min_span=2,
        )
    )

    def sample(label: str, sample_id: str) -> BenchmarkSample:
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

    model.observe_feedback(sample("DENY", "one"), episode_index=0)
    model.observe_feedback(sample("DENY", "two"), episode_index=1)
    model.observe_feedback(sample("ALLOW", "three"), episode_index=10)
    deferred = model.observe_feedback(sample("ALLOW", "four"), episode_index=11)

    assert deferred.reason == "dynamic_pending_change_span"
    assert deferred.trust == 0.10
    assert deferred.change_posterior == pytest.approx(0.80)


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
