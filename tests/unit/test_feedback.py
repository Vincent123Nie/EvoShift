from evoshift.config import EvolutionConfig
from evoshift.evolution import FeedbackTrustModel
from evoshift.schemas import BenchmarkSample


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
