from __future__ import annotations

from dataclasses import dataclass

from evoshift.config import EvolutionConfig
from evoshift.schemas import BenchmarkSample


@dataclass(frozen=True)
class FeedbackAssessment:
    source: str
    trust: float
    reason: str


class FeedbackTrustModel:
    """Assign observable provenance trust without consulting hidden correctness.

    The model deliberately reads only ``feedback_source``. Benchmark-only fields
    such as ``feedback_kind``, ``feedback_corrupted``, and the hidden reference
    are never inputs to the online decision.
    """

    def __init__(self, config: EvolutionConfig):
        self.default_trust = config.feedback_default_trust
        self.source_trust = {
            source.strip(): float(trust) for source, trust in config.feedback_source_trust.items()
        }

    def assess(self, sample: BenchmarkSample) -> FeedbackAssessment:
        source = str(sample.metadata.get("feedback_source", "")).strip() or "unspecified"
        if source in self.source_trust:
            return FeedbackAssessment(
                source=source,
                trust=self.source_trust[source],
                reason=f"configured_source:{source}",
            )
        return FeedbackAssessment(
            source=source,
            trust=self.default_trust,
            reason=f"default_source:{source}",
        )


__all__ = ["FeedbackAssessment", "FeedbackTrustModel"]
