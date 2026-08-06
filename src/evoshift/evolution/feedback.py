from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from evoshift.config import EvolutionConfig
from evoshift.schemas import BenchmarkSample


@dataclass(frozen=True)
class FeedbackAssessment:
    source: str
    context: str
    signal: str
    trust: float
    reason: str
    source_posterior_mean: float
    context_observations: int
    pending_observations: int
    candidate_evidence_mature: bool


@dataclass
class _SourceState:
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


@dataclass
class _ContextState:
    committed_signal: str = ""
    committed_observations: int = 0
    pending_signal: str = ""
    pending_observations: int = 0
    pending_first_index: int | None = None
    total_observations: int = 0


class FeedbackTrustModel:
    """Assess observable feedback provenance and temporal consistency.

    Static mode preserves the configured source-prior behavior. Dynamic mode
    additionally reads only an observable context key and learner-visible
    ``feedback_reference``. Hidden oracle correctness, corruption annotations,
    attack labels, and benchmark phase are never consulted.

    A per-context label is committed after repeated evidence. An isolated
    contradiction is quarantined as a pending change; a repeated contradiction
    becomes the new committed label. Source reliability is represented by a
    Beta posterior and is updated only when a context resolves whether pending
    evidence was consistent or transient.
    """

    def __init__(
        self,
        config: EvolutionConfig,
        *,
        dynamic_enabled: bool | None = None,
    ):
        self.default_trust = config.feedback_default_trust
        self.source_trust = {
            source.strip(): float(trust) for source, trust in config.feedback_source_trust.items()
        }
        self.dynamic_enabled = (
            config.dynamic_feedback_trust_enabled if dynamic_enabled is None else dynamic_enabled
        )
        self.context_field = config.dynamic_feedback_context_field.strip()
        if self.context_field != "feedback_context":
            raise ValueError(
                "feedback trust requires the typed learner-visible feedback_context field"
            )
        self.min_consistent = config.dynamic_feedback_min_consistent_observations
        self.change_min_span = config.dynamic_feedback_change_min_span
        self.cold_start_trust = config.dynamic_feedback_cold_start_trust
        self.conflict_trust = config.dynamic_feedback_conflict_trust
        self.prior_strength = config.dynamic_feedback_prior_strength
        self.max_contexts = config.dynamic_feedback_max_contexts
        self._sources: dict[str, _SourceState] = {}
        self._contexts: OrderedDict[tuple[str, str], _ContextState] = OrderedDict()
        self.confirmed_changes = 0
        self.temporally_deferred_changes = 0
        self.low_trust_observations = 0

    def assess(
        self,
        sample: BenchmarkSample,
        *,
        episode_index: int | None = None,
    ) -> FeedbackAssessment:
        source, observable_context = self.observable_key(sample)
        prior = self.source_trust.get(source, self.default_trust)
        if not self.dynamic_enabled or "feedback_reference" not in sample.metadata:
            reason_prefix = "configured_source" if source in self.source_trust else "default_source"
            return FeedbackAssessment(
                source=source,
                context="",
                signal="",
                trust=prior,
                reason=f"{reason_prefix}:{source}",
                source_posterior_mean=prior,
                context_observations=0,
                pending_observations=0,
                candidate_evidence_mature=True,
            )

        context = observable_context
        signal = self._canonical_signal(sample.metadata["feedback_reference"])
        if self.change_min_span > 0 and episode_index is None:
            raise ValueError(
                "episode_index is required when dynamic_feedback_change_min_span is enabled"
            )
        source_state = self._sources.setdefault(source, self._new_source_state(prior))
        context_state = self._get_context_state(source, context)
        trust, reason = self._observe(
            source_state,
            context_state,
            signal,
            episode_index=episode_index,
        )
        if trust < prior:
            self.low_trust_observations += 1
        return FeedbackAssessment(
            source=source,
            context=context,
            signal=signal,
            trust=max(0.0, min(1.0, trust)),
            reason=reason,
            source_posterior_mean=source_state.mean,
            context_observations=context_state.total_observations,
            pending_observations=context_state.pending_observations,
            candidate_evidence_mature=self._candidate_evidence_mature(reason),
        )

    @staticmethod
    def _candidate_evidence_mature(reason: str) -> bool:
        """Return whether the observation follows a committed context regime."""

        return reason in {"dynamic_consistent", "dynamic_confirmed_change"}

    def observable_key(self, sample: BenchmarkSample) -> tuple[str, str]:
        """Return the learner-visible provenance/context key without mutating trust state."""

        source = str(sample.metadata.get("feedback_source", "")).strip() or "unspecified"
        return source, self._context(sample)

    def snapshot(self) -> dict[str, Any]:
        return {
            "dynamic_enabled": self.dynamic_enabled,
            "change_min_span": self.change_min_span,
            "tracked_sources": len(self._sources),
            "tracked_contexts": len(self._contexts),
            "confirmed_context_changes": self.confirmed_changes,
            "temporally_deferred_changes": self.temporally_deferred_changes,
            "low_trust_observations": self.low_trust_observations,
            "source_posteriors": {
                source: {
                    "alpha": state.alpha,
                    "beta": state.beta,
                    "mean": state.mean,
                }
                for source, state in sorted(self._sources.items())
            },
        }

    def _new_source_state(self, prior: float) -> _SourceState:
        return _SourceState(
            alpha=1.0 + prior * self.prior_strength,
            beta=1.0 + (1.0 - prior) * self.prior_strength,
        )

    def _context(self, sample: BenchmarkSample) -> str:
        configured = str(sample.metadata.get(self.context_field, "")).strip()
        if configured:
            return configured.casefold()
        return " ".join(sample.prompt.casefold().split())

    def _get_context_state(self, source: str, context: str) -> _ContextState:
        key = (source, context)
        existing = self._contexts.pop(key, None)
        state = existing or _ContextState()
        self._contexts[key] = state
        while len(self._contexts) > self.max_contexts:
            self._contexts.popitem(last=False)
        return state

    def _observe(
        self,
        source: _SourceState,
        context: _ContextState,
        signal: str,
        *,
        episode_index: int | None,
    ) -> tuple[float, str]:
        context.total_observations += 1
        if not context.committed_signal:
            self._advance_pending(context, signal, episode_index=episode_index)
            if context.pending_observations >= self.min_consistent:
                context.committed_signal = signal
                context.committed_observations = context.pending_observations
                source.alpha += context.pending_observations
                self._clear_pending(context)
                return source.mean, "dynamic_initial_consensus"
            return min(source.mean, self.cold_start_trust), "dynamic_cold_start"

        if signal == context.committed_signal:
            if context.pending_observations:
                source.beta += context.pending_observations
            self._clear_pending(context)
            context.committed_observations += 1
            source.alpha += 1.0
            return source.mean, "dynamic_consistent"

        self._advance_pending(context, signal, episode_index=episode_index)
        if context.pending_observations >= self.min_consistent:
            first_index = context.pending_first_index
            change_span = (
                episode_index - first_index
                if episode_index is not None and first_index is not None
                else 0
            )
            if change_span < self.change_min_span:
                self.temporally_deferred_changes += 1
                return min(source.mean, self.conflict_trust), "dynamic_pending_change_span"
            context.committed_signal = signal
            context.committed_observations = context.pending_observations
            source.alpha += context.pending_observations
            self._clear_pending(context)
            self.confirmed_changes += 1
            return source.mean, "dynamic_confirmed_change"
        return min(source.mean, self.conflict_trust), "dynamic_pending_change"

    @staticmethod
    def _advance_pending(
        context: _ContextState,
        signal: str,
        *,
        episode_index: int | None,
    ) -> None:
        if context.pending_signal == signal:
            context.pending_observations += 1
        else:
            context.pending_signal = signal
            context.pending_observations = 1
            context.pending_first_index = episode_index

    @staticmethod
    def _clear_pending(context: _ContextState) -> None:
        context.pending_signal = ""
        context.pending_observations = 0
        context.pending_first_index = None

    @staticmethod
    def _canonical_signal(value: Any) -> str:
        if isinstance(value, str):
            return " ".join(value.casefold().split())
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = ["FeedbackAssessment", "FeedbackTrustModel"]
