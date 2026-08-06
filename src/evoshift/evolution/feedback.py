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
    adaptation_trust: float = 0.0
    change_point_probability: float = 0.0
    change_point_crossed: bool = False
    change_point_run_length: int = 0


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
    change_probability: float = 0.0
    change_run_length: int = 0


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
        self.change_point_enabled = config.dynamic_feedback_change_point_enabled
        self.change_hazard = config.dynamic_feedback_change_hazard
        self.change_epsilon_0 = config.dynamic_feedback_change_epsilon_0
        self.change_epsilon_1 = config.dynamic_feedback_change_epsilon_1
        self.change_threshold = config.dynamic_feedback_change_threshold
        self.change_soft_trust = config.dynamic_feedback_change_soft_trust
        self.change_max_run_length = config.dynamic_feedback_change_max_run_length
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
        self.posterior_crossings = 0
        self.posterior_resets = 0
        self.soft_change_observations = 0

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
                adaptation_trust=prior,
            )

        context = observable_context
        signal = self._canonical_signal(sample.metadata["feedback_reference"])
        if self.change_min_span > 0 and episode_index is None:
            raise ValueError(
                "episode_index is required when dynamic_feedback_change_min_span is enabled"
            )
        source_state = self._sources.setdefault(source, self._new_source_state(prior))
        context_state = self._get_context_state(source, context)
        trust, adaptation_trust, reason, change_probability, crossed = self._observe(
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
            adaptation_trust=max(0.0, min(1.0, adaptation_trust)),
            change_point_probability=change_probability,
            change_point_crossed=crossed,
            change_point_run_length=context_state.change_run_length,
        )

    def observable_key(self, sample: BenchmarkSample) -> tuple[str, str]:
        """Return the learner-visible provenance/context key without mutating trust state."""

        source = str(sample.metadata.get("feedback_source", "")).strip() or "unspecified"
        return source, self._context(sample)

    def snapshot(self) -> dict[str, Any]:
        return {
            "dynamic_enabled": self.dynamic_enabled,
            "change_point_enabled": self.change_point_enabled,
            "change_hazard": self.change_hazard,
            "change_epsilon_0": self.change_epsilon_0,
            "change_epsilon_1": self.change_epsilon_1,
            "change_threshold": self.change_threshold,
            "change_soft_trust": self.change_soft_trust,
            "change_max_run_length": self.change_max_run_length,
            "change_min_span": self.change_min_span,
            "tracked_sources": len(self._sources),
            "tracked_contexts": len(self._contexts),
            "confirmed_context_changes": self.confirmed_changes,
            "temporally_deferred_changes": self.temporally_deferred_changes,
            "low_trust_observations": self.low_trust_observations,
            "posterior_crossings": self.posterior_crossings,
            "posterior_resets": self.posterior_resets,
            "soft_change_observations": self.soft_change_observations,
            "source_posteriors": {
                source: {
                    "alpha": state.alpha,
                    "beta": state.beta,
                    "mean": state.mean,
                }
                for source, state in sorted(self._sources.items())
            },
            "change_point_contexts": {
                f"{source}\u001f{context}": {
                    "committed_signal": state.committed_signal,
                    "pending_signal": state.pending_signal,
                    "pending_observations": state.pending_observations,
                    "change_probability": state.change_probability,
                    "change_run_length": state.change_run_length,
                    "total_observations": state.total_observations,
                }
                for (source, context), state in sorted(self._contexts.items())
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
    ) -> tuple[float, float, str, float, bool]:
        context.total_observations += 1
        if not context.committed_signal:
            self._advance_pending(context, signal, episode_index=episode_index)
            if context.pending_observations >= self.min_consistent:
                context.committed_signal = signal
                context.committed_observations = context.pending_observations
                source.alpha += context.pending_observations
                self._clear_pending(context)
                return source.mean, source.mean, "dynamic_initial_consensus", 0.0, False
            cold = min(source.mean, self.cold_start_trust)
            return cold, cold, "dynamic_cold_start", 0.0, False

        if signal == context.committed_signal:
            if context.pending_observations:
                source.beta += context.pending_observations
            self._clear_pending(context)
            change_probability, _crossed, _reset = self._update_change_posterior(
                context,
                signal,
            )
            context.committed_observations += 1
            source.alpha += 1.0
            return source.mean, source.mean, "dynamic_consistent", change_probability, False

        pending_matches = not context.pending_signal or context.pending_signal == signal
        self._advance_pending(context, signal, episode_index=episode_index)
        change_probability, crossed, _reset = self._update_change_posterior(
            context,
            signal,
            pending_matches=pending_matches,
        )
        if context.pending_observations >= self.min_consistent:
            first_index = context.pending_first_index
            change_span = (
                episode_index - first_index
                if episode_index is not None and first_index is not None
                else 0
            )
            if change_span < self.change_min_span:
                self.temporally_deferred_changes += 1
                return (
                    min(source.mean, self.conflict_trust),
                    self._adaptation_trust(source.mean, change_probability, crossed),
                    "dynamic_pending_change_span",
                    change_probability,
                    crossed,
                )
            context.committed_signal = signal
            context.committed_observations = context.pending_observations
            source.alpha += context.pending_observations
            self._clear_pending(context)
            self.confirmed_changes += 1
            context.change_probability = 0.0
            context.change_run_length = 0
            return source.mean, source.mean, "dynamic_confirmed_change", 0.0, False
        return (
            min(source.mean, self.conflict_trust),
            self._adaptation_trust(source.mean, change_probability, crossed),
            "dynamic_pending_change",
            change_probability,
            crossed,
        )

    def _adaptation_trust(
        self,
        source_mean: float,
        change_probability: float,
        crossed: bool,
    ) -> float:
        if not self.change_point_enabled or not crossed:
            return min(source_mean, self.conflict_trust)
        self.soft_change_observations += 1
        return max(self.conflict_trust, min(source_mean, self.change_soft_trust))

    def _update_change_posterior(
        self,
        context: _ContextState,
        signal: str,
        *,
        pending_matches: bool = True,
    ) -> tuple[float, bool, bool]:
        """Update a bounded two-state Bayesian change-point approximation.

        ``H0`` says the committed label remains valid and ``H1`` says the
        pending label is a persistent replacement. The posterior is used only
        for the soft adaptation lane; lifecycle commits retain the existing
        repeated-evidence rule.
        """

        if not self.change_point_enabled:
            return 0.0, False, False
        if signal == context.committed_signal:
            reset = context.change_probability > 0.0
            if reset:
                self.posterior_resets += 1
            context.change_probability = 0.0
            context.change_run_length = 0
            return 0.0, False, reset
        if not pending_matches:
            # A different contradictory label is evidence that the pending
            # candidate itself was unstable; do not accumulate it into the
            # previous candidate's posterior.
            reset = context.change_probability > 0.0
            if reset:
                self.posterior_resets += 1
            context.change_probability = 0.0
            context.change_run_length = 0
            return 0.0, False, reset
        prior = context.change_probability + (1.0 - context.change_probability) * self.change_hazard
        is_contradiction = signal != context.committed_signal
        likelihood_h0 = self.change_epsilon_0 if is_contradiction else 1.0 - self.change_epsilon_0
        likelihood_h1 = (
            1.0 - self.change_epsilon_1 if pending_matches else self.change_epsilon_1
        )
        denominator = prior * likelihood_h1 + (1.0 - prior) * likelihood_h0
        posterior = (prior * likelihood_h1 / denominator) if denominator else prior
        previous = context.change_probability
        context.change_probability = max(0.0, min(1.0, posterior))
        context.change_run_length = min(
            self.change_max_run_length,
            context.change_run_length + 1,
        )
        crossed = (
            is_contradiction
            and context.change_probability >= self.change_threshold
            and previous < self.change_threshold
        )
        if crossed:
            self.posterior_crossings += 1
        return context.change_probability, crossed, False

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
