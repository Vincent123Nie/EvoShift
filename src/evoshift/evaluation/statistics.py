"""Paired uncertainty estimates and verification gates for candidate evolution."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence

from evoshift.config import EvolutionConfig
from evoshift.schemas import PromotionDecision, ValidationResult


def _finite(values: Sequence[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap_ci(
    deltas: Sequence[float],
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired candidate-control deltas.

    Pairing must be performed before calling this function.  Resampling the
    delta vector preserves that pairing and prevents between-task difficulty
    from inflating uncertainty.
    """

    values = _finite(deltas, "deltas")
    if not values:
        raise ValueError("deltas must not be empty")
    if samples < 1:
        raise ValueError("samples must be at least one")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if len(values) == 1 or all(value == values[0] for value in values):
        estimate = statistics.fmean(values)
        return estimate, estimate

    generator = random.Random(seed)
    count = len(values)
    bootstrap_means = [
        statistics.fmean(values[generator.randrange(count)] for _ in range(count))
        for _ in range(samples)
    ]
    tail = (1.0 - confidence) / 2.0
    return _percentile(bootstrap_means, tail), _percentile(bootstrap_means, 1.0 - tail)


def paired_cluster_bootstrap_ci(
    deltas_by_seed: Mapping[int, Sequence[float]],
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return a paired hierarchical interval over repeated stream seeds.

    Each replicate samples seed clusters with replacement, then paired
    observations within every selected seed. The point estimate gives each
    stream seed equal weight.
    """

    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    normalized: dict[int, list[float]] = {}
    for stream_seed, deltas in deltas_by_seed.items():
        values = _finite(deltas, f"seed {stream_seed} deltas")
        if not values:
            raise ValueError(f"seed {stream_seed} has no paired deltas")
        normalized[int(stream_seed)] = values
    if not normalized:
        raise ValueError("deltas_by_seed must not be empty")

    ordered_seeds = sorted(normalized)
    point = statistics.fmean(statistics.fmean(normalized[item]) for item in ordered_seeds)
    generator = random.Random(seed)
    bootstrap_means: list[float] = []
    for _ in range(samples):
        selected_seeds = [generator.choice(ordered_seeds) for _ in ordered_seeds]
        cluster_means: list[float] = []
        for selected_seed in selected_seeds:
            cluster = normalized[selected_seed]
            resampled = [generator.choice(cluster) for _ in cluster]
            cluster_means.append(statistics.fmean(resampled))
        bootstrap_means.append(statistics.fmean(cluster_means))
    tail = (1.0 - confidence) / 2.0
    return (
        point,
        _percentile(bootstrap_means, tail),
        _percentile(bootstrap_means, 1.0 - tail),
    )


def _cost_delta_ratio(control_costs: Sequence[float], candidate_costs: Sequence[float]) -> float:
    control_mean = statistics.fmean(control_costs)
    candidate_mean = statistics.fmean(candidate_costs)
    if control_mean <= 0.0:
        return 0.0 if candidate_mean <= 0.0 else math.inf
    return (candidate_mean - control_mean) / control_mean


class PromotionGate:
    """Apply VERA's paired gain, safety-regression, and cost constraints."""

    def __init__(self, config: EvolutionConfig) -> None:
        self.config = config

    def decide(
        self,
        candidate_id: str,
        candidate_type: str,
        control_scores: Sequence[float],
        candidate_scores: Sequence[float],
        control_costs: Sequence[float] | None = None,
        candidate_costs: Sequence[float] | None = None,
        protected_mask: Sequence[bool] | None = None,
    ) -> PromotionDecision:
        """Return a typed, auditable promotion decision for paired replay data."""

        control = _finite(control_scores, "control_scores")
        candidate = _finite(candidate_scores, "candidate_scores")
        if len(control) != len(candidate):
            raise ValueError("control_scores and candidate_scores must have the same length")
        if (control_costs is None) != (candidate_costs is None):
            raise ValueError("control_costs and candidate_costs must be provided together")
        if protected_mask is not None and len(protected_mask) != len(control):
            raise ValueError("protected_mask must have the same length as scores")

        costs_present = control_costs is not None and candidate_costs is not None
        parsed_control_costs: list[float] = []
        parsed_candidate_costs: list[float] = []
        if costs_present:
            parsed_control_costs = _finite(control_costs or [], "control_costs")
            parsed_candidate_costs = _finite(candidate_costs or [], "candidate_costs")
            if len(parsed_control_costs) != len(control) or len(parsed_candidate_costs) != len(
                control
            ):
                raise ValueError("cost arrays must have the same length as scores")
            if any(cost < 0.0 for cost in parsed_control_costs + parsed_candidate_costs):
                raise ValueError("costs must be non-negative")

        count = len(control)
        if count == 0:
            result = ValidationResult(
                candidate_id=candidate_id,
                candidate_type=candidate_type,
                n=0,
            )
            return PromotionDecision(
                promote=False,
                reason="failed gates: minimum_examples",
                result=result,
                gate_checks={
                    "minimum_examples": False,
                    "mean_gain": False,
                    "ci_lower_bound": False,
                    "regression_rate": False,
                    "protected_slice": False,
                    "cost": False,
                },
            )

        deltas = [new - old for old, new in zip(control, candidate)]
        mean_delta = statistics.fmean(deltas)
        ci_low, ci_high = paired_bootstrap_ci(
            deltas,
            samples=self.config.bootstrap_samples,
            confidence=self.config.confidence_level,
            seed=42,
        )
        regression_rate = statistics.fmean(float(delta < 0.0) for delta in deltas)

        protected_regression = 0.0
        if protected_mask is not None:
            protected_indices = [index for index, selected in enumerate(protected_mask) if selected]
            if protected_indices:
                protected_delta = statistics.fmean(deltas[index] for index in protected_indices)
                protected_regression = max(0.0, -protected_delta)

        cost_ratio = (
            _cost_delta_ratio(parsed_control_costs, parsed_candidate_costs)
            if costs_present
            else 0.0
        )
        checks = {
            "minimum_examples": count >= self.config.min_validation_examples,
            "mean_gain": mean_delta >= self.config.min_mean_gain,
            "ci_lower_bound": ci_low >= self.config.min_ci_lower_bound,
            "regression_rate": regression_rate <= self.config.max_regression_rate,
            "protected_slice": (protected_regression <= self.config.max_protected_slice_regression),
            "cost": cost_ratio <= self.config.max_cost_increase_ratio,
        }
        promote = all(checks.values())
        failed = [name for name, passed in checks.items() if not passed]
        reason = "all promotion gates passed" if promote else f"failed gates: {', '.join(failed)}"
        result = ValidationResult(
            candidate_id=candidate_id,
            candidate_type=candidate_type,
            n=count,
            control_mean=statistics.fmean(control),
            candidate_mean=statistics.fmean(candidate),
            mean_delta=mean_delta,
            ci_low=ci_low,
            ci_high=ci_high,
            regression_rate=regression_rate,
            cost_delta_ratio=cost_ratio,
            protected_slice_regression=protected_regression,
            deltas=deltas,
        )
        return PromotionDecision(
            promote=promote,
            reason=reason,
            result=result,
            gate_checks=checks,
        )


__all__ = ["PromotionGate", "paired_bootstrap_ci", "paired_cluster_bootstrap_ci"]
