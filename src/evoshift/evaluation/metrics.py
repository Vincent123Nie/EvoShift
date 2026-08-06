"""Online and continual-learning metrics for EvoShift evaluation streams."""

from __future__ import annotations

import math
import statistics
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Callable, Union

from evoshift.schemas import Episode, PromotionDecision

Number = Union[int, float]


def _validate_finite(values: Sequence[Number], name: str = "values") -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite numbers")
    return result


def _scores(values: Sequence[Union[Episode, Number]]) -> list[float]:
    if not values:
        return []
    first = values[0]
    if isinstance(first, Episode):
        return [float(item.score.primary) for item in values]  # type: ignore[union-attr]
    return [float(item) for item in values]  # type: ignore[arg-type]


def mean_score(values: Sequence[Union[Episode, Number]]) -> float:
    """Mean primary score, returning 0 for an empty collection."""

    scores = _scores(values)
    return statistics.fmean(scores) if scores else 0.0


def success_rate(episodes: Sequence[Episode]) -> float:
    """Fraction of episodes whose evaluator marked them successful."""

    if not episodes:
        return 0.0
    return statistics.fmean(float(episode.score.success) for episode in episodes)


def phase_metrics(episodes: Sequence[Episode]) -> dict[str, dict[str, float | int]]:
    """Aggregate first-pass performance by stream phase, preserving stream order."""

    grouped: OrderedDict[str, list[Episode]] = OrderedDict()
    for episode in episodes:
        grouped.setdefault(episode.sample.phase, []).append(episode)
    return {
        phase: {
            "n": len(items),
            "mean_score": mean_score(items),
            "success_rate": success_rate(items),
        }
        for phase, items in grouped.items()
    }


def percentile(values: Sequence[Number], quantile: float) -> float:
    """Linearly interpolated percentile matching NumPy's default convention."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(_validate_finite(values))
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def area_under_adaptation_curve(
    accuracies: Sequence[Number],
    steps: Sequence[Number] | None = None,
    *,
    normalize: bool = True,
) -> float:
    """Trapezoidal area under accuracy versus adaptation-feedback steps.

    When ``normalize`` is true, the result is divided by the covered x-range
    and remains on the same [0, 1] scale as an accuracy curve.
    """

    values = _validate_finite(accuracies, "accuracies")
    if steps is None:
        x_values = [float(index) for index in range(len(values))]
    else:
        x_values = _validate_finite(steps, "steps")
        if len(x_values) != len(values):
            raise ValueError("steps and accuracies must have the same length")
        if any(right <= left for left, right in zip(x_values, x_values[1:])):
            raise ValueError("steps must be strictly increasing")
    if len(values) == 1:
        return values[0]

    area = sum(
        (right_x - left_x) * (left_y + right_y) / 2.0
        for left_x, right_x, left_y, right_y in zip(
            x_values,
            x_values[1:],
            values,
            values[1:],
        )
    )
    if normalize:
        area /= x_values[-1] - x_values[0]
    return area


def auac(
    accuracies: Sequence[Number],
    steps: Sequence[Number] | None = None,
    *,
    normalize: bool = True,
) -> float:
    """Short alias for :func:`area_under_adaptation_curve`."""

    return area_under_adaptation_curve(accuracies, steps, normalize=normalize)


def cumulative_regret(
    rewards: Sequence[Number],
    comparator: Union[Number, Sequence[Number]] = 1.0,
    *,
    clip_negative: bool = False,
) -> float:
    """Return cumulative comparator reward minus observed reward.

    ``comparator=1`` gives cumulative error for bounded task rewards.  Passing
    a score sequence supports a static baseline or domain-oracle comparator.
    Negative terms are retained by default so improvement over the comparator
    remains visible; set ``clip_negative`` for pseudo-regret.
    """

    observed = _validate_finite(rewards, "rewards")
    if isinstance(comparator, (int, float)):
        expected = [float(comparator)] * len(observed)
    else:
        expected = _validate_finite(comparator, "comparator")
        if len(expected) != len(observed):
            raise ValueError("comparator and rewards must have the same length")
    deltas = [target - reward for target, reward in zip(expected, observed)]
    if clip_negative:
        deltas = [max(0.0, value) for value in deltas]
    return sum(deltas)


def infer_shift_indices(episodes: Sequence[Episode]) -> list[int]:
    """Infer boundaries from phase transitions and explicit detector events."""

    boundaries = set()
    for index, episode in enumerate(episodes):
        if index > 0 and episode.sample.phase != episodes[index - 1].sample.phase:
            boundaries.add(index)
        if episode.shift is not None and episode.shift.detected:
            boundaries.add(index)
    return sorted(boundary for boundary in boundaries if 0 < boundary < len(episodes))


def post_shift_gain(
    candidate: Sequence[Union[Episode, Number]],
    baseline: Sequence[Union[Episode, Number]],
    shift_indices: Sequence[int],
    *,
    window: int = 20,
) -> float:
    """Mean paired gain in the first ``window`` episodes after each shift."""

    if window < 1:
        raise ValueError("window must be at least one")
    candidate_scores = _scores(candidate)
    baseline_scores = _scores(baseline)
    if len(candidate_scores) != len(baseline_scores):
        raise ValueError("candidate and baseline must have the same length")
    selected: set[int] = set()
    for shift_index in shift_indices:
        if not 0 <= shift_index < len(candidate_scores):
            raise ValueError("shift index is outside the score stream")
        selected.update(range(shift_index, min(len(candidate_scores), shift_index + window)))
    if not selected:
        return 0.0
    return statistics.fmean(
        candidate_scores[index] - baseline_scores[index] for index in sorted(selected)
    )


def recovery_steps(
    rewards: Sequence[Number],
    shift_index: int,
    *,
    recovery_fraction: float = 0.90,
    target: float | None = None,
    window: int = 5,
    pre_window: int | None = None,
) -> int | None:
    """Episodes needed for a post-shift rolling mean to recover.

    The default target is ``recovery_fraction`` times the pre-shift mean.  A
    return value of ``None`` means the stream ended before recovery.
    """

    values = _validate_finite(rewards, "rewards")
    if not 0 < shift_index < len(values):
        raise ValueError("shift_index must be inside the stream and greater than zero")
    if not 0.0 < recovery_fraction <= 1.0:
        raise ValueError("recovery_fraction must be in (0, 1]")
    if window < 1:
        raise ValueError("window must be at least one")
    if pre_window is not None and pre_window < 1:
        raise ValueError("pre_window must be at least one")

    pre_start = 0 if pre_window is None else max(0, shift_index - pre_window)
    pre_shift = values[pre_start:shift_index]
    threshold = (
        float(target) if target is not None else statistics.fmean(pre_shift) * recovery_fraction
    )
    if not math.isfinite(threshold):
        raise ValueError("target must be finite")

    post_shift = values[shift_index:]
    for consumed in range(window, len(post_shift) + 1):
        if statistics.fmean(post_shift[consumed - window : consumed]) >= threshold:
            return consumed
    return None


def _matrix_value(matrix: Sequence[Sequence[Number | None]], row: int, column: int) -> float | None:
    if row >= len(matrix) or column >= len(matrix[row]):
        return None
    value = matrix[row][column]
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def backward_transfer(performance_matrix: Sequence[Sequence[Number | None]]) -> float:
    """Average final-minus-post-training performance on previously seen domains."""

    if len(performance_matrix) < 2:
        return 0.0
    final_row = len(performance_matrix) - 1
    deltas = []
    for domain in range(final_row):
        learned = _matrix_value(performance_matrix, domain, domain)
        final = _matrix_value(performance_matrix, final_row, domain)
        if learned is not None and final is not None:
            deltas.append(final - learned)
    return statistics.fmean(deltas) if deltas else 0.0


def forgetting(performance_matrix: Sequence[Sequence[Number | None]]) -> float:
    """Average peak-to-final loss over previously learned domains."""

    if len(performance_matrix) < 2:
        return 0.0
    final_row = len(performance_matrix) - 1
    losses = []
    for domain in range(final_row):
        final = _matrix_value(performance_matrix, final_row, domain)
        history = [
            value
            for row in range(domain, len(performance_matrix))
            if (value := _matrix_value(performance_matrix, row, domain)) is not None
        ]
        if final is not None and history:
            losses.append(max(history) - final)
    return statistics.fmean(losses) if losses else 0.0


def promotion_precision(
    promotions: Sequence[Union[PromotionDecision, bool]],
    realized_gains: Sequence[Number] | None = None,
    *,
    beneficial_threshold: float = 0.0,
) -> float:
    """Fraction of promoted candidates that later show positive utility."""

    if realized_gains is not None and len(realized_gains) != len(promotions):
        raise ValueError("realized_gains and promotions must have the same length")
    promoted = 0
    beneficial = 0
    for index, value in enumerate(promotions):
        if isinstance(value, PromotionDecision):
            was_promoted = value.promote
            gain = (
                value.result.mean_delta if realized_gains is None else float(realized_gains[index])
            )
        else:
            was_promoted = bool(value)
            if realized_gains is None:
                raise ValueError("realized_gains are required for boolean promotion labels")
            gain = float(realized_gains[index])
        if was_promoted:
            promoted += 1
            beneficial += int(gain > beneficial_threshold)
    return beneficial / promoted if promoted else 0.0


def usage_metrics(episodes: Sequence[Episode]) -> dict[str, float | int]:
    """Aggregate all foreground token, dollar, and latency measurements."""

    if not episodes:
        return {
            "requests": 0,
            "cached_episodes": 0,
            "uncached_episodes": 0,
            "cache_hit_episode_rate": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "tokens_per_episode": 0.0,
            "tokens_per_success": 0.0,
            "cost_per_success_usd": 0.0,
            "latency_mean_ms": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
        }

    input_tokens = sum(episode.usage.input_tokens for episode in episodes)
    output_tokens = sum(episode.usage.output_tokens for episode in episodes)
    total_tokens = sum(
        episode.usage.total_tokens
        if episode.usage.total_tokens > 0
        else episode.usage.input_tokens + episode.usage.output_tokens
        for episode in episodes
    )
    cost = sum(episode.usage.cost_usd for episode in episodes)
    latencies = [episode.usage.latency_ms for episode in episodes]
    cached_episodes = sum(int(episode.usage.cached) for episode in episodes)
    successes = sum(int(episode.score.success) for episode in episodes)
    return {
        "requests": len(episodes),
        "cached_episodes": cached_episodes,
        "uncached_episodes": len(episodes) - cached_episodes,
        "cache_hit_episode_rate": cached_episodes / len(episodes),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost,
        "tokens_per_episode": total_tokens / len(episodes),
        "tokens_per_success": total_tokens / successes if successes else 0.0,
        "cost_per_success_usd": cost / successes if successes else 0.0,
        "latency_mean_ms": statistics.fmean(latencies),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
    }


def retrieval_metrics(episodes: Sequence[Episode]) -> dict[str, float | int]:
    """Report memory selection and solver-attributed application rates."""

    if not episodes:
        return {
            "episodes_with_retrieval": 0,
            "retrieval_episode_rate": 0.0,
            "selected_memories": 0,
            "episodes_with_applied_memory": 0,
            "applied_memory_episode_rate": 0.0,
            "rerank_attempts": 0,
            "rerank_applied": 0,
            "rerank_fallbacks": 0,
            "rerank_fallback_rate": 0.0,
        }
    retrieved = sum(bool(episode.selected_memory_ids) for episode in episodes)
    applied = sum(bool(episode.output.applied_memory_ids) for episode in episodes)
    rerank_attempts = sum(episode.rerank_attempted for episode in episodes)
    rerank_fallbacks = sum(episode.rerank_fallback for episode in episodes)
    return {
        "episodes_with_retrieval": retrieved,
        "retrieval_episode_rate": retrieved / len(episodes),
        "selected_memories": sum(len(episode.selected_memory_ids) for episode in episodes),
        "episodes_with_applied_memory": applied,
        "applied_memory_episode_rate": applied / len(episodes),
        "rerank_attempts": rerank_attempts,
        "rerank_applied": sum(episode.rerank_applied for episode in episodes),
        "rerank_fallbacks": rerank_fallbacks,
        "rerank_fallback_rate": (rerank_fallbacks / rerank_attempts if rerank_attempts else 0.0),
    }


def feedback_metrics(episodes: Sequence[Episode]) -> dict[str, float | int]:
    """Compare learner-visible feedback with hidden oracle evaluation."""

    if not episodes:
        return {
            "mean_score": 0.0,
            "success_rate": 0.0,
            "oracle_success_agreement_rate": 0.0,
            "false_positive_feedback_rate": 0.0,
            "false_negative_feedback_rate": 0.0,
            "mean_absolute_score_gap": 0.0,
            "mean_trust": 0.0,
            "eligible_feedback_rate": 0.0,
            "quarantined_feedback_n": 0,
            "corrupted_feedback_quarantine_rate": 0.0,
            "clean_feedback_quarantine_rate": 0.0,
            "annotated_noise_rate": 0.0,
            "clean_episodes": 0,
            "noise_episodes": 0,
            "attack_episodes": 0,
            "feedback_source_count": 0,
            "single_source_stream": False,
        }
    visible = [episode.adaptation_score for episode in episodes]
    agreement = sum(
        feedback.success == episode.score.success for episode, feedback in zip(episodes, visible)
    )
    false_positive = sum(
        feedback.success and not episode.score.success
        for episode, feedback in zip(episodes, visible)
    )
    false_negative = sum(
        episode.score.success and not feedback.success
        for episode, feedback in zip(episodes, visible)
    )
    kinds = [str(episode.sample.metadata.get("feedback_kind", "clean")) for episode in episodes]
    clean = sum(kind == "clean" for kind in kinds)
    noise = sum(kind == "noise" for kind in kinds)
    attack = sum(kind == "attack" for kind in kinds)
    sources = {
        str(episode.sample.metadata.get("feedback_source", "unspecified")) for episode in episodes
    }
    quarantined = [episode for episode in episodes if not episode.feedback_eligible]
    corrupted = [
        episode for episode in episodes if bool(episode.sample.metadata.get("feedback_corrupted"))
    ]
    annotated_clean = [
        episode
        for episode in episodes
        if not bool(episode.sample.metadata.get("feedback_corrupted"))
    ]
    return {
        "mean_score": statistics.fmean(score.primary for score in visible),
        "success_rate": statistics.fmean(float(score.success) for score in visible),
        "oracle_success_agreement_rate": agreement / len(episodes),
        "false_positive_feedback_rate": false_positive / len(episodes),
        "false_negative_feedback_rate": false_negative / len(episodes),
        "mean_absolute_score_gap": statistics.fmean(
            abs(feedback.primary - episode.score.primary)
            for episode, feedback in zip(episodes, visible)
        ),
        "mean_trust": statistics.fmean(episode.feedback_trust for episode in episodes),
        "eligible_feedback_rate": statistics.fmean(
            float(episode.feedback_eligible) for episode in episodes
        ),
        "quarantined_feedback_n": len(quarantined),
        "corrupted_feedback_quarantine_rate": (
            statistics.fmean(float(not episode.feedback_eligible) for episode in corrupted)
            if corrupted
            else 0.0
        ),
        "clean_feedback_quarantine_rate": (
            statistics.fmean(float(not episode.feedback_eligible) for episode in annotated_clean)
            if annotated_clean
            else 0.0
        ),
        "annotated_noise_rate": (noise + attack) / len(episodes),
        "clean_episodes": clean,
        "noise_episodes": noise,
        "attack_episodes": attack,
        "feedback_source_count": len(sources),
        "single_source_stream": len(sources) == 1,
    }


def policy_shift_metrics(episodes: Sequence[Episode]) -> dict[str, Any]:
    """Oracle metrics for policy updates, invariant retention, and corrupted feedback."""

    policy_episodes = [
        episode
        for episode in episodes
        if episode.sample.metadata.get("benchmark") == "policy_shift"
    ]
    if not policy_episodes:
        return {}

    def rate(items: Sequence[Episode], predicate: Callable[[Episode], bool]) -> float:
        return statistics.fmean(float(predicate(item)) for item in items) if items else 0.0

    changed = [
        episode
        for episode in policy_episodes
        if bool(episode.sample.metadata.get("policy_changed_case"))
    ]
    first_changed_by_phase: dict[str, Episode] = {}
    for episode in changed:
        first_changed_by_phase.setdefault(episode.sample.phase, episode)
    first_changed = list(first_changed_by_phase.values())
    protected = [
        episode for episode in policy_episodes if bool(episode.sample.metadata.get("protected"))
    ]
    future_change = [
        episode
        for episode in policy_episodes
        if bool(episode.sample.metadata.get("future_change_case"))
    ]
    corrupted = [
        episode
        for episode in policy_episodes
        if bool(episode.sample.metadata.get("feedback_corrupted"))
    ]
    noise = [
        episode
        for episode in policy_episodes
        if episode.sample.metadata.get("feedback_kind") == "noise"
    ]
    attacks = [
        episode
        for episode in policy_episodes
        if episode.sample.metadata.get("feedback_kind") == "attack"
    ]
    premature_attacks = [
        episode
        for episode in attacks
        if episode.sample.metadata.get("feedback_attack_goal") == "premature_update"
    ]
    post_attack_clean_future: list[Episode] = []
    for attack in premature_attacks:
        context = attack.sample.metadata.get("feedback_context")
        phase_index = attack.sample.metadata.get("phase_index")
        post_attack_clean_future.extend(
            episode
            for episode in policy_episodes
            if episode.index > attack.index
            and episode.sample.metadata.get("phase_index") == phase_index
            and episode.sample.metadata.get("feedback_context") == context
            and episode.sample.metadata.get("feedback_kind") == "clean"
            and bool(episode.sample.metadata.get("future_change_case"))
        )
    post_attack_clean_future = list(
        {episode.episode_id: episode for episode in post_attack_clean_future}.values()
    )
    phase_slice_counts: dict[str, dict[str, int]] = {}
    for episode in policy_episodes:
        counts = phase_slice_counts.setdefault(
            episode.sample.phase,
            {"total": 0, "transition": 0, "future": 0, "protected": 0},
        )
        counts["total"] += 1
        if bool(
            episode.sample.metadata.get(
                "transition_case",
                episode.sample.metadata.get("policy_changed_case"),
            )
        ):
            counts["transition"] += 1
        elif bool(episode.sample.metadata.get("future_change_case")):
            counts["future"] += 1
        elif bool(episode.sample.metadata.get("protected")):
            counts["protected"] += 1
    return {
        "n": len(policy_episodes),
        "phase_slice_counts": phase_slice_counts,
        "changed_case_n": len(changed),
        "changed_case_success_rate": rate(changed, lambda item: item.score.success),
        "first_changed_case_n": len(first_changed),
        "first_changed_case_success_rate": rate(
            first_changed,
            lambda item: item.score.success,
        ),
        "old_rule_leakage_rate": rate(changed, lambda item: not item.score.success),
        "invariant_n": len(protected),
        "invariant_retention_rate": rate(protected, lambda item: item.score.success),
        "future_change_case_n": len(future_change),
        "future_change_case_success_rate": rate(future_change, lambda item: item.score.success),
        "premature_update_rate": rate(future_change, lambda item: not item.score.success),
        "corrupted_feedback_n": len(corrupted),
        "corrupted_feedback_follow_rate": rate(
            corrupted, lambda item: item.adaptation_score.success
        ),
        "noise_oracle_success_rate": rate(noise, lambda item: item.score.success),
        "attack_oracle_success_rate": rate(attacks, lambda item: item.score.success),
        "attack_feedback_follow_rate": rate(attacks, lambda item: item.adaptation_score.success),
        "premature_attack_n": len(premature_attacks),
        "premature_attack_follow_rate": rate(
            premature_attacks,
            lambda item: item.adaptation_score.success,
        ),
        "post_attack_clean_future_n": len(post_attack_clean_future),
        "poison_persistence_error_rate": rate(
            post_attack_clean_future,
            lambda item: not item.score.success,
        ),
    }


def _align_baseline(episodes: Sequence[Episode], baseline: Sequence[Episode]) -> list[Episode]:
    by_sample = {episode.sample.sample_id: episode for episode in baseline}
    if len(by_sample) != len(baseline):
        raise ValueError("baseline contains duplicate sample ids")
    try:
        return [by_sample[episode.sample.sample_id] for episode in episodes]
    except KeyError as exc:
        raise ValueError(f"baseline is missing sample id: {exc.args[0]}") from exc


def compute_stream_metrics(
    episodes: Sequence[Episode],
    *,
    baseline_episodes: Sequence[Episode] | None = None,
    shift_indices: Sequence[int] | None = None,
    post_shift_window: int = 20,
    recovery_fraction: float = 0.90,
    recovery_window: int = 5,
    performance_matrix: Sequence[Sequence[Number | None]] | None = None,
    promotions: Sequence[PromotionDecision] | None = None,
    realized_promotion_gains: Sequence[Number] | None = None,
) -> dict[str, Any]:
    """Build the canonical JSON-ready summary for one prequential run."""

    rewards = _scores(episodes)
    boundaries = list(shift_indices) if shift_indices is not None else infer_shift_indices(episodes)
    report: dict[str, Any] = {
        "n_episodes": len(episodes),
        "overall": {
            "mean_score": mean_score(episodes),
            "success_rate": success_rate(episodes),
        },
        "phases": phase_metrics(episodes),
        "auac": area_under_adaptation_curve(rewards) if rewards else 0.0,
        "cumulative_regret": cumulative_regret(rewards) if rewards else 0.0,
        "shift_indices": boundaries,
        "feedback": feedback_metrics(episodes),
        "retrieval": retrieval_metrics(episodes),
        "resources": usage_metrics(episodes),
    }
    policy_report = policy_shift_metrics(episodes)
    if policy_report:
        report["policy_shift"] = policy_report

    recovery: dict[str, int | None] = {}
    for boundary in boundaries:
        recovery[str(boundary)] = recovery_steps(
            rewards,
            boundary,
            recovery_fraction=recovery_fraction,
            window=recovery_window,
        )
    report["recovery_steps"] = recovery

    if baseline_episodes is not None:
        aligned = _align_baseline(episodes, baseline_episodes)
        baseline_rewards = _scores(aligned)
        report["post_shift_gain"] = post_shift_gain(
            rewards,
            baseline_rewards,
            boundaries,
            window=post_shift_window,
        )
        report["cumulative_regret_vs_baseline"] = cumulative_regret(
            rewards,
            baseline_rewards,
        )
    if performance_matrix is not None:
        report["continual_learning"] = {
            "backward_transfer": backward_transfer(performance_matrix),
            "forgetting": forgetting(performance_matrix),
        }
    if promotions is not None:
        report["promotion_precision"] = promotion_precision(
            promotions,
            realized_promotion_gains,
        )
    return report


__all__ = [
    "area_under_adaptation_curve",
    "auac",
    "backward_transfer",
    "compute_stream_metrics",
    "cumulative_regret",
    "feedback_metrics",
    "forgetting",
    "infer_shift_indices",
    "mean_score",
    "percentile",
    "phase_metrics",
    "policy_shift_metrics",
    "post_shift_gain",
    "promotion_precision",
    "recovery_steps",
    "retrieval_metrics",
    "success_rate",
    "usage_metrics",
]
