from __future__ import annotations

import json

import pytest

from evoshift.evaluation import (
    area_under_adaptation_curve,
    backward_transfer,
    binary_choice_score,
    compute_stream_metrics,
    cumulative_regret,
    exact_match_score,
    forgetting,
    multiple_choice_score,
    normalized_exact_match_score,
    numeric_score,
    phase_metrics,
    post_shift_gain,
    promotion_precision,
    recovery_steps,
    render_markdown_report,
    score_sample,
    success_rate,
    to_json_report,
    token_f1_score,
    usage_metrics,
)
from evoshift.schemas import (
    BenchmarkSample,
    Episode,
    LLMUsage,
    ScoreBundle,
    SolverOutput,
)


def _episode(
    index: int,
    score: float,
    phase: str,
    *,
    sample_id: str = "",
    latency_ms: float = 100.0,
    cached: bool = False,
) -> Episode:
    return Episode(
        episode_id=f"episode-{index}",
        run_id="run-1",
        index=index,
        sample=BenchmarkSample(
            sample_id=sample_id or f"sample-{index}",
            prompt="question",
            reference="answer",
            phase=phase,
        ),
        output=SolverOutput(answer="answer"),
        score=ScoreBundle(primary=score, success=score >= 1.0),
        usage=LLMUsage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cost_usd=0.01,
            latency_ms=latency_ms,
            cached=cached,
        ),
    )


def test_deterministic_scorers_cover_common_dataset_formats() -> None:
    assert exact_match_score("Paris", ["London", "Paris"]) == 1.0
    assert exact_match_score("paris", "Paris") == 0.0
    assert normalized_exact_match_score(" The, QUICK fox! ", "quick fox") == 1.0
    assert token_f1_score("red blue", "red green") == pytest.approx(0.5)
    assert token_f1_score("北京", "北京") == 1.0
    assert numeric_score("The result is 50%.", 0.5) == 1.0
    assert numeric_score("answer: 1/4", 0.25) == 1.0
    assert numeric_score("3.1415", 3.14, absolute_tolerance=0.01) == 1.0
    assert (
        multiple_choice_score(
            "The answer is (B).",
            "B",
            choices=["red", "blue", "green"],
        )
        == 1.0
    )
    assert multiple_choice_score("blue", "B", choices=["red", "blue", "green"]) == 1.0
    assert multiple_choice_score("09/09/1908 (B)", "(B)") == 1.0
    assert multiple_choice_score("prose ending in B", "(B)") == 0.0
    assert binary_choice_score("Yes, Christie tells the truth.", "Yes") == 1.0
    assert binary_choice_score("No. Lorine lies.", "No") == 1.0
    assert binary_choice_score("True, because the expression holds.", "Yes") == 1.0
    assert binary_choice_score("The explanation ends with yes", "Yes") == 0.0


def test_score_sample_is_stable_public_dispatch_api() -> None:
    sample = BenchmarkSample(
        sample_id="n-1",
        prompt="What is half?",
        reference=0.5,
        evaluator="numeric",
        metadata={"abs_tol": 1e-3},
    )
    result = score_sample(sample, "0.5001")
    assert result.primary == 1.0
    assert result.success is True
    assert result.metrics == {"numeric": 1.0}


def test_online_metrics_and_phase_aggregation() -> None:
    episodes = [
        _episode(0, 1.0, "warmup"),
        _episode(1, 0.0, "warmup"),
        _episode(2, 1.0, "shifted"),
        _episode(3, 1.0, "shifted"),
    ]
    assert success_rate(episodes) == pytest.approx(0.75)
    phases = phase_metrics(episodes)
    assert list(phases) == ["warmup", "shifted"]
    assert phases["warmup"] == {
        "n": 2,
        "mean_score": 0.5,
        "success_rate": 0.5,
    }
    assert area_under_adaptation_curve([0.0, 0.5, 1.0]) == pytest.approx(0.5)
    assert cumulative_regret([1.0, 0.5, 0.0]) == pytest.approx(1.5)


def test_shift_gain_and_recovery_have_explicit_window_semantics() -> None:
    assert (
        post_shift_gain(
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [2],
            window=2,
        )
        == 1.0
    )
    assert (
        recovery_steps(
            [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            2,
            recovery_fraction=0.9,
            window=2,
        )
        == 4
    )
    assert recovery_steps([1.0, 1.0, 0.0, 0.0], 2, window=2) is None


def test_backward_transfer_and_forgetting_follow_continual_learning_definitions() -> None:
    matrix = [
        [0.80, None, None],
        [0.70, 0.75, None],
        [0.60, 0.80, 0.90],
    ]
    assert backward_transfer(matrix) == pytest.approx(-0.075)
    assert forgetting(matrix) == pytest.approx(0.10)


def test_resource_metrics_report_tokens_cost_and_latency_tails() -> None:
    episodes = [
        _episode(
            index,
            1.0 if index != 1 else 0.0,
            "stream",
            latency_ms=latency,
            cached=index == 3,
        )
        for index, latency in enumerate([100.0, 200.0, 300.0, 400.0])
    ]
    resources = usage_metrics(episodes)
    assert resources["total_tokens"] == 60.0
    assert resources["cost_usd"] == pytest.approx(0.04)
    assert resources["tokens_per_success"] == 20.0
    assert resources["cached_episodes"] == 1
    assert resources["uncached_episodes"] == 3
    assert resources["cache_hit_episode_rate"] == 0.25
    assert resources["latency_p50_ms"] == 250.0
    assert resources["latency_p95_ms"] == pytest.approx(385.0)


def test_promotion_precision_uses_only_promoted_candidates() -> None:
    assert promotion_precision([True, False, True], [0.2, 0.5, -0.1]) == 0.5
    assert promotion_precision([False, False], [1.0, 1.0]) == 0.0


def test_stream_report_aligns_baseline_by_sample_id_and_serializes() -> None:
    episodes = [
        _episode(0, 1.0, "base", sample_id="a"),
        _episode(1, 0.0, "base", sample_id="b"),
        _episode(2, 1.0, "shift", sample_id="c"),
        _episode(3, 1.0, "shift", sample_id="d"),
    ]
    baseline: list[Episode] = [
        _episode(0, 0.0, "base", sample_id="b"),
        _episode(1, 0.0, "shift", sample_id="d"),
        _episode(2, 0.0, "base", sample_id="a"),
        _episode(3, 0.0, "shift", sample_id="c"),
    ]
    report = compute_stream_metrics(
        episodes,
        baseline_episodes=baseline,
        post_shift_window=2,
        recovery_window=1,
        performance_matrix=[[0.8, None], [0.7, 0.9]],
    )
    assert report["shift_indices"] == [2]
    assert report["post_shift_gain"] == 1.0
    assert report["continual_learning"]["backward_transfer"] == pytest.approx(-0.1)

    json_text = to_json_report(report)
    assert json.loads(json_text)["overall"]["success_rate"] == 0.75
    markdown = render_markdown_report(report, title="Test Run")
    assert "# Test Run" in markdown
    assert "Performance by phase" in markdown
    assert "latency p95 ms" in markdown
