import json
from pathlib import Path

import pytest

from evoshift.schemas import (
    Algorithm,
    BenchmarkSample,
    Episode,
    ScoreBundle,
    SolverOutput,
)
from evoshift.sweep import (
    SweepSpec,
    aggregate_sweep,
    compare_sweep_runs,
    compare_sweep_variants,
    expand_sweep,
    load_sweep_spec,
    run_sweep,
)


def test_sweep_cartesian_product() -> None:
    spec = SweepSpec(
        base_config=Path("base.yaml"),
        algorithms=[Algorithm.STATIC, Algorithm.EVOSHIFT],
        seeds=[1, 2],
        grid={"policy.top_k": [2, 4]},
    )
    assignments = expand_sweep(spec)
    assert len(assignments) == 8
    assert {item["algorithm"] for item in assignments} == {"static", "evoshift"}
    assert {item["variant"] for item in assignments} == {"default"}


def test_replay_only_is_a_first_class_sweep_algorithm() -> None:
    spec = SweepSpec(
        base_config=Path("base.yaml"),
        algorithms=[Algorithm.REPLAY_ONLY],
        seeds=[11, 22],
        grid={},
    )

    assignments = expand_sweep(spec)

    assert [item["algorithm"] for item in assignments] == ["replay_only", "replay_only"]


def test_sweep_expands_named_ablation_variants() -> None:
    spec = SweepSpec(
        base_config=Path("base.yaml"),
        algorithms=[Algorithm.EVOSHIFT],
        seeds=[1, 2],
        grid={"benchmark.feedback_noise_rate": [0.0, 0.1]},
        variants={
            "full": {},
            "historical_replay": {"evolution.replay_current_regime_only": False},
        },
        max_runs=8,
    )

    assignments = expand_sweep(spec)

    assert len(assignments) == 8
    assert {item["variant"] for item in assignments} == {"full", "historical_replay"}
    historical = [item for item in assignments if item["variant"] == "historical_replay"]
    assert all(
        item["parameters"]["evolution.replay_current_regime_only"] is False for item in historical
    )


def test_sweep_spec_blocks_accidental_overspend(tmp_path: Path) -> None:
    spec = tmp_path / "sweep.yaml"
    spec.write_text(
        "base_config: configs/default.yaml\n"
        "algorithms: [static, evoshift]\n"
        "seeds: [1, 2]\n"
        "grid:\n  policy.top_k: [1, 2]\n"
        "max_runs: 3\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="max_runs"):
        load_sweep_spec(spec, Path.cwd())


def test_sweep_spec_rejects_unknown_fields_and_empty_axes(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(
        "base_config: configs/default.yaml\nruns_dir: custom\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown sweep fields: runs_dir"):
        load_sweep_spec(unknown, Path.cwd())

    empty = tmp_path / "empty.yaml"
    empty.write_text(
        "base_config: configs/default.yaml\nalgorithms: []\nseeds: []\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="algorithms must not be empty"):
        load_sweep_spec(empty, Path.cwd())


def test_sweep_aggregation_reports_seed_variance() -> None:
    rows = [
        {
            "algorithm": "evoshift",
            "variant": "full",
            "parameters": {},
            "mean_score": 0.7,
            "changed_case_success_rate": 0.5,
            "old_rule_leakage_rate": 0.5,
            "total_tokens": 10,
            "total_requests": 4,
            "revival_context_recorded_opportunities": 1,
            "revival_context_guard_opportunities": 2,
            "run_id": "a",
        },
        {
            "algorithm": "evoshift",
            "variant": "full",
            "parameters": {},
            "mean_score": 0.9,
            "changed_case_success_rate": 0.9,
            "old_rule_leakage_rate": 0.1,
            "total_tokens": 14,
            "total_requests": 6,
            "revival_context_recorded_opportunities": 2,
            "revival_context_guard_opportunities": 2,
            "run_id": "b",
        },
    ]
    aggregate = aggregate_sweep(rows)[0]
    assert aggregate["score_mean"] == pytest.approx(0.8)
    assert aggregate["score_std"] > 0
    assert aggregate["total_tokens_mean"] == 12
    assert aggregate["total_requests_mean"] == 5
    assert aggregate["changed_case_success_mean"] == pytest.approx(0.7)
    assert aggregate["old_rule_leakage_mean"] == pytest.approx(0.3)
    assert aggregate["revival_context_record_coverage_micro"] == pytest.approx(0.75)
    assert aggregate["variant"] == "full"


def _write_comparison_run(
    run_dir: Path,
    *,
    run_id: str,
    scores: list[float],
) -> None:
    run_dir.mkdir(parents=True)
    episodes = []
    for index, score in enumerate(scores):
        changed = index == 0
        episodes.append(
            Episode(
                episode_id=f"{run_id}-{index}",
                run_id=run_id,
                index=index,
                sample=BenchmarkSample(
                    sample_id=f"sample-{index}",
                    prompt="case",
                    reference="ALLOW",
                    metadata={
                        "policy_changed_case": changed,
                        "protected": not changed,
                        "future_change_case": False,
                        "feedback_kind": "clean",
                    },
                ),
                output=SolverOutput(answer="ALLOW" if score else "DENY"),
                score=ScoreBundle(primary=score, success=bool(score)),
                feedback_score=ScoreBundle(primary=score, success=bool(score)),
            )
        )
    (run_dir / "predictions.jsonl").write_text(
        "\n".join(episode.model_dump_json() for episode in episodes) + "\n",
        encoding="utf-8",
    )


def test_compare_sweep_runs_reports_clustered_capability_safety_and_cost(
    tmp_path: Path,
) -> None:
    rows = []
    for stream_seed in (11, 22):
        baseline_dir = tmp_path / f"baseline-{stream_seed}"
        candidate_dir = tmp_path / f"candidate-{stream_seed}"
        _write_comparison_run(
            baseline_dir,
            run_id=f"baseline-{stream_seed}",
            scores=[0.0, 1.0] if stream_seed == 11 else [0.0, 0.0],
        )
        _write_comparison_run(
            candidate_dir,
            run_id=f"candidate-{stream_seed}",
            scores=[1.0, 1.0] if stream_seed == 11 else [0.0, 1.0],
        )
        common = {
            "variant": "default",
            "seed": stream_seed,
            "parameters": {"benchmark.feedback_noise_rate": 0.0},
            "cost_usd": 0.0,
            "harmful_active_memory_exposure_n": 0,
            "stale_memory_retention_rate": 0.0,
            "false_retirement_rate": 0.0,
        }
        rows.extend(
            [
                {
                    **common,
                    "algorithm": "static",
                    "run_dir": str(baseline_dir),
                    "total_requests": 2,
                    "total_tokens": 20,
                },
                {
                    **common,
                    "algorithm": "evoshift",
                    "run_dir": str(candidate_dir),
                    "total_requests": 3,
                    "total_tokens": 30,
                },
            ]
        )

    comparisons = compare_sweep_runs(rows, samples=1000, bootstrap_seed=7)

    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison["n_seeds"] == 2
    assert comparison["bootstrap_unit"] == "seed_then_paired_sample"
    assert comparison["metrics"]["score"]["delta_mean"] == pytest.approx(0.5)
    assert comparison["metrics"]["score"]["n_pairs"] == 4
    assert comparison["metrics"]["total_requests"]["delta_mean"] == 1.0
    assert comparison["metrics"]["total_tokens"]["delta_mean"] == 10.0


def test_compare_sweep_variants_holds_grid_and_algorithm_fixed(tmp_path: Path) -> None:
    rows = []
    for stream_seed in (11, 22):
        baseline_dir = tmp_path / f"full-{stream_seed}"
        candidate_dir = tmp_path / f"circuit-{stream_seed}"
        _write_comparison_run(
            baseline_dir,
            run_id=f"full-{stream_seed}",
            scores=[0.0, 1.0],
        )
        _write_comparison_run(
            candidate_dir,
            run_id=f"circuit-{stream_seed}",
            scores=[1.0, 1.0],
        )
        common = {
            "algorithm": "evoshift",
            "seed": stream_seed,
            "cost_usd": 0.0,
            "harmful_active_memory_exposure_n": 0,
            "stale_memory_retention_rate": 0.0,
            "false_retirement_rate": 0.0,
            "total_requests": 2,
            "total_tokens": 20,
        }
        rows.extend(
            [
                {
                    **common,
                    "variant": "current_full",
                    "parameters": {
                        "benchmark.feedback_noise_rate": 0.1,
                        "evolution.active_audit_circuit_breaker_enabled": False,
                    },
                    "run_dir": str(baseline_dir),
                },
                {
                    **common,
                    "variant": "recurrence_circuit_breaker",
                    "parameters": {
                        "benchmark.feedback_noise_rate": 0.1,
                        "evolution.active_audit_circuit_breaker_enabled": True,
                    },
                    "run_dir": str(candidate_dir),
                },
            ]
        )

    comparisons = compare_sweep_variants(
        rows,
        variant_parameters={
            "current_full": {
                "evolution.active_audit_circuit_breaker_enabled": False,
            },
            "recurrence_circuit_breaker": {
                "evolution.active_audit_circuit_breaker_enabled": True,
            },
        },
        samples=1000,
        bootstrap_seed=7,
    )

    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison["candidate_algorithm"] == "evoshift:recurrence_circuit_breaker"
    assert comparison["baseline_algorithm"] == "evoshift:current_full"
    assert comparison["parameters"] == {"benchmark.feedback_noise_rate": 0.1}
    assert comparison["metrics"]["score"]["delta_mean"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_run_sweep_writes_matrix_csv_and_report(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        "project: sweep-test\n"
        "algorithm: static\n"
        "provider:\n  kind: demo\n  model: evoshift-demo\n  reasoning_effort: null\n"
        "storage:\n  cache_enabled: false\n  isolate_runs: true\n"
        "benchmark:\n  kind: synthetic_shift\n  path: null\n  phase_size: 2\n"
        "evolution:\n  enabled: false\n",
        encoding="utf-8",
    )
    spec = SweepSpec(
        base_config=base,
        algorithms=[Algorithm.STATIC],
        seeds=[7],
        grid={"policy.top_k": [2]},
        max_runs=1,
    )

    destination = await run_sweep(spec, tmp_path)

    matrix = json.loads((destination / "matrix.json").read_text(encoding="utf-8"))
    state = json.loads((destination / "sweep_state.json").read_text(encoding="utf-8"))
    assert matrix["complete"] is True
    assert matrix["completed_runs"] == 1
    assert len(matrix["runs"]) == 1
    assert matrix["runs"][0]["parameters"] == {"policy.top_k": 2}
    assert matrix["runs"][0]["variant"] == "default"
    assert "mean_recovery_steps" in matrix["runs"][0]
    assert matrix["runs"][0]["promotion_precision_basis"] == "not_applicable"
    assert matrix["runs"][0]["replay_estimated_promotion_precision"] is None
    assert matrix["runs"][0]["realized_promotion_precision"] is None
    assert matrix["runs"][0]["phase_slice_counts"] == {}
    assert "run_id,algorithm,variant,seed" in (destination / "matrix.csv").read_text(
        encoding="utf-8"
    )
    report = (destination / "report.md").read_text(encoding="utf-8")
    assert "Realized precision" in report
    assert "Lifecycle path diagnostics" in report
    assert "Only same-model" in report
    assert state["status"] == "complete"
    assert state["completed_runs"] == 1
    assert state["assignments"][0]["status"] == "completed"
    assert state["assignments"][0]["attempts"][0]["status"] == "completed"
    assert not (destination / ".sweep_state.json.tmp").exists()


@pytest.mark.asyncio
async def test_run_sweep_resumes_failed_assignment_without_rerunning_completed_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        "project: sweep-resume-test\n"
        "algorithm: static\n"
        "provider:\n  kind: demo\n  model: evoshift-demo\n  reasoning_effort: null\n"
        "storage:\n  cache_enabled: false\n  isolate_runs: true\n"
        "benchmark:\n  kind: synthetic_shift\n  path: null\n  phase_size: 1\n"
        "evolution:\n  enabled: false\n",
        encoding="utf-8",
    )
    spec = SweepSpec(
        base_config=base,
        algorithms=[Algorithm.STATIC],
        seeds=[7, 8],
        grid={},
        max_runs=2,
    )
    from evoshift.runner import EvoShiftRunner as RealRunner

    original_run = RealRunner.run
    call_count = 0

    async def fail_second_run(self: RealRunner):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("injected interruption")
        return await original_run(self)

    monkeypatch.setattr("evoshift.sweep.EvoShiftRunner.run", fail_second_run)
    with pytest.raises(RuntimeError, match="injected interruption"):
        await run_sweep(spec, tmp_path)

    destination = next((tmp_path / "runs" / "sweeps").iterdir())
    partial_matrix = json.loads((destination / "matrix.json").read_text(encoding="utf-8"))
    partial_state = json.loads((destination / "sweep_state.json").read_text(encoding="utf-8"))
    first_run_id = partial_matrix["runs"][0]["run_id"]
    assert partial_matrix["complete"] is False
    assert partial_matrix["completed_runs"] == 1
    assert [entry["status"] for entry in partial_state["assignments"]] == [
        "completed",
        "failed",
    ]
    assert partial_state["assignments"][1]["attempts"][0]["error"].endswith("injected interruption")

    resumed_calls = 0

    async def count_resumed_run(self: RealRunner):  # type: ignore[no-untyped-def]
        nonlocal resumed_calls
        resumed_calls += 1
        return await original_run(self)

    monkeypatch.setattr("evoshift.sweep.EvoShiftRunner.run", count_resumed_run)
    resumed = await run_sweep(spec, tmp_path, resume=destination)

    matrix = json.loads((resumed / "matrix.json").read_text(encoding="utf-8"))
    state = json.loads((resumed / "sweep_state.json").read_text(encoding="utf-8"))
    assert resumed == destination
    assert resumed_calls == 1
    assert matrix["complete"] is True
    assert matrix["completed_runs"] == 2
    assert matrix["runs"][0]["run_id"] == first_run_id
    assert state["status"] == "complete"
    assert [entry["status"] for entry in state["assignments"]] == [
        "completed",
        "completed",
    ]
    assert [attempt["status"] for attempt in state["assignments"][1]["attempts"]] == [
        "failed",
        "completed",
    ]


@pytest.mark.asyncio
async def test_run_sweep_rejects_resume_when_spec_fingerprint_changes(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        "project: sweep-fingerprint-test\n"
        "algorithm: static\n"
        "provider:\n  kind: demo\n  model: evoshift-demo\n  reasoning_effort: null\n"
        "storage:\n  cache_enabled: false\n  isolate_runs: true\n"
        "benchmark:\n  kind: synthetic_shift\n  path: null\n  phase_size: 1\n"
        "evolution:\n  enabled: false\n",
        encoding="utf-8",
    )
    original = SweepSpec(base, [Algorithm.STATIC], [7], {}, max_runs=1)
    destination = await run_sweep(original, tmp_path)
    changed = SweepSpec(
        base,
        [Algorithm.STATIC],
        [7],
        {"policy.top_k": [2]},
        max_runs=1,
    )

    with pytest.raises(ValueError, match="fingerprint"):
        await run_sweep(changed, tmp_path, resume=destination)
