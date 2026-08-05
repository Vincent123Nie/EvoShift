import json
from pathlib import Path

import pytest

from evoshift.schemas import Algorithm
from evoshift.sweep import (
    SweepSpec,
    aggregate_sweep,
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
    assert aggregate["variant"] == "full"


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
    assert len(matrix["runs"]) == 1
    assert matrix["runs"][0]["parameters"] == {"policy.top_k": 2}
    assert matrix["runs"][0]["variant"] == "default"
    assert "mean_recovery_steps" in matrix["runs"][0]
    assert matrix["runs"][0]["promotion_precision_basis"] == "not_applicable"
    assert matrix["runs"][0]["replay_estimated_promotion_precision"] is None
    assert matrix["runs"][0]["realized_promotion_precision"] is None
    assert "run_id,algorithm,variant,seed" in (destination / "matrix.csv").read_text(
        encoding="utf-8"
    )
    report = (destination / "report.md").read_text(encoding="utf-8")
    assert "Realized precision" in report
    assert "Only same-model" in report
