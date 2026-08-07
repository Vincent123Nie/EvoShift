from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evoshift.cli import app
from evoshift.schemas import Algorithm, GenerationResponse, LLMUsage
from evoshift.sweep import SweepSpec

runner = CliRunner()


def test_doctor_is_offline_and_reports_config() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "EvoShift doctor" in result.stdout
    assert "Config hash" in result.stdout


def test_benchmark_list() -> None:
    result = runner.invoke(app, ["benchmark", "list"])
    retrieval_help = runner.invoke(app, ["benchmark", "retrieval-eval", "--help"])
    assert result.exit_code == 0
    assert "synthetic_shift" in result.stdout
    assert "bbh" in result.stdout
    assert "longmemeval_s" in result.stdout
    assert retrieval_help.exit_code == 0
    assert "bm25_llm_rerank" in retrieval_help.stdout
    assert "max-per-type" in retrieval_help.stdout


def test_version_and_provider_smoke_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    class Client:
        async def generate(self, _request: Any) -> GenerationResponse:
            return GenerationResponse(
                text="OK",
                model="evoshift-demo",
                usage=LLMUsage(total_tokens=3, latency_ms=1.25),
            )

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("evoshift.cli.create_client", lambda _config: Client())

    version = runner.invoke(app, ["version"])
    smoke = runner.invoke(
        app,
        [
            "provider",
            "smoke",
            "--config",
            "configs/experiments/offline_demo.yaml",
        ],
    )

    assert version.exit_code == 0
    assert smoke.exit_code == 0
    assert "PASS" in smoke.stdout
    assert "evoshift-demo" in smoke.stdout


def test_data_pull_success_and_error_are_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Adapter:
        manifest_path = tmp_path / "manifest.json"

        def download(self) -> dict[str, Path]:
            return {"sample": tmp_path / "sample.json"}

    monkeypatch.setattr(
        "evoshift.benchmarks.create_benchmark",
        lambda *_args, **_kwargs: Adapter(),
    )
    success = runner.invoke(
        app,
        [
            "data",
            "pull",
            "bbh",
            "--config",
            "configs/experiments/evoshift_bbh_smoke.yaml",
        ],
    )
    failure = runner.invoke(app, ["data", "pull", "huggingface"])

    assert success.exit_code == 0
    assert "Verified 1 BBH files" in success.stdout
    assert failure.exit_code == 1
    assert "currently supports 'bbh'" in failure.stdout


def test_cli_run_audit_compare_report_and_memory_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("evoshift.runner.git_state", lambda _workdir: ("test-commit", False))
    source_root = tmp_path / "source"
    baseline_root = tmp_path / "baseline"
    audit_root = tmp_path / "audit"
    source_config = Path("configs/experiments/offline_demo.yaml").resolve()
    baseline_config = Path("configs/experiments/static_demo.yaml").resolve()

    source_result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(source_config),
            "--set",
            f"storage.runs_dir={source_root}",
            "--set",
            "storage.cache_enabled=false",
        ],
    )
    baseline_result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(baseline_config),
            "--set",
            f"storage.runs_dir={baseline_root}",
            "--set",
            "storage.cache_enabled=false",
        ],
    )
    assert source_result.exit_code == 0, source_result.stdout
    assert baseline_result.exit_code == 0, baseline_result.stdout
    source_run = next(path for path in source_root.iterdir() if path.is_dir())
    baseline_run = next(path for path in baseline_root.iterdir() if path.is_dir())

    audit_result = runner.invoke(
        app,
        [
            "audit",
            "--source-run",
            str(source_run),
            "--config",
            str(source_config),
            "--set",
            "evaluation.seed=99",
            "--set",
            f"storage.runs_dir={audit_root}",
        ],
    )
    compare_result = runner.invoke(app, ["compare", str(baseline_run), str(source_run)])
    report_result = runner.invoke(app, ["report", str(source_run)])
    memory_result = runner.invoke(
        app,
        [
            "memory",
            "inspect",
            "--database",
            str(source_run / "state.sqlite3"),
            "--status",
            "all",
        ],
    )

    assert audit_result.exit_code == 0, audit_result.stdout
    assert "Frozen audit complete" in audit_result.stdout
    assert compare_result.exit_code == 0, compare_result.stdout
    assert '"comparable_for_claims": true' in compare_result.stdout
    assert '"resource_comparable": true' in compare_result.stdout
    assert '"cache_disabled_for_both": true' in compare_result.stdout
    assert '"provenance_comparable": true' in compare_result.stdout
    assert (source_run / f"compare_vs_{baseline_run.name}.json").exists()
    assert report_result.exit_code == 0
    assert memory_result.exit_code == 0
    assert "Memories: all" in memory_result.stdout


def test_sweep_cli_reports_bounded_run_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text("placeholder: true\n", encoding="utf-8")
    spec = SweepSpec(
        base_config=Path("unused.yaml"),
        algorithms=[Algorithm.STATIC],
        seeds=[1],
        grid={},
        max_runs=1,
    )

    monkeypatch.setattr("evoshift.sweep.load_sweep_spec", lambda *_args: spec)
    monkeypatch.setattr("evoshift.sweep.expand_sweep", lambda _spec: [{}])

    async def fake_run_sweep(_spec: SweepSpec, _root: Path) -> Path:
        return tmp_path / "sweep-output"

    monkeypatch.setattr("evoshift.sweep.run_sweep", fake_run_sweep)
    result = runner.invoke(app, ["sweep", "--spec", str(spec_path)])

    assert result.exit_code == 0
    assert "Executing 1 bounded sweep runs" in result.stdout
    assert "Sweep complete" in result.stdout
