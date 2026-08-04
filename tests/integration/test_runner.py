import json
from pathlib import Path

import pytest

from evoshift.audit import load_evolved_state
from evoshift.benchmarks import create_benchmark
from evoshift.benchmarks.base import sample_fingerprint
from evoshift.config import load_config
from evoshift.runner import EvoShiftRunner


@pytest.mark.asyncio
@pytest.mark.integration
async def test_demo_runner_evolves_and_writes_auditable_artifacts(tmp_path: Path) -> None:
    config = load_config(Path("configs/experiments/offline_demo.yaml"))
    config = config.model_copy(
        update={
            "storage": config.storage.model_copy(
                update={"runs_dir": str(tmp_path / "runs"), "cache_enabled": False}
            )
        }
    )
    adapter = create_benchmark(config.benchmark, root=Path.cwd(), seed=config.evaluation.seed)

    result = await EvoShiftRunner(config, adapter, workdir=Path.cwd()).run()

    assert result.metrics["overall"]["mean_score"] >= 0.80
    assert result.metrics["evolution"]["final_active_memories"] >= 2
    assert any(decision.promote for decision in result.decisions)
    for name in (
        "manifest.json",
        "resolved_config.yaml",
        "predictions.jsonl",
        "traces.jsonl",
        "promotion_decisions.jsonl",
        "metrics.json",
        "costs.json",
        "summary.json",
        "report.md",
        "state.sqlite3",
    ):
        assert (result.run_dir / name).exists(), name
    predictions = (result.run_dir / "predictions.jsonl").read_text(encoding="utf-8")
    assert "EVOSHIFT_OPENAI_API_KEY" not in predictions
    assert "sk-" not in predictions


@pytest.mark.asyncio
@pytest.mark.integration
async def test_policy_shift_runner_keeps_oracle_and_feedback_channels_separate(
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/experiments/static_policy_shift_demo.yaml"))
    config = config.model_copy(
        update={
            "storage": config.storage.model_copy(
                update={"runs_dir": str(tmp_path / "runs"), "cache_enabled": False}
            ),
            "benchmark": config.benchmark.model_copy(
                update={"phase_size": 8, "feedback_noise_rate": 1.0}
            ),
        }
    )
    adapter = create_benchmark(config.benchmark, root=Path.cwd(), seed=config.evaluation.seed)

    result = await EvoShiftRunner(config, adapter, workdir=Path.cwd()).run()

    assert result.metrics["feedback"]["annotated_noise_rate"] == 1.0
    assert result.metrics["feedback"]["oracle_success_agreement_rate"] == 0.0
    predictions = (result.run_dir / "predictions.jsonl").read_text(encoding="utf-8")
    assert '"feedback_score"' in predictions


@pytest.mark.asyncio
@pytest.mark.integration
async def test_frozen_audit_reuses_state_without_mutating_it(tmp_path: Path) -> None:
    source_config = load_config(Path("configs/experiments/offline_demo.yaml"))
    source_config = source_config.model_copy(
        update={
            "storage": source_config.storage.model_copy(
                update={"runs_dir": str(tmp_path / "source"), "cache_enabled": False}
            )
        }
    )
    source_adapter = create_benchmark(
        source_config.benchmark,
        root=Path.cwd(),
        seed=source_config.evaluation.seed,
    )
    source_result = await EvoShiftRunner(
        source_config,
        source_adapter,
        workdir=Path.cwd(),
    ).run()
    state = load_evolved_state(source_result.run_dir)

    audit_config = source_config.model_copy(
        update={
            "storage": source_config.storage.model_copy(
                update={"runs_dir": str(tmp_path / "audit"), "cache_enabled": False}
            ),
            "evaluation": source_config.evaluation.model_copy(update={"seed": 99}),
        }
    )
    audit_adapter = create_benchmark(
        audit_config.benchmark,
        root=Path.cwd(),
        seed=audit_config.evaluation.seed,
    )
    assert sample_fingerprint(audit_adapter.load()) != state.source_dataset_hash

    result = await EvoShiftRunner(
        audit_config,
        audit_adapter,
        workdir=Path.cwd(),
        initial_memories=state.memories,
        initial_policy=state.policy,
        frozen_audit=True,
        source_run_id=state.source_run_id,
        source_state_hash=state.fingerprint,
        source_dataset_hash=state.source_dataset_hash,
    ).run()

    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((result.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert manifest["run_mode"] == "frozen_audit"
    assert manifest["source_state_hash"] == state.fingerprint
    assert result.metrics["audit"]["frozen"] is True
    assert result.metrics["audit"]["state_unchanged"] is True
    assert result.metrics["promotion_precision"] is None
    assert result.metrics["evolution"]["candidates_evaluated"] == 0
    assert summary["store_counts"]["validations"] == 0
    assert summary["notes"]["state_mutation"] == "disabled"
    assert {(item["memory_id"], item["version"]) for item in summary["active_memories"]} == {
        (item.memory_id, item.version) for item in state.memories
    }


@pytest.mark.asyncio
@pytest.mark.integration
async def test_frozen_audit_rejects_source_stream_reuse(tmp_path: Path) -> None:
    config = load_config(Path("configs/experiments/offline_demo.yaml"))
    config = config.model_copy(
        update={
            "storage": config.storage.model_copy(
                update={"runs_dir": str(tmp_path / "runs"), "cache_enabled": False}
            )
        }
    )
    adapter = create_benchmark(config.benchmark, root=Path.cwd(), seed=config.evaluation.seed)
    source_hash = sample_fingerprint(adapter.load())
    runner = EvoShiftRunner(
        config,
        adapter,
        workdir=Path.cwd(),
        frozen_audit=True,
        source_run_id="source-run",
        source_dataset_hash=source_hash,
    )

    with pytest.raises(ValueError, match="matches the source"):
        await runner.run()
