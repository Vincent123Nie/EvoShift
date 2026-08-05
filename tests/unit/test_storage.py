import json
from pathlib import Path

from evoshift.runtime.artifacts import RunArtifacts
from evoshift.schemas import (
    Algorithm,
    BenchmarkSample,
    Episode,
    MemoryItem,
    MemoryStatus,
    RunManifest,
    ScoreBundle,
    SolverOutput,
)
from evoshift.storage import SQLiteStore


def test_store_round_trip(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "state.sqlite3")
    manifest = RunManifest(
        run_id="r1",
        config_hash="abc",
        model="fake",
        algorithm=Algorithm.EVOSHIFT,
        seed=7,
        python_version="3.11",
        platform="test",
    )
    store.save_run(manifest)
    episode = Episode(
        episode_id="e1",
        run_id="r1",
        index=0,
        sample=BenchmarkSample(sample_id="s1", prompt="2+2", reference="4"),
        output=SolverOutput(answer="4"),
        score=ScoreBundle(primary=1.0, success=True),
    )
    store.save_episode(episode)
    memory = MemoryItem(
        memory_id="m1",
        status=MemoryStatus.ACTIVE,
        trigger="addition question",
        directive="Add the operands carefully.",
    )
    store.save_memory(memory)

    assert store.get_run("r1") == manifest
    assert store.list_episodes("r1") == [episode]
    assert store.get_memory("m1") == memory
    assert store.counts("r1") == {
        "episodes": 1,
        "validations": 0,
        "active_memories": 1,
    }
    store.close()


def test_store_keeps_memory_versions(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "state.sqlite3")
    first = MemoryItem(memory_id="m1", trigger="x", directive="old")
    second = first.model_copy(
        update={"version": 2, "status": MemoryStatus.ACTIVE, "directive": "new"}
    )
    store.save_memory(first)
    store.save_memory(second)

    assert store.get_memory("m1", version=1).directive == "old"  # type: ignore[union-attr]
    assert store.get_memory("m1").directive == "new"  # type: ignore[union-attr]
    assert store.list_memories([MemoryStatus.ACTIVE]) == [second]
    store.close()


def test_episode_and_trace_persist_change_point_diagnostics(tmp_path: Path) -> None:
    artifacts = RunArtifacts(tmp_path, "change-point-run")
    episode = Episode(
        episode_id="e-change",
        run_id="change-point-run",
        index=3,
        sample=BenchmarkSample(sample_id="s-change", prompt="case", reference="ALLOW"),
        output=SolverOutput(answer="ALLOW"),
        score=ScoreBundle(primary=1.0, success=True),
        feedback_change_probability=0.84,
        feedback_source_regime=2,
        feedback_grace_observations=5,
    )

    artifacts.append_episode(episode)

    prediction = json.loads((artifacts.run_dir / "predictions.jsonl").read_text(encoding="utf-8"))
    trace = json.loads((artifacts.run_dir / "traces.jsonl").read_text(encoding="utf-8"))
    assert prediction["feedback_change_probability"] == 0.84
    assert prediction["feedback_source_regime"] == 2
    assert prediction["feedback_grace_observations"] == 5
    assert trace["feedback_change_probability"] == 0.84
    assert trace["feedback_source_regime"] == 2
    assert trace["feedback_grace_observations"] == 5
