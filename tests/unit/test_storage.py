import sqlite3
from pathlib import Path

import pytest

from evoshift.runtime.artifacts import RunArtifacts, load_episodes
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


def test_artifacts_round_trip_preserves_change_point_trace(tmp_path: Path) -> None:
    artifacts = RunArtifacts(tmp_path, "run-1")
    episode = Episode(
        episode_id="e1",
        run_id="run-1",
        index=0,
        sample=BenchmarkSample(sample_id="s1", prompt="case", reference="hidden"),
        output=SolverOutput(answer="answer"),
        score=ScoreBundle(primary=1.0, success=True),
        feedback_adaptation_trust=0.65,
        feedback_change_point_probability=0.61,
        feedback_change_point_crossed=True,
        feedback_change_point_run_length=1,
    )

    artifacts.append_episode(episode)

    loaded = load_episodes(artifacts.run_dir / "predictions.jsonl")
    assert loaded == [episode]
    trace = (artifacts.run_dir / "traces.jsonl").read_text(encoding="utf-8")
    assert '"feedback_adaptation_trust": 0.65' in trace
    assert '"feedback_change_point_probability": 0.61' in trace
    assert '"feedback_change_point_crossed": true' in trace
    assert '"feedback_change_point_run_length": 1' in trace


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


def test_atomic_memory_batch_rolls_back_all_items_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteStore(tmp_path / "state.sqlite3")
    first = MemoryItem(memory_id="first", trigger="x", directive="one")
    second = MemoryItem(memory_id="second", trigger="y", directive="two")
    original = store._save_memory_in_transaction

    def fail_after_write(connection: sqlite3.Connection, item: MemoryItem) -> None:
        original(connection, item)
        if item.memory_id == "second":
            raise RuntimeError("injected transaction failure")

    monkeypatch.setattr(store, "_save_memory_in_transaction", fail_after_write)

    with pytest.raises(RuntimeError, match="injected"):
        store.save_memories_atomic([first, second])

    assert store.get_memory("first") is None
    assert store.get_memory("second") is None
    store.close()
