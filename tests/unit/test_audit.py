import json
from pathlib import Path

import pytest

from evoshift.audit import load_evolved_state, require_same_model, state_fingerprint
from evoshift.schemas import (
    Algorithm,
    MemoryItem,
    MemoryStatus,
    PolicyGenome,
    RunManifest,
)


def _source_run(
    tmp_path: Path,
    *,
    name: str = "source-run",
    memory_status: MemoryStatus = MemoryStatus.ACTIVE,
) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir()
    manifest = RunManifest(
        run_id=name,
        config_hash="config-hash",
        dataset_hash="source-dataset-hash",
        model="model-a",
        algorithm=Algorithm.EVOSHIFT,
        seed=42,
        python_version="3.11",
        platform="test",
        finished_at="2026-08-04T00:00:00Z",
    )
    policy = PolicyGenome(version=3, top_k=2)
    memory = MemoryItem(
        memory_id="memory-1",
        version=2,
        status=memory_status,
        trigger="when a symbolic task changes output format",
        directive="Check the requested output label before answering.",
    )
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "final_policy": policy.model_dump(mode="json"),
                "active_memories": [memory.model_dump(mode="json")],
            },
            default=str,
        ),
        encoding="utf-8",
    )
    return run_dir


def test_load_evolved_state_validates_and_fingerprints_snapshot(tmp_path: Path) -> None:
    state = load_evolved_state(_source_run(tmp_path))

    assert state.source_run_id == "source-run"
    assert state.source_dataset_hash == "source-dataset-hash"
    assert state.policy.version == 3
    assert state.memories[0].memory_id == "memory-1"
    assert state.fingerprint == state_fingerprint(state.policy, state.memories)
    require_same_model(state, "model-a")


def test_audit_state_rejects_non_active_memory_and_model_change(tmp_path: Path) -> None:
    invalid = _source_run(
        tmp_path,
        name="invalid",
        memory_status=MemoryStatus.SHADOW,
    )
    with pytest.raises(ValueError, match="non-active"):
        load_evolved_state(invalid)

    state = load_evolved_state(_source_run(tmp_path, name="valid"))
    with pytest.raises(ValueError, match="same model"):
        require_same_model(state, "model-b")
