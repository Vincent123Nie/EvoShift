"""Load and fingerprint evolved state for immutable held-out evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evoshift.schemas import MemoryItem, MemoryStatus, PolicyGenome, RunManifest, RunMode


@dataclass(frozen=True)
class EvolvedState:
    """Validated state snapshot exported by a completed prequential run."""

    source_run_id: str
    source_dataset_hash: str
    source_model: str
    policy: PolicyGenome
    memories: tuple[MemoryItem, ...]
    fingerprint: str


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"required source-run artifact is missing: {path}") from None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read source-run artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"source-run artifact must contain a JSON object: {path}")
    return payload


def state_fingerprint(policy: PolicyGenome, memories: tuple[MemoryItem, ...]) -> str:
    """Return a stable SHA-256 over the exact policy and active-memory snapshot."""

    ordered = sorted(memories, key=lambda item: (item.memory_id, item.version))
    payload = {
        "policy": policy.model_dump(mode="json"),
        "active_memories": [item.model_dump(mode="json") for item in ordered],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_evolved_state(run_dir: Path) -> EvolvedState:
    """Validate and load the frozen state exported by ``EvoShiftRunner``."""

    source = Path(run_dir)
    if not source.is_dir():
        raise ValueError(f"source run is not a directory: {source}")
    manifest = RunManifest.model_validate(_read_object(source / "manifest.json"))
    if manifest.finished_at is None:
        raise ValueError("source run is incomplete; finished_at is missing")
    if manifest.run_mode != RunMode.PREQUENTIAL:
        raise ValueError("a frozen audit cannot be used as the source of another audit")

    summary = _read_object(source / "summary.json")
    try:
        policy = PolicyGenome.model_validate(summary["final_policy"])
        raw_memories = summary["active_memories"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("source summary has no valid final policy or memory snapshot") from exc
    if not isinstance(raw_memories, list):
        raise ValueError("source summary active_memories must be a list")
    memories = tuple(MemoryItem.model_validate(item) for item in raw_memories)
    if any(item.status != MemoryStatus.ACTIVE for item in memories):
        raise ValueError("source summary contains a non-active memory in active_memories")
    keys = [(item.memory_id, item.version) for item in memories]
    if len(keys) != len(set(keys)):
        raise ValueError("source summary contains duplicate memory versions")

    return EvolvedState(
        source_run_id=manifest.run_id,
        source_dataset_hash=manifest.dataset_hash,
        source_model=manifest.model,
        policy=policy,
        memories=memories,
        fingerprint=state_fingerprint(policy, memories),
    )


def require_same_model(state: EvolvedState, target_model: str) -> None:
    """Reject model changes that would confound forward-transfer claims."""

    if state.source_model != target_model:
        raise ValueError(
            "frozen audit requires the same model as the source run: "
            f"source={state.source_model!r}, target={target_model!r}"
        )


__all__ = [
    "EvolvedState",
    "load_evolved_state",
    "require_same_model",
    "state_fingerprint",
]
