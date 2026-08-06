from __future__ import annotations

import copy
import json
import random
import re
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from evoshift.benchmarks import create_benchmark
from evoshift.benchmarks.base import BenchmarkAdapter
from evoshift.config import load_config
from evoshift.runner import EvoShiftRunner, ExperimentResult
from evoshift.schemas import BenchmarkSample


class _FixedSamplesAdapter(BenchmarkAdapter):
    def __init__(self, samples: list[BenchmarkSample]) -> None:
        self.samples = samples

    def load(self) -> list[BenchmarkSample]:
        return [sample.model_copy(deep=True) for sample in self.samples]


def _with_protected_labels(
    samples: list[BenchmarkSample], labels: list[bool]
) -> list[BenchmarkSample]:
    transformed: list[BenchmarkSample] = []
    for sample, protected in zip(samples, labels):
        metadata = dict(sample.metadata)
        metadata["protected"] = protected
        transformed.append(sample.model_copy(update={"metadata": metadata}))
    return transformed


def _episode_identity_map(database_path: Path) -> dict[str, str]:
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            "SELECT episode_id, episode_index, payload_json FROM episodes ORDER BY episode_index"
        ).fetchall()
    finally:
        connection.close()
    identities: dict[str, str] = {}
    for episode_id, episode_index, payload_json in rows:
        payload = json.loads(payload_json)
        identities[str(episode_id)] = (
            f"episode:{int(episode_index)}:{payload['sample']['sample_id']}"
        )
    return identities


_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)$")


def _canonicalize_online_value(
    value: Any,
    *,
    episode_identities: dict[str, str],
    failure_identities: dict[str, str],
) -> Any:
    if isinstance(value, dict):
        return {
            key: _canonicalize_online_value(
                item,
                episode_identities=episode_identities,
                failure_identities=failure_identities,
            )
            for key, item in sorted(value.items())
        }
    if isinstance(value, list):
        return [
            _canonicalize_online_value(
                item,
                episode_identities=episode_identities,
                failure_identities=failure_identities,
            )
            for item in value
        ]
    if isinstance(value, str):
        if value in episode_identities:
            return episode_identities[value]
        if value.startswith("fail-"):
            return failure_identities.setdefault(value, f"failure:{len(failure_identities)}")
        if _TIMESTAMP.fullmatch(value):
            return "<run-local-timestamp>"
    return value


def _canonical_event_rows(result: ExperimentResult) -> list[tuple[str, str, Any]]:
    database_path = result.run_dir / "state.sqlite3"
    episode_identities = _episode_identity_map(database_path)
    failure_identities: dict[str, str] = {}
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            "SELECT event_type, entity_id, payload_json FROM evolution_events ORDER BY event_id"
        ).fetchall()
    finally:
        connection.close()
    return [
        (
            str(event_type),
            _canonicalize_online_value(
                str(entity_id),
                episode_identities=episode_identities,
                failure_identities=failure_identities,
            ),
            _canonicalize_online_value(
                json.loads(payload_json),
                episode_identities=episode_identities,
                failure_identities=failure_identities,
            ),
        )
        for event_type, entity_id, payload_json in rows
    ]


def _canonical_persistent_state(result: ExperimentResult) -> dict[str, Any]:
    database_path = result.run_dir / "state.sqlite3"
    episode_identities = _episode_identity_map(database_path)
    failure_identities: dict[str, str] = {}
    connection = sqlite3.connect(database_path)
    try:
        memory_rows = connection.execute(
            "SELECT memory_id, version, status, kind, payload_json FROM memory_items "
            "ORDER BY memory_id, version"
        ).fetchall()
        policy_rows = connection.execute(
            "SELECT version, status, parent_version, payload_json FROM policies "
            "ORDER BY version, status"
        ).fetchall()
    finally:
        connection.close()
    return {
        "memories": [
            (
                memory_id,
                version,
                status,
                kind,
                _canonicalize_online_value(
                    json.loads(payload_json),
                    episode_identities=episode_identities,
                    failure_identities=failure_identities,
                ),
            )
            for memory_id, version, status, kind, payload_json in memory_rows
        ],
        "policies": [
            (
                version,
                status,
                parent_version,
                _canonicalize_online_value(
                    json.loads(payload_json),
                    episode_identities=episode_identities,
                    failure_identities=failure_identities,
                ),
            )
            for version, status, parent_version, payload_json in policy_rows
        ],
    }


def _canonical_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    canonical = copy.deepcopy(metrics)
    canonical.pop("config_hash", None)
    canonical.pop("dataset_hash", None)
    policy_shift = canonical["policy_shift"]
    policy_shift.pop("invariant_n", None)
    policy_shift.pop("invariant_retention_rate", None)
    for phase_counts in policy_shift["phase_slice_counts"].values():
        phase_counts.pop("protected", None)
    canonical["audit"].pop("final_state_hash", None)
    return canonical


def _replay_traces(result: ExperimentResult) -> list[tuple[str, tuple[str, ...], tuple[int, ...]]]:
    return [
        (
            decision.result.candidate_id,
            tuple(decision.result.replay_sample_ids),
            tuple(decision.result.replay_episode_indices),
        )
        for decision in result.decisions
        if decision.result.replay_sample_ids
    ]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_hidden_protected_labels_cannot_change_online_evolution(
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/experiments/policy_shift_hard_demo.yaml"))
    base_adapter = create_benchmark(
        config.benchmark,
        root=Path.cwd(),
        seed=config.evaluation.seed,
    )
    samples = base_adapter.load()
    original = [bool(sample.metadata.get("protected")) for sample in samples]
    shuffled = list(original)
    random.Random(20260806).shuffle(shuffled)
    modes = {
        "original": original,
        "all_false": [False] * len(original),
        "complement": [not value for value in original],
        "shuffle": shuffled,
    }

    assert set(original) == {False, True}
    assert sum(shuffled) == sum(original)
    assert shuffled != original

    results: dict[str, ExperimentResult] = {}
    stripped_samples = [
        sample.model_copy(
            update={
                "metadata": {
                    key: value for key, value in sample.metadata.items() if key != "protected"
                }
            }
        ).model_dump(mode="json")
        for sample in samples
    ]
    for mode, labels in modes.items():
        transformed = _with_protected_labels(samples, labels)
        assert [
            sample.model_copy(
                update={
                    "metadata": {
                        key: value for key, value in sample.metadata.items() if key != "protected"
                    }
                }
            ).model_dump(mode="json")
            for sample in transformed
        ] == stripped_samples
        mode_config = config.model_copy(
            update={
                "storage": config.storage.model_copy(
                    update={
                        "runs_dir": str(tmp_path / mode / "runs"),
                        "cache_enabled": False,
                    }
                )
            }
        )
        results[mode] = await EvoShiftRunner(
            mode_config,
            _FixedSamplesAdapter(transformed),
            workdir=Path.cwd(),
        ).run()

    baseline = results["original"]
    baseline_decisions = [decision.model_dump(mode="json") for decision in baseline.decisions]
    baseline_replay = _replay_traces(baseline)
    baseline_events = _canonical_event_rows(baseline)
    baseline_state = _canonical_persistent_state(baseline)
    baseline_metrics = _canonical_metrics(baseline.metrics)

    assert baseline_replay
    assert any(
        decision.result.candidate_type == "memory" and decision.promote
        for decision in baseline.decisions
    )
    assert any(
        decision.result.candidate_type == "memory_future_audit" and not decision.promote
        for decision in baseline.decisions
    )
    assert all(
        not any(decision.result.replay_protected_mask)
        for decision in baseline.decisions
        if decision.result.replay_sample_ids
    )

    for mode, result in results.items():
        assert [decision.model_dump(mode="json") for decision in result.decisions] == (
            baseline_decisions
        ), mode
        assert _replay_traces(result) == baseline_replay, mode
        assert _canonical_event_rows(result) == baseline_events, mode
        assert _canonical_persistent_state(result) == baseline_state, mode
        assert _canonical_metrics(result.metrics) == baseline_metrics, mode
        assert (
            result.metrics["audit"]["semantic_final_state_hash"]
            == baseline.metrics["audit"]["semantic_final_state_hash"]
        ), mode
        assert all(
            payload["configured_protected_observations"] == 0
            and payload["protection_basis"] == "configured_benchmark_protected_phases"
            for event_type, _, payload in _canonical_event_rows(result)
            if event_type == "future_counterfactual_audit"
        ), mode


@pytest.mark.asyncio
@pytest.mark.integration
async def test_rerank_path_cannot_consume_hidden_protected_labels(tmp_path: Path) -> None:
    base = load_config(Path("configs/experiments/policy_shift_hard_demo.yaml"))
    base = base.model_copy(
        update={
            "benchmark": base.benchmark.model_copy(
                update={
                    "feedback_noise_rate": 0.0,
                    "feedback_attack_burst_length": 0,
                    "policy_prompt_style": "paraphrase",
                }
            ),
            "evolution": base.evolution.model_copy(update={"conflict_supersession_enabled": False}),
            "policy": base.policy.model_copy(
                update={
                    "top_k": 1,
                    "llm_rerank_enabled": True,
                    "llm_rerank_candidate_k": 4,
                }
            ),
        }
    )
    samples = create_benchmark(
        base.benchmark,
        root=Path.cwd(),
        seed=base.evaluation.seed,
    ).load()
    protected = [bool(sample.metadata.get("protected")) for sample in samples]
    modes = {
        "original": protected,
        "complement": [not value for value in protected],
    }
    results: dict[str, ExperimentResult] = {}
    for mode, labels in modes.items():
        config = base.model_copy(
            update={
                "storage": base.storage.model_copy(
                    update={
                        "runs_dir": str(tmp_path / mode / "runs"),
                        "cache_enabled": False,
                    }
                )
            }
        )
        results[mode] = await EvoShiftRunner(
            config,
            _FixedSamplesAdapter(_with_protected_labels(samples, labels)),
            workdir=Path.cwd(),
        ).run()

    baseline = results["original"]
    assert baseline.metrics["retrieval"]["rerank_attempts"] > 0
    baseline_decisions = [decision.model_dump(mode="json") for decision in baseline.decisions]
    baseline_events = _canonical_event_rows(baseline)
    baseline_state = _canonical_persistent_state(baseline)
    baseline_metrics = _canonical_metrics(baseline.metrics)
    for mode, result in results.items():
        assert [decision.model_dump(mode="json") for decision in result.decisions] == (
            baseline_decisions
        ), mode
        assert _canonical_event_rows(result) == baseline_events, mode
        assert _canonical_persistent_state(result) == baseline_state, mode
        assert _canonical_metrics(result.metrics) == baseline_metrics, mode
        assert (
            result.metrics["audit"]["semantic_final_state_hash"]
            == baseline.metrics["audit"]["semantic_final_state_hash"]
        ), mode
