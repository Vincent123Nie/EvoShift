from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from evoshift.benchmarks import create_benchmark
from evoshift.benchmarks.base import BenchmarkAdapter
from evoshift.config import load_config
from evoshift.runner import EvoShiftRunner
from evoshift.schemas import BenchmarkSample


class _FixedSamplesAdapter(BenchmarkAdapter):
    def __init__(self, samples: list[BenchmarkSample]) -> None:
        self.samples = samples

    def load(self) -> list[BenchmarkSample]:
        return [sample.model_copy(deep=True) for sample in self.samples]


def _prediction_at(run_dir: Path, episode_index: int) -> dict[str, object]:
    rows = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return next(row for row in rows if row["index"] == episode_index)


def _context_probation_payloads(run_dir: Path) -> list[dict[str, object]]:
    connection = sqlite3.connect(run_dir / "state.sqlite3")
    try:
        events = connection.execute(
            "SELECT payload_json FROM evolution_events "
            "WHERE event_type='context_local_probation_intervention' ORDER BY event_id"
        ).fetchall()
    finally:
        connection.close()
    return [json.loads(row[0]) for row in events]


def _event_payloads(run_dir: Path, event_type: str) -> list[dict[str, object]]:
    connection = sqlite3.connect(run_dir / "state.sqlite3")
    try:
        events = connection.execute(
            "SELECT payload_json FROM evolution_events "
            "WHERE event_type=? ORDER BY event_id",
            (event_type,),
        ).fetchall()
    finally:
        connection.close()
    return [json.loads(row[0]) for row in events]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_context_local_probation_uses_only_historical_feedback_for_intervention(
    tmp_path: Path,
) -> None:
    base = load_config(Path("configs/experiments/policy_shift_hard_demo.yaml"))
    common = {
        "storage": base.storage.model_copy(
            update={"runs_dir": str(tmp_path / "runs"), "cache_enabled": False}
        ),
        "evaluation": base.evaluation.model_copy(update={"seed": 233}),
        "policy": base.policy.model_copy(update={"top_k": 0}),
        "benchmark": base.benchmark.model_copy(
            update={
                "feedback_noise_rate": 0.0,
                "feedback_attack_burst_length": 0,
                "feedback_warmup_attack_burst_length": 2,
            }
        ),
    }
    control_config = base.model_copy(update=common)
    candidate_config = control_config.model_copy(
        update={
            "evolution": control_config.evolution.model_copy(
                update={
                    "context_local_probation_fast_path_enabled": True,
                    "context_local_probation_fast_path_max_age": 12,
                    "context_local_probation_fast_path_max_uses": 2,
                    "context_local_probation_fast_path_min_trust": 0.10,
                }
            )
        }
    )
    control = await EvoShiftRunner(
        control_config,
        create_benchmark(control_config.benchmark, root=Path.cwd(), seed=233),
        workdir=Path.cwd(),
    ).run()
    candidate = await EvoShiftRunner(
        candidate_config,
        create_benchmark(candidate_config.benchmark, root=Path.cwd(), seed=233),
        workdir=Path.cwd(),
    ).run()

    assert (
        candidate.metrics["policy_shift"]["changed_case_success_rate"]
        > control.metrics["policy_shift"]["changed_case_success_rate"]
    )
    assert (
        candidate.metrics["policy_shift"]["first_changed_case_success_rate"]
        == control.metrics["policy_shift"]["first_changed_case_success_rate"]
    )
    evolution = candidate.metrics["evolution"]
    assert evolution["context_probation_interventions"] >= 1
    assert (
        evolution["context_probation_paired_controls"]
        == evolution["context_probation_interventions"]
    )
    assert (
        evolution["context_probation_forced_applications"]
        == evolution["context_probation_interventions"]
    )
    assert candidate.metrics["future_audit"]["confirmed"] >= 1

    payloads = _context_probation_payloads(candidate.run_dir)
    assert len(payloads) == evolution["context_probation_interventions"]
    assert all(payload["paired_control"] for payload in payloads)
    assert all(payload["candidate_only"] for payload in payloads)
    assert all(not payload["persistent_state_changed"] for payload in payloads)
    assert all(
        payload["pre_predict_context_observations"] > payload["registered_context_observations"]
        for payload in payloads
    )

    target_index = int(payloads[0]["episode_index"])
    samples = create_benchmark(
        candidate_config.benchmark,
        root=Path.cwd(),
        seed=233,
    ).load()
    target = samples[target_index]
    original_feedback = str(target.metadata["feedback_reference"])
    assert original_feedback.casefold() != payloads[0]["pre_predict_signal"]
    flipped_feedback = "DENY" if original_feedback == "APPROVE" else "APPROVE"
    metadata = {**target.metadata, "feedback_reference": flipped_feedback}
    samples[target_index] = target.model_copy(update={"metadata": metadata})
    flipped_config = candidate_config.model_copy(
        update={
            "storage": candidate_config.storage.model_copy(
                update={"runs_dir": str(tmp_path / "flipped"), "cache_enabled": False}
            )
        }
    )
    flipped = await EvoShiftRunner(
        flipped_config,
        _FixedSamplesAdapter(samples),
        workdir=Path.cwd(),
    ).run()
    original_prediction = _prediction_at(candidate.run_dir, target_index)
    flipped_prediction = _prediction_at(flipped.run_dir, target_index)
    assert flipped_prediction["output"] == original_prediction["output"]
    assert flipped_prediction["selected_memory_ids"] == original_prediction["selected_memory_ids"]
    flipped_payloads = _context_probation_payloads(flipped.run_dir)
    flipped_target = next(
        payload for payload in flipped_payloads if payload["episode_index"] == target_index
    )
    for key in (
        "context",
        "registered_index",
        "registered_context_observations",
        "pre_predict_context_observations",
        "pre_predict_signal",
        "pre_predict_trust",
        "pre_predict_reason",
    ):
        assert flipped_target[key] == payloads[0][key]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_provisional_lane_is_confirmed_and_additive(tmp_path: Path) -> None:
    base = load_config(Path("configs/experiments/policy_shift_hard_demo.yaml"))
    common = {
        "storage": base.storage.model_copy(
            update={"runs_dir": str(tmp_path / "runs"), "cache_enabled": False}
        ),
        "evaluation": base.evaluation.model_copy(update={"seed": 42}),
        "policy": base.policy.model_copy(update={"top_k": 3}),
    }
    control_config = base.model_copy(update=common)
    lane_config = control_config.model_copy(
        update={
            "evolution": control_config.evolution.model_copy(
                update={
                    "context_local_provisional_lane_enabled": True,
                    "context_local_provisional_lane_min_trust": 0.10,
                }
            )
        }
    )
    control = await EvoShiftRunner(
        control_config,
        create_benchmark(control_config.benchmark, root=Path.cwd(), seed=42),
        workdir=Path.cwd(),
    ).run()
    lane = await EvoShiftRunner(
        lane_config,
        create_benchmark(lane_config.benchmark, root=Path.cwd(), seed=42),
        workdir=Path.cwd(),
    ).run()

    assert lane.metrics["overall"] == control.metrics["overall"]
    assert lane.metrics["policy_shift"] == control.metrics["policy_shift"]
    evolution = lane.metrics["evolution"]
    assert evolution["context_provisional_interventions"] >= 1
    assert evolution["context_provisional_low_trust_interventions"] == 0
    registrations = _event_payloads(
        lane.run_dir,
        "context_local_provisional_registered",
    )
    assert registrations
    assert all(item["pending_observations"] == 0 for item in registrations)
    probes = _event_payloads(lane.run_dir, "context_local_provisional_probe")
    assert probes
    assert any(item["feedback_success"] for item in probes)
