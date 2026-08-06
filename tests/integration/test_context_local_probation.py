from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from evoshift.benchmarks import create_benchmark
from evoshift.config import load_config
from evoshift.runner import EvoShiftRunner


@pytest.mark.asyncio
@pytest.mark.integration
async def test_context_local_probation_repairs_retrieval_miss_with_paired_control(
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
        > control.metrics["policy_shift"]["first_changed_case_success_rate"]
    )
    evolution = candidate.metrics["evolution"]
    assert evolution["context_probation_interventions"] >= 4
    assert (
        evolution["context_probation_paired_controls"]
        == evolution["context_probation_interventions"]
    )
    assert (
        evolution["context_probation_forced_applications"]
        == evolution["context_probation_interventions"]
    )
    assert candidate.metrics["future_audit"]["confirmed"] >= 1

    connection = sqlite3.connect(candidate.run_dir / "state.sqlite3")
    try:
        events = connection.execute(
            "SELECT payload_json FROM evolution_events "
            "WHERE event_type='context_local_probation_intervention'"
        ).fetchall()
    finally:
        connection.close()
    payloads = [json.loads(row[0]) for row in events]
    assert len(payloads) == evolution["context_probation_interventions"]
    assert all(payload["paired_control"] for payload in payloads)
    assert all(payload["candidate_only"] for payload in payloads)
    assert all(not payload["persistent_state_changed"] for payload in payloads)
