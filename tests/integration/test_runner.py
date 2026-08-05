import json
import sqlite3
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
@pytest.mark.parametrize(
    ("noise_rate", "attack_rate", "expected_quarantined"),
    [(0.0, 0.0, 0), (0.10, 0.10, 8)],
)
async def test_policy_shift_evolution_adapts_and_quarantines_untrusted_feedback(
    tmp_path: Path,
    noise_rate: float,
    attack_rate: float,
    expected_quarantined: int,
) -> None:
    config = load_config(Path("configs/experiments/policy_shift_demo.yaml"))
    config = config.model_copy(
        update={
            "storage": config.storage.model_copy(
                update={"runs_dir": str(tmp_path / "runs"), "cache_enabled": False}
            ),
            "benchmark": config.benchmark.model_copy(
                update={
                    "feedback_noise_rate": noise_rate,
                    "feedback_attack_rate": attack_rate,
                }
            ),
        }
    )
    adapter = create_benchmark(config.benchmark, root=Path.cwd(), seed=config.evaluation.seed)

    result = await EvoShiftRunner(config, adapter, workdir=Path.cwd()).run()

    assert result.metrics["overall"]["mean_score"] == pytest.approx(0.9722222222)
    assert result.metrics["policy_shift"]["changed_case_success_rate"] == pytest.approx(
        0.8571428571
    )
    assert result.metrics["policy_shift"]["old_rule_leakage_rate"] == pytest.approx(0.1428571429)
    assert result.metrics["policy_shift"]["invariant_retention_rate"] == 1.0
    assert result.metrics["policy_shift"]["future_change_case_success_rate"] == 1.0
    assert result.metrics["feedback"]["clean_feedback_quarantine_rate"] == 0.0
    assert result.metrics["feedback"]["quarantined_feedback_n"] == expected_quarantined
    assert result.metrics["evolution"]["feedback_quarantined"] == expected_quarantined
    assert result.metrics["evolution"]["candidate_replay_attempts"] == 2
    assert result.metrics["evolution"]["memory_candidates_evaluated"] == 2
    assert result.metrics["evolution"]["memory_candidates_promoted"] == 2
    assert result.metrics["evolution"]["policy_candidates_evaluated"] == 0
    assert result.metrics["evolution"]["policy_candidates_promoted"] == 0
    assert result.metrics["evolution"]["policy_evolution_suppressed_by_memory"] == 2
    assert result.metrics["evolution"]["shift_detection_events"] == 2
    assert result.metrics["evolution"]["domains_with_detected_shift"] == 1
    if expected_quarantined:
        assert result.metrics["feedback"]["corrupted_feedback_quarantine_rate"] == 1.0
        assert result.metrics["policy_shift"]["attack_feedback_follow_rate"] == 0.0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_policy_shift_hard_rolls_back_poison_and_relearns_real_change(
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/experiments/policy_shift_hard_demo.yaml"))
    config = config.model_copy(
        update={
            "storage": config.storage.model_copy(
                update={"runs_dir": str(tmp_path / "runs"), "cache_enabled": False}
            )
        }
    )
    adapter = create_benchmark(config.benchmark, root=Path.cwd(), seed=config.evaluation.seed)

    result = await EvoShiftRunner(config, adapter, workdir=Path.cwd()).run()

    future = result.metrics["future_audit"]
    policy_shift = result.metrics["policy_shift"]
    assert future["registered"] == 3
    assert future["completed"] == 3
    assert future["confirmed"] == 2
    assert future["rolled_back"] == 1
    assert future["harmful_promotion_rate"] == pytest.approx(1.0 / 3.0)
    assert future["false_rollback_rate"] == 0.0
    assert future["mean_rollback_observations"] == 1.0
    assert result.metrics["promotion_precision_basis"] == "future_counterfactual"
    assert result.metrics["replay_estimated_promotion_precision"] == 1.0
    assert result.metrics["realized_promotion_precision"] == pytest.approx(2.0 / 3.0)
    assert result.metrics["promotion_precision"] == pytest.approx(2.0 / 3.0)
    evolution = result.metrics["evolution"]
    assert (
        evolution["candidates_promoted"] + evolution["candidates_rejected"]
        == evolution["candidates_evaluated"]
    )
    assert evolution["memory_candidates_evaluated"] == 6
    assert evolution["memory_replay_gates_passed"] == 3
    assert evolution["memory_candidates_promoted"] == 2
    assert evolution["memory_candidates_rejected"] == 4
    assert policy_shift["premature_update_rate"] == 0.125
    assert policy_shift["poison_persistence_error_rate"] == pytest.approx(2.0 / 7.0)
    assert policy_shift["invariant_retention_rate"] == 1.0

    audits = [
        decision
        for decision in result.decisions
        if decision.result.candidate_type == "memory_future_audit"
    ]
    harmful = [decision for decision in audits if not decision.promote]
    confirmed = [decision for decision in audits if decision.promote]
    assert len(harmful) == 1
    assert harmful[0].reason.startswith("early rollback")
    rejected_memory_id = harmful[0].result.candidate_id.split("@", maxsplit=1)[0]
    assert any(
        decision.result.candidate_id.startswith(f"{rejected_memory_id}@") for decision in confirmed
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_causal_memory_governance_forgets_stale_and_reacquires_recurring_rule(
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/experiments/policy_shift_causal_memory_demo.yaml"))
    config = config.model_copy(
        update={
            "storage": config.storage.model_copy(
                update={"runs_dir": str(tmp_path / "runs"), "cache_enabled": False}
            )
        }
    )
    baseline_config = config.model_copy(
        update={"evolution": config.evolution.model_copy(update={"active_audit_enabled": False})}
    )
    causal_adapter = create_benchmark(
        config.benchmark,
        root=Path.cwd(),
        seed=config.evaluation.seed,
    )
    baseline_adapter = create_benchmark(
        baseline_config.benchmark,
        root=Path.cwd(),
        seed=baseline_config.evaluation.seed,
    )

    causal = await EvoShiftRunner(config, causal_adapter, workdir=Path.cwd()).run()
    baseline = await EvoShiftRunner(
        baseline_config,
        baseline_adapter,
        workdir=Path.cwd(),
    ).run()

    governance = causal.metrics["active_memory_governance"]
    baseline_governance = baseline.metrics["active_memory_governance"]
    assert governance["causal_retirements"] >= 1
    assert governance["selective_forgetting_precision"] == 1.0
    assert (
        governance["selective_forgetting_recall"]
        > baseline_governance["selective_forgetting_recall"]
    )
    assert governance["false_retirement_rate"] == 0.0
    assert governance["early_causal_retirements"] >= 1
    assert governance["early_causal_retirement_precision"] == 1.0
    assert governance["early_causal_false_retirement_rate"] == 0.0
    assert (
        governance["harmful_active_memory_exposure_n"]
        < baseline_governance["harmful_active_memory_exposure_n"]
    )
    assert (
        governance["stale_memory_retention_rate"]
        < baseline_governance["stale_memory_retention_rate"]
    )
    assert governance["counterfactual_audit_coverage"] == 1.0
    assert governance["control_requests"] <= (
        causal.metrics["n_episodes"] * config.evolution.active_audit_max_per_episode
    )
    assert governance["reacquisitions"] >= 1
    assert governance["correct_reacquisition_rate"] == 1.0
    assert governance["reactivations"] >= 1
    assert governance["correct_reactivation_rate"] == 1.0
    assert (
        causal.metrics["policy_shift"]["invariant_retention_rate"]
        >= baseline.metrics["policy_shift"]["invariant_retention_rate"]
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_recurrence_circuit_breaker_improves_score_and_cancels_noise_safely(
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/experiments/policy_shift_causal_memory_demo.yaml"))
    config = config.model_copy(
        update={
            "storage": config.storage.model_copy(
                update={"runs_dir": str(tmp_path / "runs"), "cache_enabled": False}
            ),
            "evaluation": config.evaluation.model_copy(update={"seed": 11}),
            "benchmark": config.benchmark.model_copy(
                update={"feedback_noise_rate": 0.0, "feedback_attack_burst_length": 0}
            ),
        }
    )
    current_config = config.model_copy(
        update={
            "evolution": config.evolution.model_copy(
                update={"active_audit_circuit_breaker_enabled": False}
            )
        }
    )
    circuit_config = config.model_copy(
        update={
            "evolution": config.evolution.model_copy(
                update={"active_audit_circuit_breaker_enabled": True}
            )
        }
    )
    current = await EvoShiftRunner(
        current_config,
        create_benchmark(current_config.benchmark, root=Path.cwd(), seed=11),
        workdir=Path.cwd(),
    ).run()
    circuit = await EvoShiftRunner(
        circuit_config,
        create_benchmark(circuit_config.benchmark, root=Path.cwd(), seed=11),
        workdir=Path.cwd(),
    ).run()

    governance = circuit.metrics["active_memory_governance"]
    circuit_metrics = governance["circuit_breaker"]
    assert circuit.metrics["overall"]["mean_score"] > current.metrics["overall"]["mean_score"]
    assert (
        circuit.metrics["policy_shift"]["changed_case_success_rate"]
        > current.metrics["policy_shift"]["changed_case_success_rate"]
    )
    assert circuit_metrics["confirmations"] >= 2
    assert circuit_metrics["confirmation_precision"] == 1.0
    assert circuit_metrics["unconfirmed_persistent_transitions"] == 0
    assert governance["false_retirement_rate"] == 0.0

    noisy_config = circuit_config.model_copy(
        update={
            "benchmark": circuit_config.benchmark.model_copy(update={"feedback_noise_rate": 0.10})
        }
    )
    noisy = await EvoShiftRunner(
        noisy_config,
        create_benchmark(noisy_config.benchmark, root=Path.cwd(), seed=11),
        workdir=Path.cwd(),
    ).run()
    noisy_governance = noisy.metrics["active_memory_governance"]
    noisy_circuit = noisy_governance["circuit_breaker"]
    assert noisy_circuit["cancellations"] >= 1
    assert noisy_circuit["unconfirmed_persistent_transitions"] == 0
    assert noisy_governance["false_retirement_rate"] == 0.0
    assert noisy.metrics["policy_shift"]["invariant_retention_rate"] == 1.0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_cooldown_dormant_revival_adds_recurrent_gain_without_false_revival(
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/experiments/policy_shift_causal_memory_demo.yaml"))
    config = config.model_copy(
        update={
            "storage": config.storage.model_copy(
                update={"runs_dir": str(tmp_path / "runs"), "cache_enabled": False}
            ),
            "evaluation": config.evaluation.model_copy(update={"seed": 11}),
            "benchmark": config.benchmark.model_copy(
                update={"feedback_noise_rate": 0.0, "feedback_attack_burst_length": 0}
            ),
            "evolution": config.evolution.model_copy(
                update={"active_audit_circuit_breaker_enabled": True}
            ),
        }
    )
    combined_config = config.model_copy(
        update={"evolution": config.evolution.model_copy(update={"dormant_revival_enabled": True})}
    )
    recurrence = await EvoShiftRunner(
        config,
        create_benchmark(config.benchmark, root=Path.cwd(), seed=11),
        workdir=Path.cwd(),
    ).run()
    combined = await EvoShiftRunner(
        combined_config,
        create_benchmark(combined_config.benchmark, root=Path.cwd(), seed=11),
        workdir=Path.cwd(),
    ).run()

    revival = combined.metrics["active_memory_governance"]["dormant_revival"]
    assert combined.metrics["overall"]["mean_score"] > recurrence.metrics["overall"]["mean_score"]
    assert revival["confirmations"] == 1
    assert revival["confirmation_precision"] == 1.0
    assert revival["unconfirmed_persistent_transitions"] == 0
    assert combined.metrics["policy_shift"]["invariant_retention_rate"] == 1.0

    noisy_recurrence_config = config.model_copy(
        update={
            "evaluation": config.evaluation.model_copy(update={"seed": 33}),
            "benchmark": config.benchmark.model_copy(update={"feedback_noise_rate": 0.10}),
        }
    )
    noisy_combined_config = noisy_recurrence_config.model_copy(
        update={
            "evolution": noisy_recurrence_config.evolution.model_copy(
                update={"dormant_revival_enabled": True}
            )
        }
    )
    noisy_recurrence = await EvoShiftRunner(
        noisy_recurrence_config,
        create_benchmark(noisy_recurrence_config.benchmark, root=Path.cwd(), seed=33),
        workdir=Path.cwd(),
    ).run()
    noisy_combined = await EvoShiftRunner(
        noisy_combined_config,
        create_benchmark(noisy_combined_config.benchmark, root=Path.cwd(), seed=33),
        workdir=Path.cwd(),
    ).run()

    noisy_revival = noisy_combined.metrics["active_memory_governance"]["dormant_revival"]
    assert (
        noisy_combined.metrics["overall"]["mean_score"]
        > noisy_recurrence.metrics["overall"]["mean_score"]
    )
    assert noisy_revival["confirmation_precision"] == 1.0
    assert noisy_revival["false_confirmation_rate"] == 0.0
    assert noisy_combined.metrics["policy_shift"]["invariant_retention_rate"] == 1.0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_lineage_counterfactual_control_removes_coarse_memory_off_loss(
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/experiments/policy_shift_causal_memory_demo.yaml"))
    config = config.model_copy(
        update={
            "storage": config.storage.model_copy(
                update={"runs_dir": str(tmp_path / "runs"), "cache_enabled": False}
            ),
            "evaluation": config.evaluation.model_copy(update={"seed": 111}),
            "benchmark": config.benchmark.model_copy(
                update={"feedback_noise_rate": 0.10, "feedback_attack_burst_length": 0}
            ),
            "evolution": config.evolution.model_copy(
                update={
                    "active_audit_circuit_breaker_enabled": True,
                    "dormant_revival_enabled": True,
                    "dormant_revival_min_retired_age": 15,
                }
            ),
        }
    )
    lineage_config = config.model_copy(
        update={
            "evolution": config.evolution.model_copy(
                update={"active_audit_lineage_control_enabled": True}
            )
        }
    )

    memory_off = await EvoShiftRunner(
        config,
        create_benchmark(config.benchmark, root=Path.cwd(), seed=111),
        workdir=Path.cwd(),
    ).run()
    lineage = await EvoShiftRunner(
        lineage_config,
        create_benchmark(lineage_config.benchmark, root=Path.cwd(), seed=111),
        workdir=Path.cwd(),
    ).run()

    circuit = lineage.metrics["active_memory_governance"]["circuit_breaker"]
    assert lineage.metrics["overall"]["mean_score"] > memory_off.metrics["overall"]["mean_score"]
    assert circuit["lineage_probes"] >= 1
    assert circuit["lineage_registrations"] >= 1
    assert circuit["lineage_interventions"] >= 1
    assert circuit["confirmation_precision"] in {None, 1.0}
    assert circuit["unconfirmed_persistent_transitions"] == 0
    assert lineage.metrics["policy_shift"]["invariant_retention_rate"] == 1.0

    connection = sqlite3.connect(lineage.run_dir / "state.sqlite3")
    try:
        intervention_payloads = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT payload_json FROM evolution_events "
                "WHERE event_type='causal_circuit_breaker_intervention'"
            )
        ]
    finally:
        connection.close()
    lineage_interventions = [
        payload
        for payload in intervention_payloads
        if payload.get("control_memory_key") is not None
    ]
    assert lineage_interventions
    assert all(
        payload["action"] == "temporarily_replace_with_exact_direct_predecessor"
        for payload in lineage_interventions
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_temporally_diverse_recurrence_blocks_adjacent_noise_failure_chain(
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/experiments/policy_shift_causal_memory_demo.yaml"))
    lineage_config = config.model_copy(
        update={
            "storage": config.storage.model_copy(
                update={"runs_dir": str(tmp_path / "runs"), "cache_enabled": False}
            ),
            "evaluation": config.evaluation.model_copy(update={"seed": 122}),
            "benchmark": config.benchmark.model_copy(
                update={"feedback_noise_rate": 0.10, "feedback_attack_burst_length": 0}
            ),
            "evolution": config.evolution.model_copy(
                update={
                    "active_audit_circuit_breaker_enabled": True,
                    "active_audit_lineage_control_enabled": True,
                    "dormant_revival_enabled": True,
                    "dormant_revival_min_retired_age": 15,
                }
            ),
        }
    )
    chain_safe_config = lineage_config.model_copy(
        update={
            "evolution": lineage_config.evolution.model_copy(
                update={
                    "dynamic_feedback_change_min_span": 2,
                    "dormant_revival_status_index_enabled": True,
                }
            )
        }
    )

    lineage = await EvoShiftRunner(
        lineage_config,
        create_benchmark(lineage_config.benchmark, root=Path.cwd(), seed=122),
        workdir=Path.cwd(),
    ).run()
    chain_safe = await EvoShiftRunner(
        chain_safe_config,
        create_benchmark(chain_safe_config.benchmark, root=Path.cwd(), seed=122),
        workdir=Path.cwd(),
    ).run()

    governance = chain_safe.metrics["active_memory_governance"]
    circuit = governance["circuit_breaker"]
    revival = governance["dormant_revival"]
    assert chain_safe.metrics["overall"]["mean_score"] > lineage.metrics["overall"]["mean_score"]
    assert chain_safe.metrics["feedback_trust_model"]["temporally_deferred_changes"] >= 1
    assert chain_safe.metrics["policy_shift"]["invariant_retention_rate"] == 1.0
    assert governance["false_retirement_rate"] == 0.0
    assert circuit["confirmation_precision"] == 1.0
    assert revival["false_confirmation_rate"] == 0.0

    connection = sqlite3.connect(chain_safe.run_dir / "state.sqlite3")
    try:
        quarantined_reasons = [
            json.loads(row[0])["reason"]
            for row in connection.execute(
                "SELECT payload_json FROM evolution_events WHERE event_type='feedback_quarantined'"
            )
        ]
    finally:
        connection.close()
    assert "dynamic_pending_change_span" in quarantined_reasons


@pytest.mark.asyncio
@pytest.mark.integration
async def test_reactivation_grace_blocks_destructive_noise_without_slowing_clean_changes(
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/experiments/policy_shift_causal_memory_demo.yaml"))
    unguarded_config = config.model_copy(
        update={
            "storage": config.storage.model_copy(
                update={"runs_dir": str(tmp_path / "runs"), "cache_enabled": False}
            ),
            "evaluation": config.evaluation.model_copy(update={"seed": 177}),
            "benchmark": config.benchmark.model_copy(
                update={"feedback_noise_rate": 0.10, "feedback_attack_burst_length": 0}
            ),
            "evolution": config.evolution.model_copy(
                update={
                    "active_audit_circuit_breaker_enabled": True,
                    "active_audit_lineage_control_enabled": True,
                    "dormant_revival_enabled": True,
                    "dormant_revival_min_retired_age": 15,
                    "dormant_revival_status_index_enabled": True,
                    "dynamic_feedback_change_min_span": 0,
                }
            ),
        }
    )
    guarded_config = unguarded_config.model_copy(
        update={
            "evolution": unguarded_config.evolution.model_copy(
                update={"active_audit_reactivation_grace_episodes": 8}
            )
        }
    )

    unguarded = await EvoShiftRunner(
        unguarded_config,
        create_benchmark(unguarded_config.benchmark, root=Path.cwd(), seed=177),
        workdir=Path.cwd(),
    ).run()
    guarded = await EvoShiftRunner(
        guarded_config,
        create_benchmark(guarded_config.benchmark, root=Path.cwd(), seed=177),
        workdir=Path.cwd(),
    ).run()

    unguarded_governance = unguarded.metrics["active_memory_governance"]
    guarded_governance = guarded.metrics["active_memory_governance"]
    assert guarded.metrics["overall"]["mean_score"] > unguarded.metrics["overall"]["mean_score"]
    assert guarded_governance["false_retirement_rate"] == 0.0
    assert unguarded_governance["false_retirement_rate"] > 0.0
    assert guarded_governance["reactivation_grace_audit_suppressions"] >= 1
    assert guarded.metrics["policy_shift"]["invariant_retention_rate"] == 1.0
    assert guarded_governance["circuit_breaker"]["confirmation_precision"] == 1.0
    assert guarded_governance["dormant_revival"]["confirmation_precision"] == 1.0

    connection = sqlite3.connect(guarded.run_dir / "state.sqlite3")
    try:
        suppressions = [
            (entity_id, json.loads(payload))
            for entity_id, payload in connection.execute(
                "SELECT entity_id, payload_json FROM evolution_events "
                "WHERE event_type='active_memory_audit_suppressed'"
            )
        ]
    finally:
        connection.close()
    assert suppressions
    assert all("@v" in entity_id for entity_id, _ in suppressions)
    assert all(payload["reactivation_age"] <= 8 for _, payload in suppressions)
    assert all(
        payload["online_decision_uses"] == "learner_visible_memory_lifecycle_only"
        for _, payload in suppressions
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_quarantined_feedback_can_only_reach_memory_through_verified_shadow_lane(
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/experiments/policy_shift_causal_memory_demo.yaml"))
    config = config.model_copy(
        update={
            "storage": config.storage.model_copy(
                update={"runs_dir": str(tmp_path / "runs"), "cache_enabled": False}
            ),
            "evaluation": config.evaluation.model_copy(update={"seed": 11}),
            "evolution": config.evolution.model_copy(
                update={
                    "shadow_candidate_enabled": True,
                    "min_feedback_trust_for_shadow_candidate": 0.10,
                    "candidate_min_trusted_observations": 0,
                    "shadow_eprocess_enabled": True,
                    "shadow_eprocess_null_match_probability": 0.25,
                    "shadow_eprocess_alternative_match_probability": 0.75,
                    "shadow_eprocess_alpha": 0.05,
                }
            ),
        }
    )
    adapter = create_benchmark(config.benchmark, root=Path.cwd(), seed=11)

    result = await EvoShiftRunner(config, adapter, workdir=Path.cwd()).run()

    evolution = result.metrics["evolution"]
    assert evolution["shadow_failure_extractions"] > 0
    assert evolution["shadow_candidate_observations"] > 0
    assert evolution["shadow_candidate_replay_attempts"] > 0
    assert evolution["shadow_candidate_probations"] > 0
    assert evolution["shadow_eprocess_opportunities"] > 0
    assert evolution["shadow_eprocess_crossings"] > 0
    assert evolution["trusted_candidate_shadow_cooldown_bypasses"] >= 0
    assert evolution["shadow_candidate_activations"] <= result.metrics["future_audit"]["confirmed"]

    connection = sqlite3.connect(result.run_dir / "state.sqlite3")
    try:
        events = connection.execute(
            "SELECT event_type, entity_id, payload_json FROM evolution_events "
            "WHERE event_type IN "
            "('memory_probation_started', 'future_counterfactual_audit', "
            "'memory_promoted_unverified')"
        ).fetchall()
    finally:
        connection.close()
    assert all(event_type != "memory_promoted_unverified" for event_type, _, _ in events)
    probation = {
        entity_id: json.loads(payload)
        for event_type, entity_id, payload in events
        if event_type == "memory_probation_started"
    }
    future = {
        entity_id: json.loads(payload)
        for event_type, entity_id, payload in events
        if event_type == "future_counterfactual_audit"
    }
    assert probation
    assert set(probation) <= set(future)
    assert all(payload["shadow_derived"] is True for payload in probation.values())
    assert all(payload["candidate_trust"]["shadow_observations"] > 0 for payload in future.values())
    assert all(
        0.0
        <= payload["candidate_trust"]["min"]
        <= payload["candidate_trust"]["mean"]
        <= payload["candidate_trust"]["max"]
        <= 1.0
        for payload in future.values()
    )


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
    assert result.metrics["promotion_precision_basis"] == "not_applicable"
    assert result.metrics["replay_estimated_promotion_precision"] is None
    assert result.metrics["realized_promotion_precision"] is None
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
