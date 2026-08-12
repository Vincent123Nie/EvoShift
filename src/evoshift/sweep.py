from __future__ import annotations

import csv
import hashlib
import itertools
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import yaml

from evoshift.benchmarks import create_benchmark
from evoshift.config import load_config
from evoshift.evaluation import paired_cluster_bootstrap_ci
from evoshift.runner import EvoShiftRunner
from evoshift.runtime.artifacts import load_episodes
from evoshift.schemas import Algorithm, Episode

_AGGREGATE_FIELDS = {
    "mean_score": "score",
    "auac": "auac",
    "cumulative_regret": "cumulative_regret",
    "changed_case_success_rate": "changed_case_success",
    "first_changed_case_success_rate": "first_changed_case_success",
    "old_rule_leakage_rate": "old_rule_leakage",
    "invariant_retention_rate": "invariant_retention",
    "future_change_case_success_rate": "future_change_success",
    "premature_update_rate": "premature_update",
    "corrupted_feedback_follow_rate": "corrupted_feedback_follow",
    "attack_feedback_follow_rate": "attack_feedback_follow",
    "premature_attack_follow_rate": "premature_attack_follow",
    "poison_persistence_error_rate": "poison_persistence_error",
    "corrupted_feedback_quarantine_rate": "corrupted_feedback_quarantine",
    "clean_feedback_quarantine_rate": "clean_feedback_quarantine",
    "mean_recovery_steps": "mean_recovery_steps",
    "unrecovered_shifts": "unrecovered_shifts",
    "shift_detection_events": "shift_detection_events",
    "memory_candidates_promoted": "memory_candidates_promoted",
    "shadow_failure_extractions": "shadow_failure_extractions",
    "shadow_candidate_observations": "shadow_candidate_observations",
    "shadow_candidate_replay_attempts": "shadow_candidate_replay_attempts",
    "shadow_only_candidate_replay_attempts": "shadow_only_candidate_replay_attempts",
    "shadow_candidate_probations": "shadow_candidate_probations",
    "shadow_candidate_activations": "shadow_candidate_activations",
    "shadow_candidate_rejections": "shadow_candidate_rejections",
    "shadow_candidate_expirations": "shadow_candidate_expirations",
    "shadow_eprocess_opportunities": "shadow_eprocess_opportunities",
    "shadow_eprocess_crossings": "shadow_eprocess_crossings",
    "shadow_cluster_eprocess_opportunities": "shadow_cluster_eprocess_opportunities",
    "shadow_cluster_eprocess_crossings": "shadow_cluster_eprocess_crossings",
    "shadow_cluster_eprocess_skipped_unrelated": ("shadow_cluster_eprocess_skipped_unrelated"),
    "trusted_candidate_shadow_cooldown_bypasses": ("trusted_candidate_shadow_cooldown_bypasses"),
    "context_probation_interventions": "context_probation_interventions",
    "context_probation_paired_controls": "context_probation_paired_controls",
    "context_probation_forced_applications": "context_probation_forced_applications",
    "context_probation_expired": "context_probation_expired",
    "context_probation_retrieval_hit_bypasses": ("context_probation_retrieval_hit_bypasses"),
    "rerank_attempts": "rerank_attempts",
    "rerank_applied": "rerank_applied",
    "rerank_fallbacks": "rerank_fallbacks",
    "rerank_fallback_rate": "rerank_fallback_rate",
    "promotion_precision": "promotion_precision",
    "replay_estimated_promotion_precision": "replay_estimated_promotion_precision",
    "realized_promotion_precision": "realized_promotion_precision",
    "future_audit_confirmed": "future_audit_confirmed",
    "future_audit_rolled_back": "future_audit_rolled_back",
    "future_audit_expired": "future_audit_expired",
    "future_audit_realized_promotion_coverage": "future_audit_realized_promotion_coverage",
    "future_audit_harmful_promotion_rate": "future_audit_harmful_promotion_rate",
    "future_audit_false_rollback_rate": "future_audit_false_rollback_rate",
    "future_audit_mean_oracle_delta": "future_audit_mean_oracle_delta",
    "future_audit_mean_audit_latency": "future_audit_mean_audit_latency",
    "future_audit_mean_confirmation_latency": "future_audit_mean_confirmation_latency",
    "future_audit_mean_rollback_latency": "future_audit_mean_rollback_latency",
    "future_audit_mean_harmful_exposure_latency": ("future_audit_mean_harmful_exposure_latency"),
    "future_audit_mean_confirmation_observations": ("future_audit_mean_confirmation_observations"),
    "future_audit_mean_rollback_observations": "future_audit_mean_rollback_observations",
    "future_audit_mean_harmful_exposure_observations": (
        "future_audit_mean_harmful_exposure_observations"
    ),
    "memory_supersessions": "memory_supersessions",
    "memory_reactivations": "memory_reactivations",
    "correct_reactivation_rate": "correct_reactivation",
    "harmful_active_memory_exposure_n": "harmful_active_memory_exposure",
    "stale_memory_retention_rate": "stale_memory_retention",
    "selective_forgetting_precision": "selective_forgetting_precision",
    "selective_forgetting_recall": "selective_forgetting_recall",
    "false_retirement_rate": "false_retirement",
    "correct_reacquisition_rate": "correct_reacquisition",
    "memory_reacquisitions": "memory_reacquisitions",
    "active_audit_control_requests": "active_audit_control_requests",
    "reactivation_grace_audit_suppressions": "reactivation_grace_audit_suppressions",
    "reactivation_grace_protected_rollbacks": "reactivation_grace_protected_rollbacks",
    "counterfactual_audit_coverage": "counterfactual_audit_coverage",
    "active_audit_budget_utilization": "active_audit_budget_utilization",
    "active_audit_mean_retirement_latency": "active_audit_mean_retirement_latency",
    "early_causal_retirements": "early_causal_retirements",
    "early_causal_retirement_precision": "early_causal_retirement_precision",
    "early_causal_false_retirement_rate": "early_causal_false_retirement_rate",
    "circuit_registrations": "circuit_registrations",
    "circuit_lineage_probes": "circuit_lineage_probes",
    "circuit_lineage_registrations": "circuit_lineage_registrations",
    "circuit_interventions": "circuit_interventions",
    "circuit_lineage_interventions": "circuit_lineage_interventions",
    "circuit_confirmations": "circuit_confirmations",
    "circuit_cancellations": "circuit_cancellations",
    "circuit_expirations": "circuit_expirations",
    "circuit_confirmation_precision": "circuit_confirmation_precision",
    "circuit_false_confirmation_rate": "circuit_false_confirmation_rate",
    "circuit_unconfirmed_persistent_transitions": ("circuit_unconfirmed_persistent_transitions"),
    "circuit_mean_oracle_intervention_delta": "circuit_mean_oracle_intervention_delta",
    "circuit_control_requests": "circuit_control_requests",
    "retirement_probation_registrations": "retirement_probation_registrations",
    "retirement_probation_interventions": "retirement_probation_interventions",
    "retirement_probation_confirmations": "retirement_probation_confirmations",
    "retirement_probation_cancellations": "retirement_probation_cancellations",
    "retirement_probation_deferrals": "retirement_probation_deferrals",
    "retirement_probation_expirations": "retirement_probation_expirations",
    "retirement_probation_registration_precision": ("retirement_probation_registration_precision"),
    "retirement_probation_registration_label_coverage": (
        "retirement_probation_registration_label_coverage"
    ),
    "retirement_probation_unknown_tag_registrations": (
        "retirement_probation_unknown_tag_registrations"
    ),
    "retirement_probation_paired_oracle_registration_precision": (
        "retirement_probation_paired_oracle_registration_precision"
    ),
    "retirement_probation_paired_oracle_registration_coverage": (
        "retirement_probation_paired_oracle_registration_coverage"
    ),
    "retirement_probation_confirmation_precision": ("retirement_probation_confirmation_precision"),
    "retirement_probation_confirmation_label_coverage": (
        "retirement_probation_confirmation_label_coverage"
    ),
    "retirement_probation_unknown_tag_confirmations": (
        "retirement_probation_unknown_tag_confirmations"
    ),
    "retirement_probation_confirmation_coverage": ("retirement_probation_confirmation_coverage"),
    "retirement_probation_expiry_rate": "retirement_probation_expiry_rate",
    "retirement_probation_provisional_valid_tag_exposure_n": (
        "retirement_probation_provisional_valid_tag_exposure"
    ),
    "retirement_probation_provisional_valid_tag_failure_episodes": (
        "retirement_probation_provisional_valid_tag_failures"
    ),
    "retirement_probation_false_confirmation_rate": (
        "retirement_probation_false_confirmation_rate"
    ),
    "retirement_probation_unconfirmed_persistent_transitions": (
        "retirement_probation_unconfirmed_persistent_transitions"
    ),
    "retirement_probation_mean_oracle_intervention_delta": (
        "retirement_probation_mean_oracle_intervention_delta"
    ),
    "retirement_probation_control_requests": "retirement_probation_control_requests",
    "revival_registrations": "revival_registrations",
    "revival_interventions": "revival_interventions",
    "revival_confirmations": "revival_confirmations",
    "revival_cancellations": "revival_cancellations",
    "revival_expirations": "revival_expirations",
    "revival_confirmation_precision": "revival_confirmation_precision",
    "revival_confirmation_coverage": "revival_confirmation_coverage",
    "revival_false_confirmation_rate": "revival_false_confirmation_rate",
    "revival_context_mismatch_exclusions": "revival_context_mismatch_exclusions",
    "revival_context_record_coverage": "revival_context_record_coverage",
    "revival_context_recorded_opportunities": "revival_context_recorded_opportunities",
    "revival_context_guard_opportunities": "revival_context_guard_opportunities",
    "revival_context_consistent_confirmations": "revival_context_consistent_confirmations",
    "revival_context_mismatch_confirmations": "revival_context_mismatch_confirmations",
    "revival_post_confirmation_applications": "revival_post_confirmation_applications",
    "revival_post_confirmation_tag_associated_harmful_exposure_n": (
        "revival_post_confirmation_tag_associated_harmful_exposure"
    ),
    "revival_post_confirmation_tag_associated_harmful_exposure_rate": (
        "revival_post_confirmation_tag_associated_harmful_exposure_rate"
    ),
    "revival_unconfirmed_persistent_transitions": ("revival_unconfirmed_persistent_transitions"),
    "revival_mean_oracle_intervention_delta": "revival_mean_oracle_intervention_delta",
    "revival_control_requests": "revival_control_requests",
    "confirmed_context_changes": "confirmed_context_changes",
    "temporally_deferred_changes": "temporally_deferred_changes",
    "total_requests": "total_requests",
    "total_tokens": "total_tokens",
}


@dataclass(frozen=True)
class SweepSpec:
    base_config: Path
    algorithms: Sequence[Algorithm]
    seeds: Sequence[int]
    grid: Mapping[str, Sequence[Any]]
    variants: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    max_runs: int = 100


def load_sweep_spec(path: Path, root: Path) -> SweepSpec:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("sweep spec must be a YAML mapping")
    allowed = {"base_config", "algorithms", "seeds", "grid", "variants", "max_runs"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown sweep fields: {', '.join(unknown)}")
    base = Path(str(payload.get("base_config", "")))
    if not str(base):
        raise ValueError("sweep spec requires base_config")
    base = base if base.is_absolute() else root / base
    algorithms = [Algorithm(value) for value in payload.get("algorithms", ["evoshift"])]
    seeds = [int(value) for value in payload.get("seeds", [42])]
    if not algorithms:
        raise ValueError("sweep algorithms must not be empty")
    if not seeds:
        raise ValueError("sweep seeds must not be empty")
    grid = payload.get("grid", {})
    if not isinstance(grid, dict) or any(
        not isinstance(values, list) or not values for values in grid.values()
    ):
        raise ValueError("sweep grid values must be non-empty lists")
    variants = payload.get("variants", {})
    if not isinstance(variants, dict) or any(
        not isinstance(name, str) or not name.strip() or not isinstance(parameters, dict)
        for name, parameters in variants.items()
    ):
        raise ValueError("sweep variants must map non-empty names to parameter mappings")
    maximum = int(payload.get("max_runs", 100))
    if maximum < 1:
        raise ValueError("max_runs must be positive")
    spec = SweepSpec(base, algorithms, seeds, grid, variants, maximum)
    if len(expand_sweep(spec)) > maximum:
        raise ValueError(f"sweep expands beyond max_runs={maximum}")
    return spec


def _override_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def expand_sweep(spec: SweepSpec) -> List[Dict[str, Any]]:
    keys = sorted(spec.grid)
    values = [spec.grid[key] for key in keys]
    combinations = itertools.product(*values) if values else [()]
    variants = spec.variants or {"default": {}}
    assignments: List[Dict[str, Any]] = []
    for combination in combinations:
        grid_parameters = dict(zip(keys, combination))
        for variant, variant_parameters in variants.items():
            overlap = sorted(set(grid_parameters) & set(variant_parameters))
            if overlap:
                raise ValueError(
                    f"sweep variant {variant!r} overlaps grid fields: {', '.join(overlap)}"
                )
            parameters = {**grid_parameters, **variant_parameters}
            for algorithm in spec.algorithms:
                for seed in spec.seeds:
                    assignments.append(
                        {
                            "algorithm": algorithm.value,
                            "seed": seed,
                            "variant": variant,
                            "parameters": parameters,
                        }
                    )
    return assignments


def _read_total_budget(run_dir: Path) -> Dict[str, Any]:
    payload = json.loads((run_dir / "costs.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("costs.json must contain an object")
    return {str(key): value for key, value in payload.items()}


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sweep_fingerprint(spec: SweepSpec, assignments: Sequence[Mapping[str, Any]]) -> str:
    base_config = spec.base_config.resolve()
    payload = {
        "base_config_sha256": hashlib.sha256(base_config.read_bytes()).hexdigest(),
        "assignments": list(assignments),
        "max_runs": spec.max_runs,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _new_sweep_state(
    spec: SweepSpec,
    assignments: Sequence[Mapping[str, Any]],
    fingerprint: str,
) -> Dict[str, Any]:
    now = _utc_timestamp()
    return {
        "version": 1,
        "status": "running",
        "fingerprint": fingerprint,
        "base_config": str(spec.base_config.resolve()),
        "created_at": now,
        "updated_at": now,
        "completed_runs": 0,
        "total_runs": len(assignments),
        "assignments": [
            {
                "index": index,
                "assignment": dict(assignment),
                "status": "pending",
                "attempts": [],
                "row": None,
            }
            for index, assignment in enumerate(assignments)
        ],
    }


def _load_sweep_state(
    destination: Path,
    assignments: Sequence[Mapping[str, Any]],
    fingerprint: str,
) -> Dict[str, Any]:
    state_path = destination / "sweep_state.json"
    if not state_path.is_file():
        raise ValueError(f"resume directory is missing {state_path.name}: {destination}")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("unsupported or malformed sweep state")
    if payload.get("fingerprint") != fingerprint:
        raise ValueError("resume sweep spec fingerprint does not match the saved state")
    entries = payload.get("assignments")
    if not isinstance(entries, list) or len(entries) != len(assignments):
        raise ValueError("resume sweep assignment count does not match the saved state")
    for index, (entry, assignment) in enumerate(zip(entries, assignments)):
        if not isinstance(entry, dict) or entry.get("assignment") != assignment:
            raise ValueError(f"resume sweep assignment {index} does not match the saved state")
        status = entry.get("status")
        if status not in {"pending", "running", "failed", "completed"}:
            raise ValueError(f"resume sweep assignment {index} has invalid status {status!r}")
        attempts = entry.get("attempts")
        if not isinstance(attempts, list):
            raise ValueError(f"resume sweep assignment {index} has invalid attempt history")
        if status == "completed" and not isinstance(entry.get("row"), dict):
            raise ValueError(f"resume sweep assignment {index} is completed without a row")
        if status == "running":
            if attempts and isinstance(attempts[-1], dict):
                attempts[-1].update(
                    {
                        "status": "interrupted",
                        "completed_at": _utc_timestamp(),
                        "error": "sweep resumed after an interrupted attempt",
                    }
                )
            entry["status"] = "pending"
    payload["status"] = "running"
    payload["updated_at"] = _utc_timestamp()
    payload["completed_runs"] = sum(
        entry.get("status") == "completed" for entry in entries if isinstance(entry, dict)
    )
    return {str(key): value for key, value in payload.items()}


def _run_directories(root: Path, runs_dir: str) -> set[Path]:
    configured = Path(runs_dir).expanduser()
    resolved = configured if configured.is_absolute() else root / configured
    if not resolved.is_dir():
        return set()
    return {path.resolve() for path in resolved.iterdir() if path.is_dir()}


def _write_sweep_outputs(
    destination: Path,
    rows: Sequence[Mapping[str, Any]],
    spec: SweepSpec,
    *,
    complete: bool,
) -> None:
    aggregates = aggregate_sweep(rows)
    comparisons: list[dict[str, Any]] = []
    if complete:
        analysis_config = load_config(spec.base_config)
        comparisons = compare_sweep_runs(
            rows,
            samples=analysis_config.evaluation.bootstrap_samples,
            confidence=analysis_config.evaluation.confidence_level,
        )
        comparisons.extend(
            compare_sweep_variants(
                rows,
                variant_parameters=spec.variants,
                samples=analysis_config.evaluation.bootstrap_samples,
                confidence=analysis_config.evaluation.confidence_level,
            )
        )
    _atomic_write_json(
        destination / "matrix.json",
        {
            "complete": complete,
            "completed_runs": len(rows),
            "total_runs": len(expand_sweep(spec)),
            "runs": list(rows),
            "aggregates": aggregates,
            "comparisons": comparisons,
        },
    )
    _write_csv(destination / "matrix.csv", rows)
    _write_comparison_csv(destination / "comparisons.csv", comparisons)
    _write_sweep_markdown(destination / "report.md", aggregates, comparisons)


async def run_sweep(spec: SweepSpec, root: Path, *, resume: Path | None = None) -> Path:
    assignments = expand_sweep(spec)
    fingerprint = _sweep_fingerprint(spec, assignments)
    if resume is None:
        sweep_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = root / "runs" / "sweeps" / sweep_id
        destination.mkdir(parents=True, exist_ok=False)
        state = _new_sweep_state(spec, assignments, fingerprint)
    else:
        destination = resume.expanduser()
        destination = destination if destination.is_absolute() else root / destination
        destination = destination.resolve()
        if not destination.is_dir():
            raise ValueError(f"resume sweep directory does not exist: {destination}")
        state = _load_sweep_state(destination, assignments, fingerprint)
    state_path = destination / "sweep_state.json"
    _atomic_write_json(state_path, state)
    entries = state["assignments"]
    rows: List[Dict[str, Any]] = [
        dict(entry["row"])
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("status") == "completed"
        and isinstance(entry.get("row"), dict)
    ]
    _write_sweep_outputs(destination, rows, spec, complete=len(rows) == len(assignments))
    for index, assignment in enumerate(assignments):
        entry = entries[index]
        if entry["status"] == "completed":
            continue
        attempt: Dict[str, Any] = {
            "attempt": len(entry["attempts"]) + 1,
            "status": "running",
            "started_at": _utc_timestamp(),
            "run_dirs": [],
        }
        entry["attempts"].append(attempt)
        entry["status"] = "running"
        entry["row"] = None
        state["updated_at"] = _utc_timestamp()
        _atomic_write_json(state_path, state)
        overrides = [
            f"algorithm={assignment['algorithm']}",
            f"evaluation.seed={assignment['seed']}",
        ]
        overrides.extend(
            f"{key}={_override_value(value)}" for key, value in assignment["parameters"].items()
        )
        config = load_config(spec.base_config, overrides)
        adapter = create_benchmark(config.benchmark, root=root, seed=config.evaluation.seed)
        before_run_dirs = _run_directories(root, config.storage.runs_dir)
        try:
            result = await EvoShiftRunner(config, adapter, workdir=root).run()
        except Exception as exc:
            after_run_dirs = _run_directories(root, config.storage.runs_dir)
            attempt.update(
                {
                    "status": "failed",
                    "completed_at": _utc_timestamp(),
                    "run_dirs": [str(path) for path in sorted(after_run_dirs - before_run_dirs)],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            entry["status"] = "failed"
            state["status"] = "incomplete"
            state["updated_at"] = _utc_timestamp()
            _atomic_write_json(state_path, state)
            _write_sweep_outputs(destination, rows, spec, complete=False)
            raise
        attempt["run_dirs"] = [str(result.run_dir.resolve())]
        state["updated_at"] = _utc_timestamp()
        _atomic_write_json(state_path, state)
        budget = _read_total_budget(result.run_dir)
        policy_shift = result.metrics.get("policy_shift", {})
        feedback = result.metrics.get("feedback", {})
        retrieval = result.metrics.get("retrieval", {})
        evolution = result.metrics.get("evolution", {})
        future_audit = result.metrics.get("future_audit", {})
        active_governance = result.metrics.get("active_memory_governance", {})
        circuit = active_governance.get("circuit_breaker", {})
        retirement_probation = active_governance.get("retirement_probation", {})
        revival = active_governance.get("dormant_revival", {})
        trust_model = result.metrics.get("feedback_trust_model", {})
        recovery_steps = result.metrics.get("recovery_steps", {})
        recovered = [
            float(value) for value in recovery_steps.values() if isinstance(value, (int, float))
        ]
        rows.append(
            {
                "run_id": result.run_id,
                "run_dir": str(result.run_dir),
                "algorithm": assignment["algorithm"],
                "variant": assignment["variant"],
                "seed": assignment["seed"],
                "parameters": assignment["parameters"],
                "mean_score": result.metrics["overall"]["mean_score"],
                "success_rate": result.metrics["overall"]["success_rate"],
                "auac": result.metrics["auac"],
                "cumulative_regret": result.metrics["cumulative_regret"],
                "foreground_tokens": result.metrics["resources"]["total_tokens"],
                "total_tokens": budget["total_tokens"],
                "total_requests": budget["requests"],
                "cost_usd": budget["cost_usd"],
                "changed_case_success_rate": policy_shift.get("changed_case_success_rate"),
                "first_changed_case_success_rate": policy_shift.get(
                    "first_changed_case_success_rate"
                ),
                "old_rule_leakage_rate": policy_shift.get("old_rule_leakage_rate"),
                "invariant_retention_rate": policy_shift.get("invariant_retention_rate"),
                "future_change_case_success_rate": policy_shift.get(
                    "future_change_case_success_rate"
                ),
                "future_change_case_n": policy_shift.get("future_change_case_n"),
                "phase_slice_counts": policy_shift.get("phase_slice_counts", {}),
                "premature_update_rate": policy_shift.get("premature_update_rate"),
                "corrupted_feedback_follow_rate": policy_shift.get(
                    "corrupted_feedback_follow_rate"
                ),
                "attack_feedback_follow_rate": policy_shift.get("attack_feedback_follow_rate"),
                "premature_attack_follow_rate": policy_shift.get("premature_attack_follow_rate"),
                "poison_persistence_error_rate": policy_shift.get("poison_persistence_error_rate"),
                "corrupted_feedback_quarantine_rate": feedback.get(
                    "corrupted_feedback_quarantine_rate"
                ),
                "clean_feedback_quarantine_rate": feedback.get("clean_feedback_quarantine_rate"),
                "mean_recovery_steps": statistics.fmean(recovered) if recovered else None,
                "unrecovered_shifts": sum(value is None for value in recovery_steps.values()),
                "shift_detection_events": evolution.get("shift_detection_events"),
                "memory_candidates_promoted": evolution.get("memory_candidates_promoted"),
                "shadow_failure_extractions": evolution.get("shadow_failure_extractions"),
                "shadow_candidate_observations": evolution.get("shadow_candidate_observations"),
                "shadow_candidate_replay_attempts": evolution.get(
                    "shadow_candidate_replay_attempts"
                ),
                "shadow_only_candidate_replay_attempts": evolution.get(
                    "shadow_only_candidate_replay_attempts"
                ),
                "shadow_candidate_probations": evolution.get("shadow_candidate_probations"),
                "shadow_candidate_activations": evolution.get("shadow_candidate_activations"),
                "shadow_candidate_rejections": evolution.get("shadow_candidate_rejections"),
                "shadow_candidate_expirations": evolution.get("shadow_candidate_expirations"),
                "shadow_eprocess_opportunities": evolution.get("shadow_eprocess_opportunities"),
                "shadow_eprocess_crossings": evolution.get("shadow_eprocess_crossings"),
                "shadow_cluster_eprocess_opportunities": evolution.get(
                    "shadow_cluster_eprocess_opportunities"
                ),
                "shadow_cluster_eprocess_crossings": evolution.get(
                    "shadow_cluster_eprocess_crossings"
                ),
                "shadow_cluster_eprocess_skipped_unrelated": evolution.get(
                    "shadow_cluster_eprocess_skipped_unrelated"
                ),
                "trusted_candidate_shadow_cooldown_bypasses": evolution.get(
                    "trusted_candidate_shadow_cooldown_bypasses"
                ),
                "context_probation_interventions": evolution.get("context_probation_interventions"),
                "context_probation_paired_controls": evolution.get(
                    "context_probation_paired_controls"
                ),
                "context_probation_forced_applications": evolution.get(
                    "context_probation_forced_applications"
                ),
                "context_probation_expired": evolution.get("context_probation_expired"),
                "context_probation_retrieval_hit_bypasses": evolution.get(
                    "context_probation_retrieval_hit_bypasses"
                ),
                "rerank_attempts": retrieval.get("rerank_attempts"),
                "rerank_applied": retrieval.get("rerank_applied"),
                "rerank_fallbacks": retrieval.get("rerank_fallbacks"),
                "rerank_fallback_rate": retrieval.get("rerank_fallback_rate"),
                "promotion_precision": result.metrics.get("promotion_precision"),
                "promotion_precision_basis": result.metrics.get("promotion_precision_basis"),
                "replay_estimated_promotion_precision": result.metrics.get(
                    "replay_estimated_promotion_precision"
                ),
                "realized_promotion_precision": result.metrics.get("realized_promotion_precision"),
                "future_audit_confirmed": future_audit.get("confirmed"),
                "future_audit_rolled_back": future_audit.get("rolled_back"),
                "future_audit_expired": future_audit.get("expired"),
                "future_audit_realized_promotion_coverage": future_audit.get(
                    "realized_promotion_coverage"
                ),
                "future_audit_harmful_promotion_rate": future_audit.get("harmful_promotion_rate"),
                "future_audit_false_rollback_rate": future_audit.get("false_rollback_rate"),
                "future_audit_mean_oracle_delta": future_audit.get("mean_oracle_delta"),
                "future_audit_mean_audit_latency": future_audit.get("mean_audit_latency"),
                "future_audit_mean_confirmation_latency": future_audit.get(
                    "mean_confirmation_latency"
                ),
                "future_audit_mean_rollback_latency": future_audit.get("mean_rollback_latency"),
                "future_audit_mean_harmful_exposure_latency": future_audit.get(
                    "mean_harmful_exposure_latency"
                ),
                "future_audit_mean_confirmation_observations": future_audit.get(
                    "mean_confirmation_observations"
                ),
                "future_audit_mean_rollback_observations": future_audit.get(
                    "mean_rollback_observations"
                ),
                "future_audit_mean_harmful_exposure_observations": future_audit.get(
                    "mean_harmful_exposure_observations"
                ),
                "memory_supersessions": evolution.get("memory_supersessions"),
                "memory_reactivations": evolution.get("memory_reactivations"),
                "correct_reactivation_rate": active_governance.get("correct_reactivation_rate"),
                "harmful_active_memory_exposure_n": active_governance.get(
                    "harmful_active_memory_exposure_n"
                ),
                "stale_memory_retention_rate": active_governance.get("stale_memory_retention_rate"),
                "selective_forgetting_precision": active_governance.get(
                    "selective_forgetting_precision"
                ),
                "selective_forgetting_recall": active_governance.get("selective_forgetting_recall"),
                "false_retirement_rate": active_governance.get("false_retirement_rate"),
                "correct_reacquisition_rate": active_governance.get("correct_reacquisition_rate"),
                "memory_reacquisitions": active_governance.get("reacquisitions"),
                "active_audit_control_requests": active_governance.get("control_requests"),
                "reactivation_grace_audit_suppressions": active_governance.get(
                    "reactivation_grace_audit_suppressions"
                ),
                "reactivation_grace_protected_rollbacks": active_governance.get(
                    "reactivation_grace_protected_rollbacks"
                ),
                "counterfactual_audit_coverage": active_governance.get(
                    "counterfactual_audit_coverage"
                ),
                "active_audit_budget_utilization": active_governance.get(
                    "audit_budget_utilization"
                ),
                "active_audit_mean_retirement_latency": active_governance.get(
                    "mean_retirement_latency"
                ),
                "early_causal_retirements": active_governance.get("early_causal_retirements"),
                "early_causal_retirement_precision": active_governance.get(
                    "early_causal_retirement_precision"
                ),
                "early_causal_false_retirement_rate": active_governance.get(
                    "early_causal_false_retirement_rate"
                ),
                "circuit_registrations": circuit.get("registrations"),
                "circuit_lineage_probes": circuit.get("lineage_probes"),
                "circuit_lineage_registrations": circuit.get("lineage_registrations"),
                "circuit_interventions": circuit.get("interventions"),
                "circuit_lineage_interventions": circuit.get("lineage_interventions"),
                "circuit_confirmations": circuit.get("confirmations"),
                "circuit_cancellations": circuit.get("cancellations"),
                "circuit_expirations": circuit.get("expirations"),
                "circuit_confirmation_precision": circuit.get("confirmation_precision"),
                "circuit_false_confirmation_rate": circuit.get("false_confirmation_rate"),
                "circuit_unconfirmed_persistent_transitions": circuit.get(
                    "unconfirmed_persistent_transitions"
                ),
                "circuit_mean_oracle_intervention_delta": circuit.get(
                    "mean_post_hoc_oracle_intervention_delta"
                ),
                "circuit_control_requests": circuit.get("control_requests"),
                "retirement_probation_registrations": retirement_probation.get("registrations"),
                "retirement_probation_interventions": retirement_probation.get("interventions"),
                "retirement_probation_confirmations": retirement_probation.get("confirmations"),
                "retirement_probation_cancellations": retirement_probation.get("cancellations"),
                "retirement_probation_deferrals": retirement_probation.get("deferrals"),
                "retirement_probation_expirations": retirement_probation.get("expirations"),
                "retirement_probation_registration_precision": retirement_probation.get(
                    "registration_precision"
                ),
                "retirement_probation_registration_label_coverage": retirement_probation.get(
                    "registration_label_coverage"
                ),
                "retirement_probation_unknown_tag_registrations": retirement_probation.get(
                    "unknown_tag_registrations"
                ),
                "retirement_probation_paired_oracle_registration_precision": (
                    retirement_probation.get("paired_oracle_registration_precision")
                ),
                "retirement_probation_paired_oracle_registration_coverage": (
                    retirement_probation.get("paired_oracle_registration_coverage")
                ),
                "retirement_probation_confirmation_precision": retirement_probation.get(
                    "confirmation_precision"
                ),
                "retirement_probation_confirmation_label_coverage": retirement_probation.get(
                    "confirmation_label_coverage"
                ),
                "retirement_probation_unknown_tag_confirmations": retirement_probation.get(
                    "unknown_tag_confirmations"
                ),
                "retirement_probation_confirmation_coverage": retirement_probation.get(
                    "confirmation_coverage"
                ),
                "retirement_probation_expiry_rate": retirement_probation.get("expiry_rate"),
                "retirement_probation_provisional_valid_tag_exposure_n": (
                    retirement_probation.get("provisional_valid_tag_exposure_n")
                ),
                "retirement_probation_provisional_valid_tag_failure_episodes": (
                    retirement_probation.get("provisional_valid_tag_failure_episodes")
                ),
                "retirement_probation_false_confirmation_rate": retirement_probation.get(
                    "false_confirmation_rate"
                ),
                "retirement_probation_unconfirmed_persistent_transitions": (
                    retirement_probation.get("unconfirmed_persistent_transitions")
                ),
                "retirement_probation_mean_oracle_intervention_delta": (
                    retirement_probation.get("mean_post_hoc_oracle_intervention_delta")
                ),
                "retirement_probation_control_requests": retirement_probation.get(
                    "control_requests"
                ),
                "revival_registrations": revival.get("registrations"),
                "revival_interventions": revival.get("interventions"),
                "revival_confirmations": revival.get("confirmations"),
                "revival_cancellations": revival.get("cancellations"),
                "revival_expirations": revival.get("expirations"),
                "revival_confirmation_precision": revival.get("confirmation_precision"),
                "revival_confirmation_coverage": revival.get("confirmation_coverage"),
                "revival_false_confirmation_rate": revival.get("false_confirmation_rate"),
                "revival_context_mismatch_exclusions": revival.get("context_mismatch_exclusions"),
                "revival_context_record_coverage": revival.get("context_record_coverage"),
                "revival_context_recorded_opportunities": revival.get(
                    "context_recorded_opportunities"
                ),
                "revival_context_guard_opportunities": revival.get("context_guard_opportunities"),
                "revival_context_consistent_confirmations": revival.get(
                    "context_consistent_confirmations"
                ),
                "revival_context_mismatch_confirmations": revival.get(
                    "context_mismatch_confirmations"
                ),
                "revival_post_confirmation_applications": revival.get(
                    "post_confirmation_applications"
                ),
                "revival_post_confirmation_tag_associated_harmful_exposure_n": revival.get(
                    "post_confirmation_tag_associated_harmful_exposure_n"
                ),
                "revival_post_confirmation_tag_associated_harmful_exposure_rate": revival.get(
                    "post_confirmation_tag_associated_harmful_exposure_rate"
                ),
                "revival_unconfirmed_persistent_transitions": revival.get(
                    "unconfirmed_persistent_transitions"
                ),
                "revival_mean_oracle_intervention_delta": revival.get(
                    "mean_post_hoc_oracle_intervention_delta"
                ),
                "revival_control_requests": revival.get("control_requests"),
                "confirmed_context_changes": trust_model.get("confirmed_context_changes"),
                "temporally_deferred_changes": trust_model.get("temporally_deferred_changes"),
            }
        )
        row = rows[-1]
        attempt.update(
            {
                "status": "completed",
                "completed_at": _utc_timestamp(),
            }
        )
        entry["status"] = "completed"
        entry["row"] = row
        state["completed_runs"] = len(rows)
        state["updated_at"] = _utc_timestamp()
        _atomic_write_json(state_path, state)
        _write_sweep_outputs(
            destination,
            rows,
            spec,
            complete=len(rows) == len(assignments),
        )
    state["status"] = "complete"
    state["completed_runs"] = len(rows)
    state["completed_at"] = _utc_timestamp()
    state["updated_at"] = state["completed_at"]
    _atomic_write_json(state_path, state)
    return destination


def compare_sweep_runs(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_algorithm: str = "evoshift",
    samples: int = 2000,
    confidence: float = 0.95,
    bootstrap_seed: int = 42,
) -> list[dict[str, Any]]:
    """Build paired repeated-seed capability, safety, and cost intervals."""

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        identity = json.dumps(
            {
                "variant": row.get("variant", "default"),
                "parameters": row.get("parameters", {}),
            },
            sort_keys=True,
        )
        grouped.setdefault(identity, []).append(row)

    output: list[dict[str, Any]] = []
    for identity, group in grouped.items():
        condition = json.loads(identity)
        by_algorithm: dict[str, dict[int, Mapping[str, Any]]] = {}
        for row in group:
            algorithm = str(row["algorithm"])
            stream_seed = int(row["seed"])
            algorithm_rows = by_algorithm.setdefault(algorithm, {})
            if stream_seed in algorithm_rows:
                raise ValueError(
                    f"duplicate sweep row for algorithm={algorithm} seed={stream_seed}"
                )
            algorithm_rows[stream_seed] = row
        candidate_rows = by_algorithm.get(candidate_algorithm)
        if candidate_rows is None:
            continue
        for baseline_algorithm, baseline_rows in sorted(by_algorithm.items()):
            if baseline_algorithm == candidate_algorithm:
                continue
            if set(candidate_rows) != set(baseline_rows):
                raise ValueError(
                    "paired repeated-seed comparison requires identical seed sets for "
                    f"{candidate_algorithm} and {baseline_algorithm}"
                )
            seeds = sorted(candidate_rows)
            episode_pairs: dict[int, tuple[list[Episode], list[Episode]]] = {}
            for stream_seed in seeds:
                baseline_episodes = load_episodes(
                    Path(str(baseline_rows[stream_seed]["run_dir"])) / "predictions.jsonl"
                )
                candidate_episodes = load_episodes(
                    Path(str(candidate_rows[stream_seed]["run_dir"])) / "predictions.jsonl"
                )
                episode_pairs[stream_seed] = _align_episode_pair(
                    baseline_episodes,
                    candidate_episodes,
                )

            metrics: dict[str, dict[str, Any]] = {}
            episode_specs: dict[
                str,
                tuple[Callable[[Episode], bool], Callable[[Episode], float]],
            ] = {
                "score": (lambda _episode: True, lambda episode: episode.score.primary),
                "changed_case_success": (
                    lambda episode: bool(episode.sample.metadata.get("policy_changed_case")),
                    lambda episode: float(episode.score.success),
                ),
                "old_rule_leakage": (
                    lambda episode: bool(episode.sample.metadata.get("policy_changed_case")),
                    lambda episode: float(not episode.score.success),
                ),
                "invariant_retention": (
                    lambda episode: bool(episode.sample.metadata.get("protected")),
                    lambda episode: float(episode.score.success),
                ),
                "premature_update": (
                    lambda episode: bool(episode.sample.metadata.get("future_change_case")),
                    lambda episode: float(not episode.score.success),
                ),
                "attack_feedback_follow": (
                    lambda episode: episode.sample.metadata.get("feedback_kind") == "attack",
                    lambda episode: float(episode.adaptation_score.success),
                ),
            }
            for metric_name, (predicate, value) in episode_specs.items():
                deltas_by_seed: dict[int, list[float]] = {}
                for stream_seed, (baseline_episodes, candidate_episodes) in episode_pairs.items():
                    deltas = [
                        value(candidate) - value(baseline)
                        for baseline, candidate in zip(baseline_episodes, candidate_episodes)
                        if predicate(candidate)
                    ]
                    if deltas:
                        deltas_by_seed[stream_seed] = deltas
                if len(deltas_by_seed) == len(seeds):
                    metrics[metric_name] = _cluster_interval_payload(
                        deltas_by_seed,
                        samples=samples,
                        confidence=confidence,
                        bootstrap_seed=bootstrap_seed,
                    )

            for source, metric_name in (
                ("total_requests", "total_requests"),
                ("total_tokens", "total_tokens"),
                ("cost_usd", "cost_usd"),
                ("harmful_active_memory_exposure_n", "harmful_active_memory_exposure"),
                ("stale_memory_retention_rate", "stale_memory_retention"),
                ("false_retirement_rate", "false_retirement"),
            ):
                run_deltas: dict[int, list[float]] = {}
                for stream_seed in seeds:
                    baseline_value = baseline_rows[stream_seed].get(source)
                    candidate_value = candidate_rows[stream_seed].get(source)
                    if not isinstance(baseline_value, (int, float)) or not isinstance(
                        candidate_value, (int, float)
                    ):
                        break
                    run_deltas[stream_seed] = [float(candidate_value) - float(baseline_value)]
                if len(run_deltas) == len(seeds):
                    metrics[metric_name] = _cluster_interval_payload(
                        run_deltas,
                        samples=samples,
                        confidence=confidence,
                        bootstrap_seed=bootstrap_seed,
                    )

            output.append(
                {
                    **condition,
                    "candidate_algorithm": candidate_algorithm,
                    "baseline_algorithm": baseline_algorithm,
                    "n_seeds": len(seeds),
                    "seed_list": seeds,
                    "confidence": confidence,
                    "bootstrap_samples": samples,
                    "bootstrap_unit": "seed_then_paired_sample",
                    "metrics": metrics,
                }
            )
    return sorted(
        output,
        key=lambda item: (
            json.dumps(item["parameters"], sort_keys=True),
            item["variant"],
            item["baseline_algorithm"],
        ),
    )


def compare_sweep_variants(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant_parameters: Mapping[str, Mapping[str, Any]],
    baseline_variant: str | None = None,
    samples: int = 2000,
    confidence: float = 0.95,
    bootstrap_seed: int = 42,
) -> list[dict[str, Any]]:
    """Compare named variants while holding the sweep grid and algorithm fixed."""

    if len(variant_parameters) < 2:
        return []
    variant_names = list(variant_parameters)
    if baseline_variant is None:
        baseline_variant = next(
            (name for name in ("current_full", "full", "default") if name in variant_parameters),
            variant_names[0],
        )
    if baseline_variant not in variant_parameters:
        raise ValueError(f"unknown baseline sweep variant: {baseline_variant}")
    variant_override_keys = set().union(
        *(parameters.keys() for parameters in variant_parameters.values())
    )
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        parameters = row.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError("sweep row parameters must be a mapping")
        grid_parameters = {
            str(key): value for key, value in parameters.items() if key not in variant_override_keys
        }
        identity = json.dumps(
            {
                "algorithm": row["algorithm"],
                "parameters": grid_parameters,
            },
            sort_keys=True,
        )
        grouped.setdefault(identity, []).append(row)

    comparisons: list[dict[str, Any]] = []
    for identity, group in grouped.items():
        condition = json.loads(identity)
        baseline_rows = [row for row in group if row.get("variant") == baseline_variant]
        if not baseline_rows:
            continue
        for candidate_variant in variant_names:
            if candidate_variant == baseline_variant:
                continue
            candidate_rows = [row for row in group if row.get("variant") == candidate_variant]
            if not candidate_rows:
                continue
            pseudo_rows: list[dict[str, Any]] = []
            for algorithm, selected in (
                ("__baseline_variant__", baseline_rows),
                ("__candidate_variant__", candidate_rows),
            ):
                for row in selected:
                    pseudo_rows.append(
                        {
                            **row,
                            "algorithm": algorithm,
                            "variant": f"{candidate_variant}_vs_{baseline_variant}",
                            "parameters": condition["parameters"],
                        }
                    )
            paired = compare_sweep_runs(
                pseudo_rows,
                candidate_algorithm="__candidate_variant__",
                samples=samples,
                confidence=confidence,
                bootstrap_seed=bootstrap_seed,
            )
            if len(paired) != 1:
                raise ValueError("named sweep variant comparison did not produce one pair")
            comparison = paired[0]
            comparison.update(
                {
                    "candidate_algorithm": (f"{condition['algorithm']}:{candidate_variant}"),
                    "baseline_algorithm": (f"{condition['algorithm']}:{baseline_variant}"),
                }
            )
            comparisons.append(comparison)
    return sorted(
        comparisons,
        key=lambda item: (
            str(item["candidate_algorithm"]),
            json.dumps(item["parameters"], sort_keys=True),
        ),
    )


def _align_episode_pair(
    baseline: Sequence[Episode],
    candidate: Sequence[Episode],
) -> tuple[list[Episode], list[Episode]]:
    baseline_by_id = {episode.sample.sample_id: episode for episode in baseline}
    candidate_by_id = {episode.sample.sample_id: episode for episode in candidate}
    if len(baseline_by_id) != len(baseline) or len(candidate_by_id) != len(candidate):
        raise ValueError("paired sweep runs contain duplicate sample ids")
    if set(baseline_by_id) != set(candidate_by_id):
        raise ValueError("paired sweep runs do not contain identical sample ids")
    ordered_ids = [episode.sample.sample_id for episode in candidate]
    return (
        [baseline_by_id[sample_id] for sample_id in ordered_ids],
        [candidate_by_id[sample_id] for sample_id in ordered_ids],
    )


def _cluster_interval_payload(
    deltas_by_seed: Mapping[int, Sequence[float]],
    *,
    samples: int,
    confidence: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    mean_delta, ci_low, ci_high = paired_cluster_bootstrap_ci(
        deltas_by_seed,
        samples=samples,
        confidence=confidence,
        seed=bootstrap_seed,
    )
    return {
        "delta_mean": mean_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_seeds": len(deltas_by_seed),
        "n_pairs": sum(len(values) for values in deltas_by_seed.values()),
    }


def aggregate_sweep(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        key = json.dumps(
            {
                "algorithm": row["algorithm"],
                "variant": row.get("variant", "default"),
                "parameters": row["parameters"],
            },
            sort_keys=True,
        )
        grouped.setdefault(key, []).append(row)
    output = []
    for key, group in grouped.items():
        identity = json.loads(key)
        aggregate: Dict[str, Any] = {
            **identity,
            "n_seeds": len(group),
            "run_ids": [row["run_id"] for row in group],
        }
        for source, prefix in _AGGREGATE_FIELDS.items():
            values = [
                float(row[source]) for row in group if isinstance(row.get(source), (int, float))
            ]
            if not values:
                continue
            aggregate[f"{prefix}_mean"] = statistics.fmean(values)
            aggregate[f"{prefix}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        recorded_contexts = sum(
            float(row["revival_context_recorded_opportunities"])
            for row in group
            if isinstance(row.get("revival_context_recorded_opportunities"), (int, float))
        )
        context_opportunities = sum(
            float(row["revival_context_guard_opportunities"])
            for row in group
            if isinstance(row.get("revival_context_guard_opportunities"), (int, float))
        )
        aggregate["revival_context_record_coverage_micro"] = (
            recorded_contexts / context_opportunities if context_opportunities else None
        )
        output.append(aggregate)
    return sorted(output, key=lambda item: (-item["score_mean"], item["algorithm"]))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "run_id",
        "algorithm",
        "variant",
        "seed",
        "parameters",
        "mean_score",
        "success_rate",
        "auac",
        "cumulative_regret",
        "foreground_tokens",
        "total_tokens",
        "total_requests",
        "cost_usd",
        "changed_case_success_rate",
        "first_changed_case_success_rate",
        "old_rule_leakage_rate",
        "invariant_retention_rate",
        "future_change_case_success_rate",
        "future_change_case_n",
        "phase_slice_counts",
        "premature_update_rate",
        "corrupted_feedback_follow_rate",
        "attack_feedback_follow_rate",
        "premature_attack_follow_rate",
        "poison_persistence_error_rate",
        "corrupted_feedback_quarantine_rate",
        "clean_feedback_quarantine_rate",
        "mean_recovery_steps",
        "unrecovered_shifts",
        "shift_detection_events",
        "memory_candidates_promoted",
        "shadow_failure_extractions",
        "shadow_candidate_observations",
        "shadow_candidate_replay_attempts",
        "shadow_only_candidate_replay_attempts",
        "shadow_candidate_probations",
        "shadow_candidate_activations",
        "shadow_candidate_rejections",
        "shadow_candidate_expirations",
        "shadow_eprocess_opportunities",
        "shadow_eprocess_crossings",
        "shadow_cluster_eprocess_opportunities",
        "shadow_cluster_eprocess_crossings",
        "shadow_cluster_eprocess_skipped_unrelated",
        "trusted_candidate_shadow_cooldown_bypasses",
        "context_probation_interventions",
        "context_probation_paired_controls",
        "context_probation_forced_applications",
        "context_probation_expired",
        "context_probation_retrieval_hit_bypasses",
        "rerank_attempts",
        "rerank_applied",
        "rerank_fallbacks",
        "rerank_fallback_rate",
        "promotion_precision",
        "promotion_precision_basis",
        "replay_estimated_promotion_precision",
        "realized_promotion_precision",
        "future_audit_confirmed",
        "future_audit_rolled_back",
        "future_audit_expired",
        "future_audit_realized_promotion_coverage",
        "future_audit_harmful_promotion_rate",
        "future_audit_false_rollback_rate",
        "future_audit_mean_oracle_delta",
        "future_audit_mean_audit_latency",
        "future_audit_mean_confirmation_latency",
        "future_audit_mean_rollback_latency",
        "future_audit_mean_harmful_exposure_latency",
        "future_audit_mean_confirmation_observations",
        "future_audit_mean_rollback_observations",
        "future_audit_mean_harmful_exposure_observations",
        "memory_supersessions",
        "memory_reactivations",
        "correct_reactivation_rate",
        "harmful_active_memory_exposure_n",
        "stale_memory_retention_rate",
        "selective_forgetting_precision",
        "selective_forgetting_recall",
        "false_retirement_rate",
        "correct_reacquisition_rate",
        "memory_reacquisitions",
        "active_audit_control_requests",
        "reactivation_grace_audit_suppressions",
        "reactivation_grace_protected_rollbacks",
        "counterfactual_audit_coverage",
        "active_audit_budget_utilization",
        "active_audit_mean_retirement_latency",
        "early_causal_retirements",
        "early_causal_retirement_precision",
        "early_causal_false_retirement_rate",
        "circuit_registrations",
        "circuit_lineage_probes",
        "circuit_lineage_registrations",
        "circuit_interventions",
        "circuit_lineage_interventions",
        "circuit_confirmations",
        "circuit_cancellations",
        "circuit_expirations",
        "circuit_confirmation_precision",
        "circuit_false_confirmation_rate",
        "circuit_unconfirmed_persistent_transitions",
        "circuit_mean_oracle_intervention_delta",
        "circuit_control_requests",
        "retirement_probation_registrations",
        "retirement_probation_interventions",
        "retirement_probation_confirmations",
        "retirement_probation_cancellations",
        "retirement_probation_deferrals",
        "retirement_probation_expirations",
        "retirement_probation_registration_precision",
        "retirement_probation_registration_label_coverage",
        "retirement_probation_unknown_tag_registrations",
        "retirement_probation_paired_oracle_registration_precision",
        "retirement_probation_paired_oracle_registration_coverage",
        "retirement_probation_confirmation_precision",
        "retirement_probation_confirmation_label_coverage",
        "retirement_probation_unknown_tag_confirmations",
        "retirement_probation_confirmation_coverage",
        "retirement_probation_expiry_rate",
        "retirement_probation_provisional_valid_tag_exposure_n",
        "retirement_probation_provisional_valid_tag_failure_episodes",
        "retirement_probation_false_confirmation_rate",
        "retirement_probation_unconfirmed_persistent_transitions",
        "retirement_probation_mean_oracle_intervention_delta",
        "retirement_probation_control_requests",
        "revival_registrations",
        "revival_interventions",
        "revival_confirmations",
        "revival_cancellations",
        "revival_expirations",
        "revival_confirmation_precision",
        "revival_confirmation_coverage",
        "revival_false_confirmation_rate",
        "revival_context_mismatch_exclusions",
        "revival_context_record_coverage",
        "revival_context_recorded_opportunities",
        "revival_context_guard_opportunities",
        "revival_context_consistent_confirmations",
        "revival_context_mismatch_confirmations",
        "revival_post_confirmation_applications",
        "revival_post_confirmation_tag_associated_harmful_exposure_n",
        "revival_post_confirmation_tag_associated_harmful_exposure_rate",
        "revival_unconfirmed_persistent_transitions",
        "revival_mean_oracle_intervention_delta",
        "revival_control_requests",
        "confirmed_context_changes",
        "temporally_deferred_changes",
        "run_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            encoded["parameters"] = json.dumps(row["parameters"], sort_keys=True)
            encoded["phase_slice_counts"] = json.dumps(
                row.get("phase_slice_counts", {}), sort_keys=True
            )
            writer.writerow(encoded)


def _write_comparison_csv(
    path: Path,
    comparisons: Sequence[Mapping[str, Any]],
) -> None:
    fields = [
        "candidate_algorithm",
        "baseline_algorithm",
        "variant",
        "parameters",
        "metric",
        "delta_mean",
        "ci_low",
        "ci_high",
        "n_seeds",
        "n_pairs",
        "confidence",
        "bootstrap_samples",
        "bootstrap_unit",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for comparison in comparisons:
            metrics = comparison.get("metrics", {})
            if not isinstance(metrics, Mapping):
                continue
            for metric_name, interval in sorted(metrics.items()):
                if not isinstance(interval, Mapping):
                    continue
                writer.writerow(
                    {
                        "candidate_algorithm": comparison["candidate_algorithm"],
                        "baseline_algorithm": comparison["baseline_algorithm"],
                        "variant": comparison["variant"],
                        "parameters": json.dumps(comparison["parameters"], sort_keys=True),
                        "metric": metric_name,
                        "delta_mean": interval.get("delta_mean"),
                        "ci_low": interval.get("ci_low"),
                        "ci_high": interval.get("ci_high"),
                        "n_seeds": interval.get("n_seeds"),
                        "n_pairs": interval.get("n_pairs"),
                        "confidence": comparison["confidence"],
                        "bootstrap_samples": comparison["bootstrap_samples"],
                        "bootstrap_unit": comparison["bootstrap_unit"],
                    }
                )


def _write_sweep_markdown(
    path: Path,
    aggregates: Iterable[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]] = (),
) -> None:
    aggregate_rows = list(aggregates)
    lines = [
        "# EvoShift sweep report",
        "",
        (
            "| Algorithm | Variant | Parameters | Seeds | Score | Changed | Old leakage | "
            "Invariant | Premature | Attack follow | Corrupt quarantine | Promotion precision | "
            "Replay-est. precision | Realized precision | Realized coverage | Audit rollback | "
            "Harmful promotion | False rollback | Harm exposure | Stale retention | Forget P | "
            "Forget R | False retire | Reacquire | Recovery | Cluster opp | Cluster cross | "
            "Requests | Tokens |"
        ),
        (
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]
    for item in aggregate_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    item["algorithm"],
                    item["variant"],
                    f"`{json.dumps(item['parameters'], sort_keys=True)}`",
                    str(item["n_seeds"]),
                    f"{item['score_mean']:.4f}±{item['score_std']:.4f}",
                    _format_mean(item, "changed_case_success"),
                    _format_mean(item, "old_rule_leakage"),
                    _format_mean(item, "invariant_retention"),
                    _format_mean(item, "premature_update"),
                    _format_mean(item, "attack_feedback_follow"),
                    _format_mean(item, "corrupted_feedback_quarantine"),
                    _format_mean(item, "promotion_precision"),
                    _format_mean(item, "replay_estimated_promotion_precision"),
                    _format_mean(item, "realized_promotion_precision"),
                    _format_mean(item, "future_audit_realized_promotion_coverage"),
                    _format_mean(item, "future_audit_rolled_back"),
                    _format_mean(item, "future_audit_harmful_promotion_rate"),
                    _format_mean(item, "future_audit_false_rollback_rate"),
                    _format_mean(item, "harmful_active_memory_exposure"),
                    _format_mean(item, "stale_memory_retention"),
                    _format_mean(item, "selective_forgetting_precision"),
                    _format_mean(item, "selective_forgetting_recall"),
                    _format_mean(item, "false_retirement"),
                    _format_mean(item, "correct_reacquisition"),
                    _format_mean(item, "mean_recovery_steps"),
                    _format_mean(item, "shadow_cluster_eprocess_opportunities"),
                    _format_mean(item, "shadow_cluster_eprocess_crossings"),
                    f"{item.get('total_requests_mean', 0.0):.1f}",
                    f"{item.get('total_tokens_mean', 0.0):.1f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Lifecycle path diagnostics",
            "",
            (
                "Tag-based retirement precision excludes registrations without an exact "
                "stale-versus-valid tag label. Provisional exposure and failure episodes "
                "are separate proxy counts, not a causal rate."
            ),
            "",
            (
                "| Algorithm | Variant | Retire reg P | Tag coverage | Paired reg P | "
                "Paired coverage | Retire confirm P | Confirm coverage | Expiry | "
                "Valid exposure | Failure episodes | Revival P | Revival coverage | "
                "Context coverage | Mismatch excluded | Context-confirmed | "
                "Mismatch-confirmed | Post-revival apps | Post-revival harm |"
            ),
            (
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
                "---:|---:|---:|---:|---:|---:|"
            ),
        ]
    )
    for item in aggregate_rows:
        context_coverage = item.get("revival_context_record_coverage_micro")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["algorithm"]),
                    str(item["variant"]),
                    _format_mean(item, "retirement_probation_registration_precision"),
                    _format_mean(item, "retirement_probation_registration_label_coverage"),
                    _format_mean(
                        item,
                        "retirement_probation_paired_oracle_registration_precision",
                    ),
                    _format_mean(
                        item,
                        "retirement_probation_paired_oracle_registration_coverage",
                    ),
                    _format_mean(item, "retirement_probation_confirmation_precision"),
                    _format_mean(item, "retirement_probation_confirmation_coverage"),
                    _format_mean(item, "retirement_probation_expiry_rate"),
                    _format_mean(
                        item,
                        "retirement_probation_provisional_valid_tag_exposure",
                    ),
                    _format_mean(
                        item,
                        "retirement_probation_provisional_valid_tag_failures",
                    ),
                    _format_mean(item, "revival_confirmation_precision"),
                    _format_mean(item, "revival_confirmation_coverage"),
                    (
                        f"{float(context_coverage):.4f}"
                        if isinstance(context_coverage, (int, float))
                        else "N/A"
                    ),
                    _format_mean(item, "revival_context_mismatch_exclusions"),
                    _format_mean(item, "revival_context_consistent_confirmations"),
                    _format_mean(item, "revival_context_mismatch_confirmations"),
                    _format_mean(item, "revival_post_confirmation_applications"),
                    _format_mean(
                        item,
                        "revival_post_confirmation_tag_associated_harmful_exposure",
                    ),
                ]
            )
            + " |"
        )
    if comparisons:
        lines.extend(
            [
                "",
                "## Paired repeated-seed comparisons",
                "",
                (
                    "Intervals resample seed clusters first and paired samples within "
                    "each selected seed. Deltas are candidate minus baseline."
                ),
                "",
                (
                    "| Candidate | Baseline | Variant | Parameters | Seeds | Score delta | "
                    "Changed delta | Invariant delta | Old leakage delta | Requests delta | "
                    "Tokens delta |"
                ),
                "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in comparisons:
            metrics = item.get("metrics", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item["candidate_algorithm"]),
                        str(item["baseline_algorithm"]),
                        str(item["variant"]),
                        f"`{json.dumps(item['parameters'], sort_keys=True)}`",
                        str(item["n_seeds"]),
                        _format_interval(metrics, "score"),
                        _format_interval(metrics, "changed_case_success"),
                        _format_interval(metrics, "invariant_retention"),
                        _format_interval(metrics, "old_rule_leakage"),
                        _format_interval(metrics, "total_requests"),
                        _format_interval(metrics, "total_tokens"),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "Only same-model, same-dataset, same-budget rows are directly comparable.",
            (
                "The legacy promotion-precision column is basis-dependent; use replay-estimated "
                "and realized columns plus realized coverage for claims."
            ),
            "Synthetic demo results validate plumbing and must not be reported as SOTA evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_mean(item: Mapping[str, Any], prefix: str) -> str:
    value = item.get(f"{prefix}_mean")
    spread = item.get(f"{prefix}_std")
    if not isinstance(value, (int, float)):
        return "N/A"
    if not isinstance(spread, (int, float)):
        spread = 0.0
    return f"{value:.4f}±{spread:.4f}"


def _format_interval(metrics: Any, name: str) -> str:
    if not isinstance(metrics, Mapping):
        return "N/A"
    interval = metrics.get(name)
    if not isinstance(interval, Mapping):
        return "N/A"
    mean = interval.get("delta_mean")
    low = interval.get("ci_low")
    high = interval.get("ci_high")
    if not isinstance(mean, (int, float)):
        return "N/A"
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return "N/A"
    return f"{float(mean):.4f} [{float(low):.4f}, {float(high):.4f}]"


__all__ = [
    "SweepSpec",
    "aggregate_sweep",
    "compare_sweep_runs",
    "compare_sweep_variants",
    "expand_sweep",
    "load_sweep_spec",
    "run_sweep",
]
