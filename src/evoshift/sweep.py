from __future__ import annotations

import csv
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
    "trusted_candidate_shadow_cooldown_bypasses": ("trusted_candidate_shadow_cooldown_bypasses"),
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
    "harmful_active_memory_exposure_n": "harmful_active_memory_exposure",
    "stale_memory_retention_rate": "stale_memory_retention",
    "selective_forgetting_precision": "selective_forgetting_precision",
    "selective_forgetting_recall": "selective_forgetting_recall",
    "false_retirement_rate": "false_retirement",
    "correct_reacquisition_rate": "correct_reacquisition",
    "memory_reacquisitions": "memory_reacquisitions",
    "active_audit_control_requests": "active_audit_control_requests",
    "counterfactual_audit_coverage": "counterfactual_audit_coverage",
    "active_audit_budget_utilization": "active_audit_budget_utilization",
    "active_audit_mean_retirement_latency": "active_audit_mean_retirement_latency",
    "early_causal_retirements": "early_causal_retirements",
    "early_causal_retirement_precision": "early_causal_retirement_precision",
    "early_causal_false_retirement_rate": "early_causal_false_retirement_rate",
    "confirmed_context_changes": "confirmed_context_changes",
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


async def run_sweep(spec: SweepSpec, root: Path) -> Path:
    sweep_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = root / "runs" / "sweeps" / sweep_id
    destination.mkdir(parents=True, exist_ok=False)
    rows: List[Dict[str, Any]] = []
    for assignment in expand_sweep(spec):
        overrides = [
            f"algorithm={assignment['algorithm']}",
            f"evaluation.seed={assignment['seed']}",
        ]
        overrides.extend(
            f"{key}={_override_value(value)}" for key, value in assignment["parameters"].items()
        )
        config = load_config(spec.base_config, overrides)
        adapter = create_benchmark(config.benchmark, root=root, seed=config.evaluation.seed)
        result = await EvoShiftRunner(config, adapter, workdir=root).run()
        budget = _read_total_budget(result.run_dir)
        policy_shift = result.metrics.get("policy_shift", {})
        feedback = result.metrics.get("feedback", {})
        evolution = result.metrics.get("evolution", {})
        future_audit = result.metrics.get("future_audit", {})
        active_governance = result.metrics.get("active_memory_governance", {})
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
                "trusted_candidate_shadow_cooldown_bypasses": evolution.get(
                    "trusted_candidate_shadow_cooldown_bypasses"
                ),
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
                "confirmed_context_changes": trust_model.get("confirmed_context_changes"),
            }
        )
    aggregates = aggregate_sweep(rows)
    analysis_config = load_config(spec.base_config)
    comparisons = compare_sweep_runs(
        rows,
        samples=analysis_config.evaluation.bootstrap_samples,
        confidence=analysis_config.evaluation.confidence_level,
    )
    (destination / "matrix.json").write_text(
        json.dumps(
            {"runs": rows, "aggregates": aggregates, "comparisons": comparisons},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_csv(destination / "matrix.csv", rows)
    _write_comparison_csv(destination / "comparisons.csv", comparisons)
    _write_sweep_markdown(destination / "report.md", aggregates, comparisons)
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
        "trusted_candidate_shadow_cooldown_bypasses",
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
        "harmful_active_memory_exposure_n",
        "stale_memory_retention_rate",
        "selective_forgetting_precision",
        "selective_forgetting_recall",
        "false_retirement_rate",
        "correct_reacquisition_rate",
        "memory_reacquisitions",
        "active_audit_control_requests",
        "counterfactual_audit_coverage",
        "active_audit_budget_utilization",
        "active_audit_mean_retirement_latency",
        "early_causal_retirements",
        "early_causal_retirement_precision",
        "early_causal_false_retirement_rate",
        "confirmed_context_changes",
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
    lines = [
        "# EvoShift sweep report",
        "",
        (
            "| Algorithm | Variant | Parameters | Seeds | Score | Changed | Old leakage | "
            "Invariant | Premature | Attack follow | Corrupt quarantine | Promotion precision | "
            "Replay-est. precision | Realized precision | Realized coverage | Audit rollback | "
            "Harmful promotion | False rollback | Harm exposure | Stale retention | Forget P | "
            "Forget R | False retire | Reacquire | Recovery | Requests | Tokens |"
        ),
        (
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]
    for item in aggregates:
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
                    f"{item.get('total_requests_mean', 0.0):.1f}",
                    f"{item.get('total_tokens_mean', 0.0):.1f}",
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
    "expand_sweep",
    "load_sweep_spec",
    "run_sweep",
]
