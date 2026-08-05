from __future__ import annotations

import csv
import itertools
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import yaml

from evoshift.benchmarks import create_benchmark
from evoshift.config import load_config
from evoshift.runner import EvoShiftRunner
from evoshift.schemas import Algorithm

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
                "confirmed_context_changes": trust_model.get("confirmed_context_changes"),
            }
        )
    aggregates = aggregate_sweep(rows)
    (destination / "matrix.json").write_text(
        json.dumps({"runs": rows, "aggregates": aggregates}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(destination / "matrix.csv", rows)
    _write_sweep_markdown(destination / "report.md", aggregates)
    return destination


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
        "confirmed_context_changes",
        "run_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            encoded["parameters"] = json.dumps(row["parameters"], sort_keys=True)
            writer.writerow(encoded)


def _write_sweep_markdown(path: Path, aggregates: Iterable[Mapping[str, Any]]) -> None:
    lines = [
        "# EvoShift sweep report",
        "",
        (
            "| Algorithm | Variant | Parameters | Seeds | Score | Changed | Old leakage | "
            "Invariant | Premature | Attack follow | Corrupt quarantine | Promotion precision | "
            "Replay-est. precision | Realized precision | Realized coverage | Audit rollback | "
            "Harmful promotion | False rollback | Rollback observations | Recovery | Requests | "
            "Tokens |"
        ),
        (
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---:|---:|---:|---:|---:|"
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
                    _format_mean(item, "future_audit_mean_rollback_observations"),
                    _format_mean(item, "mean_recovery_steps"),
                    f"{item.get('total_requests_mean', 0.0):.1f}",
                    f"{item.get('total_tokens_mean', 0.0):.1f}",
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


__all__ = ["SweepSpec", "aggregate_sweep", "expand_sweep", "load_sweep_spec", "run_sweep"]
