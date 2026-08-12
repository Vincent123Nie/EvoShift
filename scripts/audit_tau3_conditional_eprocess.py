#!/usr/bin/env python3
"""Audit a pre-registered three-way Tau3 conditional e-process sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from evoshift.sweep import compare_sweep_variants

VARIANTS = (
    "exact_shadow_eprocess",
    "hierarchical_shadow_eprocess",
    "conditional_shadow_eprocess",
)
STAGES = {
    "development": {
        "seeds": {66, 77},
        "cache_path": "data/llm_cache_tau3_conditional_eprocess_development.sqlite3",
    },
    "confirmation": {
        "seeds": {88, 99, 111},
        "cache_path": "data/llm_cache_tau3_conditional_eprocess_confirmation.sqlite3",
    },
}


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain an object")
    return payload


def _yaml_object(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain an object")
    return payload


def _predictions(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _alignment(reference_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    reference = _predictions(reference_dir)
    candidate = _predictions(candidate_dir)
    reference_ids = [row.get("sample", {}).get("sample_id") for row in reference]
    candidate_ids = [row.get("sample", {}).get("sample_id") for row in candidate]
    return {
        "aligned": reference_ids == candidate_ids,
        "reference_n": len(reference),
        "candidate_n": len(candidate),
        "identical_output_score_n": sum(
            left.get("output") == right.get("output") and left.get("score") == right.get("score")
            for left, right in zip(reference, candidate)
        ),
        "first_divergence_index": next(
            (
                index
                for index, (left, right) in enumerate(zip(reference, candidate))
                if left.get("output") != right.get("output")
                or left.get("score") != right.get("score")
            ),
            None,
        ),
    }


def _normalized_config(run_dir: Path) -> dict[str, Any]:
    config = _yaml_object(run_dir / "resolved_config.yaml")
    copied = json.loads(json.dumps(config))
    evolution = copied.get("evolution")
    if not isinstance(evolution, dict):
        raise ValueError("resolved config is missing evolution")
    evolution.pop("shadow_hierarchical_eprocess_enabled", None)
    evolution.pop("shadow_conditional_eprocess_enabled", None)
    return copied


def _mode_flags(run_dir: Path) -> tuple[bool, bool]:
    evolution = _yaml_object(run_dir / "resolved_config.yaml").get("evolution", {})
    if not isinstance(evolution, dict):
        raise ValueError("resolved config is missing evolution")
    return (
        bool(evolution.get("shadow_hierarchical_eprocess_enabled")),
        bool(evolution.get("shadow_conditional_eprocess_enabled")),
    )


def _metric(comparison: dict[str, Any], name: str) -> dict[str, Any]:
    metrics = comparison.get("metrics", {})
    value = metrics.get(name, {}) if isinstance(metrics, dict) else {}
    return value if isinstance(value, dict) else {}


def _find_comparison(
    comparisons: list[dict[str, Any]],
    *,
    baseline_variant: str,
) -> dict[str, Any]:
    expected_candidate = f"evoshift:{VARIANTS[2]}"
    expected_baseline = f"evoshift:{baseline_variant}"
    return next(
        (
            item
            for item in comparisons
            if item.get("candidate_algorithm") == expected_candidate
            and item.get("baseline_algorithm") == expected_baseline
        ),
        {},
    )


def _comparison_gates(comparison: dict[str, Any]) -> dict[str, bool]:
    score = _metric(comparison, "score")
    changed = _metric(comparison, "changed_case_success")
    invariant = _metric(comparison, "invariant_retention")
    leakage = _metric(comparison, "old_rule_leakage")
    premature = _metric(comparison, "premature_update")
    harmful = _metric(comparison, "harmful_active_memory_exposure")
    false_retirement = _metric(comparison, "false_retirement")
    return {
        "score_noninferior": float(score.get("delta_mean", -1.0)) >= 0.0,
        "changed_success_noninferior": float(changed.get("delta_mean", -1.0)) >= 0.0,
        "invariant_retention": float(invariant.get("ci_low", -1.0)) >= -0.02,
        "old_rule_leakage": float(leakage.get("ci_high", 1.0)) <= 0.02,
        "premature_update": float(premature.get("ci_high", 1.0)) <= 0.02,
        "harmful_exposure": float(harmful.get("ci_high", 1.0)) <= 0.02,
        "false_retirement": float(false_retirement.get("ci_high", 1.0)) <= 0.02,
    }


def audit(sweep_dir: Path, *, stage: str) -> dict[str, Any]:
    stage_config = STAGES[stage]
    expected_seeds = set(stage_config["seeds"])
    expected_cache_path = str(stage_config["cache_path"])
    matrix = _json_object(sweep_dir / "matrix.json")
    state = _json_object(sweep_dir / "sweep_state.json")
    rows = [row for row in matrix.get("runs", []) if isinstance(row, dict)]
    grouped = {(str(row.get("variant")), int(row.get("seed", -1))): row for row in rows}

    commits: set[str] = set()
    models: set[str] = set()
    dirty: list[bool] = []
    cache_paths: set[str] = set()
    cache_enabled: list[bool] = []
    dataset_hashes: dict[int, set[str]] = {seed: set() for seed in expected_seeds}
    configs_equal: list[bool] = []
    flags_valid: list[bool] = []
    alignments: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []

    expected_flags = {
        VARIANTS[0]: (False, False),
        VARIANTS[1]: (True, False),
        VARIANTS[2]: (True, True),
    }
    for seed in sorted(expected_seeds):
        seed_rows = {variant: grouped.get((variant, seed)) for variant in VARIANTS}
        if any(row is None for row in seed_rows.values()):
            continue
        run_dirs = {
            variant: Path(str(row["run_dir"]))
            for variant, row in seed_rows.items()
            if row is not None
        }
        reference_dir = run_dirs[VARIANTS[0]]
        normalized_configs = [_normalized_config(run_dirs[variant]) for variant in VARIANTS]
        configs_equal.append(all(config == normalized_configs[0] for config in normalized_configs))
        flags_valid.extend(
            _mode_flags(run_dirs[variant]) == expected_flags[variant] for variant in VARIANTS
        )
        for variant in VARIANTS[1:]:
            alignments.append(
                {
                    "seed": seed,
                    "reference": VARIANTS[0],
                    "candidate": variant,
                    **_alignment(reference_dir, run_dirs[variant]),
                }
            )
        for variant in VARIANTS:
            run_dir = run_dirs[variant]
            manifest = _json_object(run_dir / "manifest.json")
            metrics = _json_object(run_dir / "metrics.json")
            config = _yaml_object(run_dir / "resolved_config.yaml")
            storage = config.get("storage", {})
            commits.add(str(manifest.get("git_commit", "")))
            models.add(str(manifest.get("model", "")))
            dirty.append(bool(manifest.get("git_dirty", True)))
            dataset_hashes[seed].add(str(manifest.get("dataset_hash", "")))
            if isinstance(storage, dict):
                cache_paths.add(str(storage.get("cache_path", "")))
                cache_enabled.append(bool(storage.get("cache_enabled")))
            evolution = metrics.get("evolution", {})
            future_audit = metrics.get("future_audit", {})
            governance = metrics.get("active_memory_governance", {})
            runs.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "run_id": seed_rows[variant].get("run_id") if seed_rows[variant] else None,
                    "score": metrics.get("overall", {}).get("mean_score"),
                    "changed_success": metrics.get("policy_shift", {}).get(
                        "changed_case_success_rate"
                    ),
                    "cluster_opportunities": evolution.get("shadow_cluster_eprocess_opportunities"),
                    "skipped_unrelated": evolution.get("shadow_cluster_eprocess_skipped_unrelated"),
                    "cluster_crossings": evolution.get("shadow_cluster_eprocess_crossings"),
                    "shadow_replays": evolution.get("shadow_only_candidate_replay_attempts"),
                    "probations": evolution.get("shadow_candidate_probations"),
                    "activations": evolution.get("shadow_candidate_activations"),
                    "future_audit_confirmed": future_audit.get("confirmed"),
                    "harmful_exposure": governance.get("harmful_active_memory_exposure_n"),
                }
            )

    comparison_kwargs = {
        "variant_parameters": {
            VARIANTS[0]: {
                "evolution.shadow_hierarchical_eprocess_enabled": False,
                "evolution.shadow_conditional_eprocess_enabled": False,
            },
            VARIANTS[1]: {
                "evolution.shadow_hierarchical_eprocess_enabled": True,
                "evolution.shadow_conditional_eprocess_enabled": False,
            },
            VARIANTS[2]: {
                "evolution.shadow_hierarchical_eprocess_enabled": True,
                "evolution.shadow_conditional_eprocess_enabled": True,
            },
        },
        "samples": 5000,
        "confidence": 0.95,
    }
    comparisons = compare_sweep_variants(
        rows,
        baseline_variant=VARIANTS[0],
        **comparison_kwargs,
    )
    comparisons.extend(
        compare_sweep_variants(
            rows,
            baseline_variant=VARIANTS[1],
            **comparison_kwargs,
        )
    )
    exact_comparison = _find_comparison(comparisons, baseline_variant=VARIANTS[0])
    hierarchical_comparison = _find_comparison(comparisons, baseline_variant=VARIANTS[1])
    exact_gates = _comparison_gates(exact_comparison)
    hierarchical_gates = _comparison_gates(hierarchical_comparison)
    conditional_runs = [run for run in runs if run["variant"] == VARIANTS[2]]
    total_crossings = sum(int(run["cluster_crossings"] or 0) for run in conditional_runs)
    total_replays = sum(int(run["shadow_replays"] or 0) for run in conditional_runs)
    total_activations = sum(int(run["activations"] or 0) for run in conditional_runs)
    total_confirmed = sum(int(run["future_audit_confirmed"] or 0) for run in conditional_runs)
    total_harmful_exposure = sum(int(run["harmful_exposure"] or 0) for run in conditional_runs)

    protocol = {
        "matrix_complete": matrix.get("complete") is True,
        "state_complete": state.get("status") == "complete",
        "completed_run_n": len(rows),
        "expected_run_n": len(expected_seeds) * len(VARIANTS),
        "all_samples_aligned": bool(alignments) and all(item.get("aligned") for item in alignments),
        "paired_dataset_hashes_match": all(
            len(hashes) == 1 and "" not in hashes for hashes in dataset_hashes.values()
        ),
        "same_clean_git_commit": len(commits) == 1 and "" not in commits and not any(dirty),
        "same_model": len(models) == 1 and "" not in models,
        "only_mode_flags_differ": bool(configs_equal) and all(configs_equal),
        "mode_flag_assignment_valid": bool(flags_valid) and all(flags_valid),
        "cache_enabled": bool(cache_enabled) and all(cache_enabled),
        "expected_cache_path": cache_paths == {expected_cache_path},
        "resource_claims_allowed": False,
    }
    protocol["valid"] = (
        all(
            value
            for key, value in protocol.items()
            if key
            not in {
                "completed_run_n",
                "expected_run_n",
                "resource_claims_allowed",
                "valid",
            }
        )
        and protocol["completed_run_n"] == protocol["expected_run_n"]
    )
    gates = {
        "mechanism_reached": total_crossings >= 1 and total_replays >= 1,
        "vs_exact": exact_gates,
        "vs_hierarchical": hierarchical_gates,
        "zero_harmful_exposure": total_harmful_exposure == 0,
        "no_unverified_activation": total_activations <= total_confirmed,
    }
    gates["all_passed"] = (
        bool(protocol["valid"])
        and bool(gates["mechanism_reached"])
        and all(exact_gates.values())
        and all(hierarchical_gates.values())
        and bool(gates["zero_harmful_exposure"])
        and bool(gates["no_unverified_activation"])
    )
    return {
        "sweep_dir": str(sweep_dir),
        "stage": stage,
        "interpretation": "public_source_clean_feedback_capability_control",
        "protocol": protocol,
        "gates": gates,
        "mechanism": {
            "conditional_cluster_crossings": total_crossings,
            "conditional_shadow_replays": total_replays,
            "conditional_activations": total_activations,
            "conditional_future_audit_confirmed": total_confirmed,
            "conditional_harmful_exposure": total_harmful_exposure,
        },
        "comparisons": {
            "conditional_vs_exact": exact_comparison,
            "conditional_vs_hierarchical": hierarchical_comparison,
        },
        "alignments": alignments,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    args = parser.parse_args()
    report = audit(args.sweep_dir, stage=args.stage)
    output = args.sweep_dir / f"tau3_conditional_eprocess_{args.stage}_audit.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
