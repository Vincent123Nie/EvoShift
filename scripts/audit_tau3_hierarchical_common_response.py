#!/usr/bin/env python3
"""Audit the pre-registered Tau3 hierarchical common-response sweep."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

EXPECTED_SEEDS = {11, 22, 33, 44, 55}
VARIANTS = ("exact_shadow_eprocess", "hierarchical_shadow_eprocess")
EXPECTED_CACHE_PATH = "data/llm_cache_tau3_hierarchical_common_response.sqlite3"


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain an object")
    return payload


def _resolved_config(run_dir: Path, *, remove_flag: bool = False) -> dict[str, Any]:
    payload = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("resolved_config.yaml must contain an object")
    copied = json.loads(json.dumps(payload))
    if not isinstance(copied, dict):
        raise ValueError("resolved config copy must contain an object")
    if remove_flag:
        evolution = copied.get("evolution")
        if not isinstance(evolution, dict):
            raise ValueError("resolved config is missing evolution")
        evolution.pop("shadow_hierarchical_eprocess_enabled", None)
    return copied


def _predictions(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("prediction row must contain an object")
            rows.append(payload)
    return rows


def _paired_alignment(exact_dir: Path, hierarchical_dir: Path) -> dict[str, Any]:
    exact = _predictions(exact_dir)
    hierarchical = _predictions(hierarchical_dir)
    if len(exact) != len(hierarchical):
        return {
            "aligned": False,
            "n": min(len(exact), len(hierarchical)),
            "exact_n": len(exact),
            "hierarchical_n": len(hierarchical),
            "identical_output_score_n": 0,
            "identical_output_score_rate": 0.0,
            "first_divergence_index": 0,
            "hierarchical_cache_hit_n": 0,
        }
    sample_match = [
        left.get("sample", {}).get("sample_id") == right.get("sample", {}).get("sample_id")
        for left, right in zip(exact, hierarchical)
    ]
    output_match = [
        left.get("output") == right.get("output") and left.get("score") == right.get("score")
        for left, right in zip(exact, hierarchical)
    ]
    return {
        "aligned": all(sample_match),
        "n": len(exact),
        "exact_n": len(exact),
        "hierarchical_n": len(hierarchical),
        "identical_output_score_n": sum(output_match),
        "identical_output_score_rate": sum(output_match) / len(output_match)
        if output_match
        else 0.0,
        "first_divergence_index": next(
            (index for index, matched in enumerate(output_match) if not matched), None
        ),
        "hierarchical_cache_hit_n": sum(
            bool(row.get("usage", {}).get("cached")) for row in hierarchical
        ),
    }


def _metric(comparison: dict[str, Any], name: str) -> dict[str, Any]:
    value = comparison.get("metrics", {}).get(name, {})
    return value if isinstance(value, dict) else {}


def _paired_dataset_hashes_match(
    exact_manifest: dict[str, Any], hierarchical_manifest: dict[str, Any]
) -> bool:
    exact_hash = str(exact_manifest.get("dataset_hash", ""))
    hierarchical_hash = str(hierarchical_manifest.get("dataset_hash", ""))
    return bool(exact_hash) and exact_hash == hierarchical_hash


def _cluster_diagnostics(evolution: dict[str, Any]) -> dict[str, Any]:
    snapshot = evolution.get("shadow_cluster_eprocess", {})
    clusters = snapshot.get("clusters", {}) if isinstance(snapshot, dict) else {}
    rows = [row for row in clusters.values() if isinstance(row, dict)]
    opportunity_n = sum(int(row.get("opportunities", 0) or 0) for row in rows)
    e_values = [float(row["shadow_e_value"]) for row in rows if row.get("shadow_e_value")]
    inferred_match_n = sum(
        (int(row.get("opportunities", 0) or 0) + math.log(float(row["shadow_e_value"]), 3)) / 2.0
        for row in rows
        if row.get("shadow_e_value")
    )
    return {
        "cluster_n": len(rows),
        "opportunity_n": opportunity_n,
        "inferred_match_n": inferred_match_n,
        "inferred_match_rate": inferred_match_n / opportunity_n if opportunity_n else None,
        "min_e_value": min(e_values) if e_values else None,
        "max_e_value": max(e_values) if e_values else None,
    }


def audit(sweep_dir: Path) -> dict[str, Any]:
    matrix = _json_object(sweep_dir / "matrix.json")
    rows = [row for row in matrix.get("runs", []) if isinstance(row, dict)]
    grouped = {(str(row.get("variant")), int(row.get("seed", 0))): row for row in rows}
    pairs: list[dict[str, Any]] = []
    commits: set[str] = set()
    models: set[str] = set()
    dirty: list[bool] = []
    cache_paths: set[str] = set()
    cache_enabled: list[bool] = []
    pair_configs_equal: list[bool] = []
    pair_flags_valid: list[bool] = []
    pair_datasets_equal: list[bool] = []
    for seed in sorted(EXPECTED_SEEDS):
        exact_row = grouped.get((VARIANTS[0], seed))
        hierarchical_row = grouped.get((VARIANTS[1], seed))
        if exact_row is None or hierarchical_row is None:
            continue
        exact_dir = Path(str(exact_row["run_dir"]))
        hierarchical_dir = Path(str(hierarchical_row["run_dir"]))
        exact_manifest = _json_object(exact_dir / "manifest.json")
        hierarchical_manifest = _json_object(hierarchical_dir / "manifest.json")
        exact_metrics = _json_object(exact_dir / "metrics.json")
        hierarchical_metrics = _json_object(hierarchical_dir / "metrics.json")
        exact_config = _resolved_config(exact_dir)
        hierarchical_config = _resolved_config(hierarchical_dir)
        pair_configs_equal.append(
            _resolved_config(exact_dir, remove_flag=True)
            == _resolved_config(hierarchical_dir, remove_flag=True)
        )
        pair_flags_valid.append(
            exact_config.get("evolution", {}).get("shadow_hierarchical_eprocess_enabled") is False
            and hierarchical_config.get("evolution", {}).get("shadow_hierarchical_eprocess_enabled")
            is True
        )
        pair_datasets_equal.append(
            _paired_dataset_hashes_match(exact_manifest, hierarchical_manifest)
        )
        for manifest in (exact_manifest, hierarchical_manifest):
            commits.add(str(manifest.get("git_commit", "")))
            models.add(str(manifest.get("model", "")))
            dirty.append(bool(manifest.get("git_dirty", True)))
        for config in (exact_config, hierarchical_config):
            storage = config.get("storage", {})
            cache_paths.add(str(storage.get("cache_path", "")))
            cache_enabled.append(bool(storage.get("cache_enabled")))
        evolution = hierarchical_metrics.get("evolution", {})
        safety = hierarchical_metrics.get("active_memory_governance", {})
        cluster_diagnostics = _cluster_diagnostics(evolution)
        pairs.append(
            {
                "seed": seed,
                "exact_run_id": exact_row.get("run_id"),
                "hierarchical_run_id": hierarchical_row.get("run_id"),
                "alignment": _paired_alignment(exact_dir, hierarchical_dir),
                "exact_score": exact_metrics.get("overall", {}).get("mean_score"),
                "hierarchical_score": hierarchical_metrics.get("overall", {}).get("mean_score"),
                "exact_changed_success": exact_metrics.get("policy_shift", {}).get(
                    "changed_case_success_rate"
                ),
                "hierarchical_changed_success": hierarchical_metrics.get("policy_shift", {}).get(
                    "changed_case_success_rate"
                ),
                "cluster_crossings": evolution.get("shadow_cluster_eprocess_crossings"),
                "exact_opportunities": evolution.get("shadow_eprocess_opportunities"),
                "cluster_opportunities": evolution.get("shadow_cluster_eprocess_opportunities"),
                "shadow_failure_extractions": evolution.get("shadow_failure_extractions"),
                "cluster_diagnostics": cluster_diagnostics,
                "shadow_replays": evolution.get("shadow_only_candidate_replay_attempts"),
                "probations": evolution.get("shadow_candidate_probations"),
                "activations": evolution.get("shadow_candidate_activations"),
                "rejections": evolution.get("shadow_candidate_rejections"),
                "harmful_exposure": safety.get("harmful_active_memory_exposure_n"),
            }
        )
    comparisons = [item for item in matrix.get("comparisons", []) if isinstance(item, dict)]
    if len(comparisons) != 1:
        raise ValueError(f"expected one exact/hierarchical comparison, found {len(comparisons)}")
    comparison = comparisons[0]
    score = _metric(comparison, "score")
    changed = _metric(comparison, "changed_case_success")
    invariant = _metric(comparison, "invariant_retention")
    leakage = _metric(comparison, "old_rule_leakage")
    premature = _metric(comparison, "premature_update")
    harmful = _metric(comparison, "harmful_active_memory_exposure")
    false_retirement = _metric(comparison, "false_retirement")
    protocol = {
        "completed_pair_n": len(pairs),
        "all_pairs_completed": len(pairs) == len(EXPECTED_SEEDS),
        "all_samples_aligned": all(pair["alignment"].get("aligned") for pair in pairs),
        "same_clean_git_commit": len(commits) == 1 and "" not in commits and not any(dirty),
        "paired_dataset_hashes_match": all(pair_datasets_equal),
        "same_model": len(models) == 1 and "" not in models,
        "only_hierarchical_flag_differs": all(pair_configs_equal),
        "hierarchical_flag_assignment_valid": all(pair_flags_valid),
        "cache_enabled": bool(cache_enabled) and all(cache_enabled),
        "expected_cache_path": cache_paths == {EXPECTED_CACHE_PATH},
        "resource_claims_allowed": False,
    }
    protocol["valid"] = all(
        value
        for key, value in protocol.items()
        if key not in {"completed_pair_n", "resource_claims_allowed"}
    ) and protocol["completed_pair_n"] == len(EXPECTED_SEEDS)
    total_crossings = sum(int(pair["cluster_crossings"] or 0) for pair in pairs)
    total_replays = sum(int(pair["shadow_replays"] or 0) for pair in pairs)
    total_cluster_opportunities = sum(int(pair["cluster_opportunities"] or 0) for pair in pairs)
    total_inferred_matches = sum(
        float(pair["cluster_diagnostics"]["inferred_match_n"] or 0.0) for pair in pairs
    )
    cluster_e_values = [
        float(value)
        for pair in pairs
        for value in (
            pair["cluster_diagnostics"]["min_e_value"],
            pair["cluster_diagnostics"]["max_e_value"],
        )
        if value is not None
    ]
    gates = {
        "score": float(score.get("delta_mean", 0.0)) >= 0.03
        and float(score.get("ci_low", -1.0)) >= 0.0,
        "changed_success": float(changed.get("delta_mean", 0.0)) >= 0.03
        and float(changed.get("ci_low", -1.0)) >= 0.0,
        "invariant_retention": float(invariant.get("ci_low", -1.0)) >= -0.02,
        "old_rule_leakage": float(leakage.get("ci_high", 1.0)) <= 0.02,
        "premature_update": float(premature.get("ci_high", 1.0)) <= 0.02,
        "harmful_exposure": float(harmful.get("ci_high", 1.0)) <= 0.02,
        "false_retirement": float(false_retirement.get("ci_high", 1.0)) <= 0.02,
        "mechanism_reached": total_crossings >= 1 and total_replays >= 1,
    }
    return {
        "sweep_dir": str(sweep_dir),
        "interpretation": "public_source_capability_control",
        "not_applicable_metrics": {
            "poison_persistence_error": (
                "No attack feedback is injected in this clean-feedback capability control."
            )
        },
        "protocol": protocol,
        "gates": {**gates, "all_passed": bool(protocol["valid"]) and all(gates.values())},
        "comparison": comparison,
        "pairs": pairs,
        "mechanism": {
            "shadow_failure_extractions": sum(
                int(pair["shadow_failure_extractions"] or 0) for pair in pairs
            ),
            "exact_opportunities": sum(int(pair["exact_opportunities"] or 0) for pair in pairs),
            "cluster_opportunities": total_cluster_opportunities,
            "cluster_count": sum(
                int(pair["cluster_diagnostics"]["cluster_n"] or 0) for pair in pairs
            ),
            "inferred_cluster_matches": total_inferred_matches,
            "inferred_cluster_match_rate": (
                total_inferred_matches / total_cluster_opportunities
                if total_cluster_opportunities
                else None
            ),
            "min_cluster_e_value": min(cluster_e_values) if cluster_e_values else None,
            "max_cluster_e_value": max(cluster_e_values) if cluster_e_values else None,
            "cluster_crossings": total_crossings,
            "shadow_replays": total_replays,
            "probations": sum(int(pair["probations"] or 0) for pair in pairs),
            "activations": sum(int(pair["activations"] or 0) for pair in pairs),
            "rejections": sum(int(pair["rejections"] or 0) for pair in pairs),
            "harmful_exposure": sum(int(pair["harmful_exposure"] or 0) for pair in pairs),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    report = audit(args.sweep_dir)
    output = args.sweep_dir / "tau3_hierarchical_common_response_audit.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
