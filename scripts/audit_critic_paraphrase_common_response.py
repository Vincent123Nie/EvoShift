#!/usr/bin/env python3
"""Audit paired runs that share a content-addressed model-response cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED_VARIANTS = ("exact_shadow_eprocess", "hierarchical_shadow_eprocess")
EXPECTED_SEEDS = {11, 22, 33, 44, 55}
EXPECTED_CACHE_PATH = "data/llm_cache_critic_paraphrase_common_response.sqlite3"


def _matrix_rows(sweep_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads((sweep_dir / "matrix.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
        raise ValueError("matrix.json must contain a runs list")
    return [row for row in payload["runs"] if isinstance(row, dict)]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain an object")
    return payload


def _read_predictions(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("prediction row must be an object")
        rows.append(payload)
    return rows


def _resolved_config(run_dir: Path) -> dict[str, Any]:
    payload = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("resolved_config.yaml must contain an object")
    return payload


def _config_without_flag(run_dir: Path) -> dict[str, Any]:
    payload = _resolved_config(run_dir)
    copied = json.loads(json.dumps(payload))
    if not isinstance(copied, dict):
        raise ValueError("resolved config copy must contain an object")
    evolution = copied.get("evolution")
    if not isinstance(evolution, dict):
        raise ValueError("resolved config is missing evolution")
    evolution.pop("shadow_hierarchical_eprocess_enabled", None)
    return copied


def _foreground_match(
    exact: list[dict[str, Any]], hierarchical: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(exact) != len(hierarchical):
        return {
            "n": min(len(exact), len(hierarchical)),
            "exact_n": len(exact),
            "hierarchical_n": len(hierarchical),
            "sample_ids_aligned": False,
            "identical_output_score_n": 0,
            "cache_hit_n": 0,
            "first_divergence_index": 0,
        }
    aligned = all(
        left.get("sample", {}).get("sample_id") == right.get("sample", {}).get("sample_id")
        for left, right in zip(exact, hierarchical)
    )
    identical = [
        left.get("output") == right.get("output") and left.get("score") == right.get("score")
        for left, right in zip(exact, hierarchical)
    ]
    divergence = next((index for index, same in enumerate(identical) if not same), None)
    hierarchical_cache_hits = sum(bool(row.get("usage", {}).get("cached")) for row in hierarchical)
    exact_cache_hits = sum(bool(row.get("usage", {}).get("cached")) for row in exact)
    return {
        "n": len(exact),
        "exact_n": len(exact),
        "hierarchical_n": len(hierarchical),
        "sample_ids_aligned": aligned,
        "identical_output_score_n": sum(identical),
        "identical_output_score_rate": sum(identical) / len(identical) if identical else 0.0,
        "first_divergence_index": divergence,
        "exact_cache_hit_n": exact_cache_hits,
        "hierarchical_cache_hit_n": hierarchical_cache_hits,
        "hierarchical_cache_hit_rate": hierarchical_cache_hits / len(hierarchical)
        if hierarchical
        else 0.0,
    }


def audit(sweep_dir: Path) -> dict[str, Any]:
    rows = _matrix_rows(sweep_dir)
    grouped: dict[tuple[str, int], dict[str, Any]] = {
        (str(row.get("variant")), int(row.get("seed", 0))): row for row in rows
    }
    seeds = sorted({seed for variant, seed in grouped if variant in REQUIRED_VARIANTS})
    pairs: list[dict[str, Any]] = []
    pair_config_equal: list[bool] = []
    pair_flag_assignments_valid: list[bool] = []
    commits: set[str] = set()
    datasets: set[str] = set()
    models: set[str] = set()
    dirty: list[bool] = []
    cache_enabled: list[bool] = []
    cache_paths: set[str] = set()
    for seed in seeds:
        exact_row = grouped.get((REQUIRED_VARIANTS[0], seed))
        hierarchical_row = grouped.get((REQUIRED_VARIANTS[1], seed))
        if exact_row is None or hierarchical_row is None:
            continue
        exact_dir = Path(str(exact_row["run_dir"]))
        hierarchical_dir = Path(str(hierarchical_row["run_dir"]))
        exact_manifest = _read_json(exact_dir / "manifest.json")
        hierarchical_manifest = _read_json(hierarchical_dir / "manifest.json")
        exact_metrics = _read_json(exact_dir / "metrics.json")
        hierarchical_metrics = _read_json(hierarchical_dir / "metrics.json")
        exact_costs = _read_json(exact_dir / "costs.json")
        hierarchical_costs = _read_json(hierarchical_dir / "costs.json")
        exact_config = _config_without_flag(exact_dir)
        hierarchical_config = _config_without_flag(hierarchical_dir)
        for config in (_resolved_config(exact_dir), _resolved_config(hierarchical_dir)):
            storage = config.get("storage", {})
            if not isinstance(storage, dict):
                raise ValueError("resolved config is missing storage")
            cache_enabled.append(bool(storage.get("cache_enabled")))
            cache_paths.add(str(storage.get("cache_path", "")))
        pair_config_equal.append(exact_config == hierarchical_config)
        exact_raw_config = _resolved_config(exact_dir)
        hierarchical_raw_config = _resolved_config(hierarchical_dir)
        exact_flag = exact_raw_config.get("evolution", {}).get(
            "shadow_hierarchical_eprocess_enabled"
        )
        hierarchical_flag = hierarchical_raw_config.get("evolution", {}).get(
            "shadow_hierarchical_eprocess_enabled"
        )
        pair_flag_assignments_valid.append(exact_flag is False and hierarchical_flag is True)
        for manifest in (exact_manifest, hierarchical_manifest):
            commits.add(str(manifest.get("git_commit", "")))
            datasets.add(str(manifest.get("dataset_hash", "")))
            models.add(str(manifest.get("model", "")))
            dirty.append(bool(manifest.get("git_dirty", True)))
        match = _foreground_match(_read_predictions(exact_dir), _read_predictions(hierarchical_dir))
        pairs.append(
            {
                "seed": seed,
                "exact_run_id": exact_row.get("run_id"),
                "hierarchical_run_id": hierarchical_row.get("run_id"),
                "match": match,
                "exact_score": exact_metrics.get("overall", {}).get("mean_score"),
                "hierarchical_score": hierarchical_metrics.get("overall", {}).get("mean_score"),
                "score_delta": (
                    hierarchical_metrics.get("overall", {}).get("mean_score", 0.0) or 0.0
                )
                - (exact_metrics.get("overall", {}).get("mean_score", 0.0) or 0.0),
                "exact_changed_score": exact_metrics.get("phases", {})
                .get("fixture_policy_change", {})
                .get("mean_score"),
                "hierarchical_changed_score": hierarchical_metrics.get("phases", {})
                .get("fixture_policy_change", {})
                .get("mean_score"),
                "exact_protected_score": exact_metrics.get("phases", {})
                .get("fixture_safe_anchor", {})
                .get("mean_score"),
                "hierarchical_protected_score": hierarchical_metrics.get("phases", {})
                .get("fixture_safe_anchor", {})
                .get("mean_score"),
                "exact_cluster_crossings": exact_metrics.get("evolution", {}).get(
                    "shadow_cluster_eprocess_crossings"
                ),
                "hierarchical_cluster_crossings": hierarchical_metrics.get("evolution", {}).get(
                    "shadow_cluster_eprocess_crossings"
                ),
                "hierarchical_shadow_replays": hierarchical_metrics.get("evolution", {}).get(
                    "shadow_only_candidate_replay_attempts"
                ),
                "hierarchical_probations": hierarchical_metrics.get("evolution", {}).get(
                    "shadow_candidate_probations"
                ),
                "hierarchical_activations": hierarchical_metrics.get("evolution", {}).get(
                    "shadow_candidate_activations"
                ),
                "hierarchical_rejections": hierarchical_metrics.get("evolution", {}).get(
                    "shadow_candidate_rejections"
                ),
                "hierarchical_future_audit_confirmed": hierarchical_metrics.get(
                    "future_audit", {}
                ).get("confirmed"),
                "hierarchical_future_audit_expired": hierarchical_metrics.get(
                    "future_audit", {}
                ).get("expired"),
                "hierarchical_harmful_exposure": hierarchical_metrics.get(
                    "active_memory_governance", {}
                ).get("harmful_active_memory_exposure_n"),
                "exact_requests": exact_costs.get("requests"),
                "hierarchical_requests": hierarchical_costs.get("requests"),
            }
        )
    all_matches = [pair["match"] for pair in pairs]
    matching_n = sum(int(match.get("identical_output_score_n", 0)) for match in all_matches)
    total_n = sum(int(match.get("n", 0)) for match in all_matches)
    protocol = {
        "expected_seeds": set(seeds) == EXPECTED_SEEDS,
        "completed_pair_n": len(pairs),
        "all_pairs_completed": len(pairs) == len(EXPECTED_SEEDS),
        "same_clean_git_commit": len(commits) == 1 and "" not in commits and not any(dirty),
        "same_dataset": len(datasets) == 1 and "" not in datasets,
        "same_model": len(models) == 1 and "" not in models,
        "only_hierarchical_flag_differs": all(pair_config_equal),
        "hierarchical_flag_assignment_valid": all(pair_flag_assignments_valid),
        "cache_enabled": bool(cache_enabled) and all(cache_enabled),
        "expected_cache_path": cache_paths == {EXPECTED_CACHE_PATH},
        "resource_claims_allowed": False,
    }
    protocol["valid"] = all(
        value
        for key, value in protocol.items()
        if key not in {"completed_pair_n", "resource_claims_allowed"}
    ) and protocol["completed_pair_n"] == len(EXPECTED_SEEDS)
    return {
        "sweep_dir": str(sweep_dir),
        "interpretation": "capability_control_only",
        "protocol": protocol,
        "pairs": pairs,
        "pooled": {
            "foreground_n": total_n,
            "identical_output_score_n": matching_n,
            "identical_output_score_rate": matching_n / total_n if total_n else 0.0,
            "mean_score_delta": sum(float(pair["score_delta"]) for pair in pairs) / len(pairs)
            if pairs
            else 0.0,
            "mean_changed_score_delta": sum(
                float(pair["hierarchical_changed_score"] or 0.0)
                - float(pair["exact_changed_score"] or 0.0)
                for pair in pairs
            )
            / len(pairs)
            if pairs
            else 0.0,
            "mean_protected_score_delta": sum(
                float(pair["hierarchical_protected_score"] or 0.0)
                - float(pair["exact_protected_score"] or 0.0)
                for pair in pairs
            )
            / len(pairs)
            if pairs
            else 0.0,
            "cluster_crossings": sum(
                int(pair["hierarchical_cluster_crossings"] or 0) for pair in pairs
            ),
            "shadow_replays": sum(int(pair["hierarchical_shadow_replays"] or 0) for pair in pairs),
            "probations": sum(int(pair["hierarchical_probations"] or 0) for pair in pairs),
            "activations": sum(int(pair["hierarchical_activations"] or 0) for pair in pairs),
            "rejections": sum(int(pair["hierarchical_rejections"] or 0) for pair in pairs),
            "future_audit_confirmed": sum(
                int(pair["hierarchical_future_audit_confirmed"] or 0) for pair in pairs
            ),
            "future_audit_expired": sum(
                int(pair["hierarchical_future_audit_expired"] or 0) for pair in pairs
            ),
            "harmful_exposure": sum(
                int(pair["hierarchical_harmful_exposure"] or 0) for pair in pairs
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    report = audit(args.sweep_dir)
    output = args.sweep_dir / "critic_paraphrase_common_response_audit.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
