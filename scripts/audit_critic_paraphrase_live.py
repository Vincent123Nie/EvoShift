#!/usr/bin/env python3
"""Audit a bounded real-model critic-paraphrase mini without hidden labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load_rows(sweep_dir: Path) -> list[dict[str, Any]]:
    path = sweep_dir / "matrix.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
        raise ValueError("matrix.json must contain a runs list")
    return [row for row in payload["runs"] if isinstance(row, dict)]


def _run_metrics(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = Path(str(row["run_dir"]))
    return (
        json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")),
        json.loads((run_dir / "costs.json").read_text(encoding="utf-8")),
    )


def _load_comparison_config(row: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(str(row["run_dir"]))
    payload = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("resolved_config.yaml must contain a mapping")
    copied = json.loads(json.dumps(payload))
    if not isinstance(copied, dict):
        raise ValueError("resolved config copy must contain a mapping")
    evolution = copied.get("evolution")
    if not isinstance(evolution, dict):
        raise ValueError("resolved config is missing evolution settings")
    evolution.pop("shadow_hierarchical_eprocess_enabled", None)
    return copied


def audit(sweep_dir: Path) -> dict[str, Any]:
    rows = _load_rows(sweep_dir)
    by_variant = {str(row.get("variant")): row for row in rows}
    required = {"exact_shadow_eprocess", "hierarchical_shadow_eprocess"}
    missing = sorted(required - set(by_variant))
    if missing:
        raise ValueError(f"missing variants: {', '.join(missing)}")

    report: dict[str, Any] = {
        "sweep_dir": str(sweep_dir),
        "variants": {},
        "protocol": {
            "same_dataset_hash": True,
            "same_seed": True,
            "same_model": True,
            "cache_disabled": True,
            "only_hierarchical_flag_differs": True,
        },
        "interpretation": "diagnostic_only",
    }
    dataset_hashes: set[str] = set()
    seeds: set[int] = set()
    models: set[str] = set()
    for variant in sorted(required):
        row = by_variant[variant]
        metrics, costs = _run_metrics(row)
        dataset_hashes.add(str(metrics.get("dataset_hash", "")))
        seeds.add(int(row.get("seed", 0)))
        models.add(str(row.get("model", metrics.get("model", ""))))
        evolution = metrics.get("evolution", {})
        critic = metrics.get("critic", {})
        report["variants"][variant] = {
            "run_id": row.get("run_id"),
            "score": metrics.get("overall", {}).get("mean_score"),
            "changed_score": metrics.get("policy_shift", {}).get("changed_case_success_rate"),
            "critic_records": critic.get("records"),
            "critic_signature_unique": critic.get("signature_unique"),
            "critic_memory_trigger_unique": critic.get("memory_trigger_unique"),
            "critic_memory_directive_unique": critic.get("memory_directive_unique"),
            "critic_structured_fallbacks": critic.get("structured_fallbacks"),
            "shadow_eprocess_opportunities": evolution.get("shadow_eprocess_opportunities"),
            "shadow_eprocess_crossings": evolution.get("shadow_eprocess_crossings"),
            "shadow_cluster_eprocess_opportunities": evolution.get(
                "shadow_cluster_eprocess_opportunities"
            ),
            "shadow_cluster_eprocess_crossings": evolution.get("shadow_cluster_eprocess_crossings"),
            "shadow_only_replays": evolution.get("shadow_only_candidate_replay_attempts"),
            "probations": evolution.get("shadow_candidate_probations"),
            "activations": evolution.get("shadow_candidate_activations"),
            "harmful_promotion_rate": metrics.get("future_audit", {}).get("harmful_promotion_rate"),
            "harmful_active_memory_exposure": metrics.get("active_memory_governance", {}).get(
                "harmful_active_memory_exposure_n"
            ),
            "requests": costs.get("requests"),
            "total_tokens": costs.get("total_tokens"),
            "provider_failures": costs.get("provider_failures", 0),
        }

    report["protocol"]["same_dataset_hash"] = len(dataset_hashes) == 1 and "" not in dataset_hashes
    report["protocol"]["same_seed"] = len(seeds) == 1
    report["protocol"]["same_model"] = len(models) == 1 and "" not in models
    report["protocol"]["cache_disabled"] = all(
        (
            yaml.safe_load(
                (Path(str(row["run_dir"])) / "resolved_config.yaml").read_text(encoding="utf-8")
            )
            or {}
        )
        .get("storage", {})
        .get("cache_enabled")
        is False
        for row in by_variant.values()
    )
    report["protocol"]["only_hierarchical_flag_differs"] = (
        len(
            {
                json.dumps(_load_comparison_config(row), sort_keys=True, separators=(",", ":"))
                for row in by_variant.values()
            }
        )
        == 1
    )
    exact = report["variants"]["exact_shadow_eprocess"]
    hierarchical = report["variants"]["hierarchical_shadow_eprocess"]
    report["delta"] = {
        "score": (hierarchical["score"] or 0.0) - (exact["score"] or 0.0),
        "changed_score": (hierarchical["changed_score"] or 0.0) - (exact["changed_score"] or 0.0),
        "cluster_crossings": (hierarchical["shadow_cluster_eprocess_crossings"] or 0)
        - (exact["shadow_cluster_eprocess_crossings"] or 0),
        "requests": (hierarchical["requests"] or 0) - (exact["requests"] or 0),
        "total_tokens": (hierarchical["total_tokens"] or 0) - (exact["total_tokens"] or 0),
    }
    report["gate"] = {
        "protocol_valid": all(report["protocol"].values()),
        "observed_critic_diversity": (
            (hierarchical["critic_memory_trigger_unique"] or 0) >= 2
            or (hierarchical["critic_memory_directive_unique"] or 0) >= 2
        ),
        "structured_critic_parse_clean": (hierarchical["critic_structured_fallbacks"] or 0) == 0,
        "no_harmful_promotion": (hierarchical["harmful_promotion_rate"] or 0.0) == 0.0,
        "no_harmful_active_memory_exposure": (hierarchical["harmful_active_memory_exposure"] or 0)
        == 0,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    report = audit(args.sweep_dir)
    output = args.sweep_dir / "critic_paraphrase_live_audit.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
