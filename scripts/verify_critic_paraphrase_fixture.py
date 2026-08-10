#!/usr/bin/env python3
"""Verify the preregistered critic-paraphrase mechanism fixture."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _row_by_variant(matrix: Mapping[str, Any], variant: str) -> Mapping[str, Any]:
    rows = [row for row in matrix.get("runs", []) if row.get("variant") == variant]
    if len(rows) != 1:
        raise ValueError(f"expected one {variant!r} run, found {len(rows)}")
    return rows[0]


def _run_artifacts(row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = Path(str(row["run_dir"]))
    return _load_json(run_dir / "metrics.json"), _load_json(run_dir / "costs.json")


def verify(sweep_dir: Path, *, require_clean: bool = True) -> dict[str, Any]:
    matrix = _load_json(sweep_dir / "matrix.json")
    exact_row = _row_by_variant(matrix, "exact_shadow_eprocess")
    hierarchical_row = _row_by_variant(matrix, "hierarchical_shadow_eprocess")
    exact_metrics, exact_costs = _run_artifacts(exact_row)
    hierarchical_metrics, hierarchical_costs = _run_artifacts(hierarchical_row)
    exact_evolution = exact_metrics["evolution"]
    hierarchical_evolution = hierarchical_metrics["evolution"]

    _require(
        exact_metrics["dataset_hash"] == hierarchical_metrics["dataset_hash"],
        "variants used different benchmark samples",
    )
    _require(exact_evolution["shadow_eprocess_crossings"] == 0, "exact path crossed")
    _require(
        exact_evolution["shadow_only_candidate_replay_attempts"] == 0,
        "exact path unexpectedly reached shadow-only replay",
    )
    _require(
        exact_evolution["shadow_candidate_probations"] == 0,
        "exact path unexpectedly created shadow probation",
    )
    _require(
        hierarchical_evolution["shadow_cluster_eprocess_crossings"] >= 1,
        "hierarchical cluster did not cross",
    )
    _require(
        hierarchical_evolution["shadow_only_candidate_replay_attempts"] >= 1,
        "cluster crossing did not reach shadow-only replay",
    )
    _require(
        hierarchical_evolution["shadow_candidate_probations"] >= 1,
        "shadow candidate did not enter probation",
    )
    _require(
        hierarchical_evolution["shadow_candidate_activations"] >= 1,
        "shadow-derived probation did not activate",
    )
    _require(
        hierarchical_metrics["future_audit"]["confirmed"] >= 1,
        "future audit did not confirm the probationary memory",
    )
    _require(
        hierarchical_metrics["future_audit"]["harmful_promotion_rate"] == 0.0,
        "hierarchical path produced a harmful promotion",
    )
    _require(
        hierarchical_metrics["active_memory_governance"]["harmful_active_memory_exposure_n"] == 0,
        "hierarchical path exposed a harmful active memory",
    )
    _require(
        hierarchical_metrics["phases"]["fixture_safe_anchor"]["success_rate"] == 1.0,
        "safe anchor slice regressed",
    )
    exact_score = float(exact_metrics["overall"]["mean_score"])
    hierarchical_score = float(hierarchical_metrics["overall"]["mean_score"])
    _require(hierarchical_score > exact_score, "hierarchical score did not improve")
    _require(
        int(hierarchical_costs["requests"]) <= int(exact_costs["requests"]),
        "hierarchical path exceeded the exact request count",
    )
    _require(
        int(hierarchical_costs["total_tokens"]) <= int(exact_costs["total_tokens"]),
        "hierarchical path exceeded the exact token count",
    )

    if require_clean:
        for row in (exact_row, hierarchical_row):
            manifest = _load_json(Path(str(row["run_dir"])) / "manifest.json")
            _require(not bool(manifest["git_dirty"]), "evidence run used a dirty worktree")
            _require(manifest["git_commit"] != "unknown", "evidence run has no git commit")

    return {
        "status": "passed",
        "dataset_hash": exact_metrics["dataset_hash"],
        "exact_score": exact_score,
        "hierarchical_score": hierarchical_score,
        "score_delta": hierarchical_score - exact_score,
        "cluster_crossings": hierarchical_evolution["shadow_cluster_eprocess_crossings"],
        "shadow_only_replays": hierarchical_evolution["shadow_only_candidate_replay_attempts"],
        "shadow_probations": hierarchical_evolution["shadow_candidate_probations"],
        "future_audit_confirmations": hierarchical_metrics["future_audit"]["confirmed"],
        "exact_requests": exact_costs["requests"],
        "hierarchical_requests": hierarchical_costs["requests"],
        "exact_tokens": exact_costs["total_tokens"],
        "hierarchical_tokens": hierarchical_costs["total_tokens"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permit development runs whose manifest records a dirty worktree.",
    )
    args = parser.parse_args()
    summary = verify(args.sweep_dir, require_clean=not args.allow_dirty)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
