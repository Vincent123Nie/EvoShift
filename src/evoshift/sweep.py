from __future__ import annotations

import csv
import itertools
import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import yaml

from evoshift.benchmarks import create_benchmark
from evoshift.config import load_config
from evoshift.runner import EvoShiftRunner
from evoshift.schemas import Algorithm


@dataclass(frozen=True)
class SweepSpec:
    base_config: Path
    algorithms: Sequence[Algorithm]
    seeds: Sequence[int]
    grid: Mapping[str, Sequence[Any]]
    max_runs: int = 100


def load_sweep_spec(path: Path, root: Path) -> SweepSpec:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("sweep spec must be a YAML mapping")
    base = Path(str(payload.get("base_config", "")))
    if not str(base):
        raise ValueError("sweep spec requires base_config")
    base = base if base.is_absolute() else root / base
    algorithms = [Algorithm(value) for value in payload.get("algorithms", ["evoshift"])]
    seeds = [int(value) for value in payload.get("seeds", [42])]
    grid = payload.get("grid", {})
    if not isinstance(grid, dict) or any(
        not isinstance(values, list) or not values for values in grid.values()
    ):
        raise ValueError("sweep grid values must be non-empty lists")
    maximum = int(payload.get("max_runs", 100))
    spec = SweepSpec(base, algorithms, seeds, grid, maximum)
    if len(expand_sweep(spec)) > maximum:
        raise ValueError(f"sweep expands beyond max_runs={maximum}")
    return spec


def _override_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def expand_sweep(spec: SweepSpec) -> List[Dict[str, Any]]:
    keys = sorted(spec.grid)
    values = [spec.grid[key] for key in keys]
    combinations = itertools.product(*values) if values else [()]
    assignments: List[Dict[str, Any]] = []
    for combination in combinations:
        parameters = dict(zip(keys, combination))
        for algorithm in spec.algorithms:
            for seed in spec.seeds:
                assignments.append(
                    {"algorithm": algorithm.value, "seed": seed, "parameters": parameters}
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
        rows.append(
            {
                "run_id": result.run_id,
                "run_dir": str(result.run_dir),
                "algorithm": assignment["algorithm"],
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
            {"algorithm": row["algorithm"], "parameters": row["parameters"]},
            sort_keys=True,
        )
        grouped.setdefault(key, []).append(row)
    output = []
    for key, group in grouped.items():
        identity = json.loads(key)
        scores = [float(row["mean_score"]) for row in group]
        total_tokens = [float(row["total_tokens"]) for row in group]
        output.append(
            {
                **identity,
                "n_seeds": len(group),
                "score_mean": statistics.fmean(scores),
                "score_std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "total_tokens_mean": statistics.fmean(total_tokens),
                "run_ids": [row["run_id"] for row in group],
            }
        )
    return sorted(output, key=lambda item: (-item["score_mean"], item["algorithm"]))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "run_id",
        "algorithm",
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
        "| Algorithm | Parameters | Seeds | Score mean | Score std | Mean total tokens |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        lines.append(
            "| {} | `{}` | {} | {:.4f} | {:.4f} | {:.1f} |".format(
                item["algorithm"],
                json.dumps(item["parameters"], sort_keys=True),
                item["n_seeds"],
                item["score_mean"],
                item["score_std"],
                item["total_tokens_mean"],
            )
        )
    lines.extend(
        [
            "",
            "Only same-model, same-dataset, same-budget rows are directly comparable.",
            "Synthetic demo results validate plumbing and must not be reported as SOTA evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = ["SweepSpec", "aggregate_sweep", "expand_sweep", "load_sweep_spec", "run_sweep"]
