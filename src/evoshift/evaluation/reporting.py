"""JSON and Markdown serialization helpers for reproducible run artifacts."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Union

from pydantic import BaseModel

PathLike = Union[str, Path]


def json_safe(value: Any) -> Any:
    """Recursively convert reports, Pydantic models, and special floats to JSON."""

    if isinstance(value, BaseModel):
        return json_safe(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def to_json_report(report: Any, *, indent: int = 2) -> str:
    """Serialize an evaluation report with stable key order and strict JSON."""

    return json.dumps(
        json_safe(report),
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
        allow_nan=False,
    )


def write_json_report(report: Any, path: PathLike, *, indent: int = 2) -> Path:
    """Write an atomic-enough standalone JSON run artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(to_json_report(report, indent=indent) + "\n", encoding="utf-8")
    return destination


def _format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "N/A"
        return f"{value:.4f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_value(item) for item in value)
    return str(value)


def _metric_table(metrics: Mapping[str, Any]) -> str:
    lines = ["| Metric | Value |", "|---|---:|"]
    for name, value in metrics.items():
        if isinstance(value, Mapping):
            continue
        lines.append(f"| {str(name).replace('_', ' ')} | {_format_value(value)} |")
    return "\n".join(lines)


def render_markdown_report(
    report: Mapping[str, Any],
    *,
    title: str = "EvoShift Evaluation Report",
) -> str:
    """Render the canonical stream summary as a compact Markdown report."""

    safe: dict[str, Any] = json_safe(report)
    lines = [f"# {title}", ""]

    overall = safe.get("overall", {})
    overview: dict[str, Any] = {"n_episodes": safe.get("n_episodes", 0)}
    if isinstance(overall, Mapping):
        overview.update(overall)
    for key in (
        "auac",
        "post_shift_gain",
        "cumulative_regret",
        "cumulative_regret_vs_baseline",
        "promotion_precision",
    ):
        if key in safe:
            overview[key] = safe[key]
    lines.extend(["## Summary", "", _metric_table(overview), ""])

    phases = safe.get("phases")
    if isinstance(phases, Mapping) and phases:
        lines.extend(
            [
                "## Performance by phase",
                "",
                "| Phase | N | Mean score | Success rate |",
                "|---|---:|---:|---:|",
            ]
        )
        for phase, values in phases.items():
            phase_values = values if isinstance(values, Mapping) else {}
            lines.append(
                "| {} | {} | {} | {} |".format(
                    phase,
                    _format_value(phase_values.get("n", 0)),
                    _format_value(phase_values.get("mean_score", 0.0)),
                    _format_value(phase_values.get("success_rate", 0.0)),
                )
            )
        lines.append("")

    recovery = safe.get("recovery_steps")
    if isinstance(recovery, Mapping) and recovery:
        lines.extend(
            [
                "## Shift recovery",
                "",
                "| Shift index | Recovery steps |",
                "|---:|---:|",
            ]
        )
        for boundary, steps in recovery.items():
            lines.append(f"| {boundary} | {_format_value(steps)} |")
        lines.append("")

    continual = safe.get("continual_learning")
    if isinstance(continual, Mapping):
        lines.extend(["## Continual learning", "", _metric_table(continual), ""])

    resources = safe.get("resources")
    if isinstance(resources, Mapping):
        lines.extend(["## Resource usage", "", _metric_table(resources), ""])

    return "\n".join(lines).rstrip() + "\n"


def write_markdown_report(
    report: Mapping[str, Any],
    path: PathLike,
    *,
    title: str = "EvoShift Evaluation Report",
) -> Path:
    """Write the human-readable companion to a JSON evaluation artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_markdown_report(report, title=title), encoding="utf-8")
    return destination


__all__ = [
    "json_safe",
    "render_markdown_report",
    "to_json_report",
    "write_json_report",
    "write_markdown_report",
]
