from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml

from evoshift.config import EvoShiftConfig
from evoshift.evaluation import write_json_report, write_markdown_report
from evoshift.schemas import Episode, FailureRecord, PromotionDecision, RunManifest


def git_state(workdir: Path) -> Tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=workdir,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
        return commit or "unknown", dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", False


class RunArtifacts:
    """Append-only experiment artifacts suitable for post-hoc auditing."""

    def __init__(self, root: Path, run_id: str):
        self.run_dir = root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)

    def initialize(self, manifest: RunManifest, config: EvoShiftConfig) -> None:
        self.write_json("manifest.json", manifest.model_dump(mode="json"))
        resolved = yaml.safe_dump(
            config.model_dump(mode="json"), sort_keys=True, allow_unicode=True
        )
        self._atomic_text("resolved_config.yaml", resolved)
        self._atomic_text(
            "README.md",
            "# Run artifact\n\n"
            f"Run ID: `{manifest.run_id}`  \n"
            f"Algorithm: `{manifest.algorithm.value}`  \n"
            f"Model: `{manifest.model}`  \n"
            f"Config SHA-256: `{manifest.config_hash}`  \n"
            f"Dataset SHA-256: `{manifest.dataset_hash}`\n",
        )

    def append_episode(self, episode: Episode) -> None:
        self._append_jsonl("predictions.jsonl", episode.model_dump(mode="json"))
        trace = {
            "episode_id": episode.episode_id,
            "index": episode.index,
            "sample_id": episode.sample.sample_id,
            "domain": episode.sample.domain,
            "phase": episode.sample.phase,
            "selected_memory_ids": episode.selected_memory_ids,
            "policy_version": episode.policy_version,
            "score": episode.score.model_dump(mode="json"),
            "usage": episode.usage.model_dump(mode="json"),
            "shift": episode.shift.model_dump(mode="json") if episode.shift else None,
        }
        self._append_jsonl("traces.jsonl", trace)

    def append_failure(self, failure: FailureRecord) -> None:
        self._append_jsonl("failures.jsonl", failure.model_dump(mode="json"))

    def append_decision(self, decision: PromotionDecision) -> None:
        self._append_jsonl("promotion_decisions.jsonl", decision.model_dump(mode="json"))

    def finalize(
        self,
        manifest: RunManifest,
        metrics: Dict[str, Any],
        budget: Dict[str, Any],
        extra: Dict[str, Any],
    ) -> None:
        self.write_json("manifest.json", manifest.model_dump(mode="json"))
        self.write_json("metrics.json", metrics)
        self.write_json("costs.json", budget)
        self.write_json("summary.json", extra)
        write_markdown_report(
            metrics,
            self.run_dir / "report.md",
            title=f"EvoShift run {manifest.run_id}",
        )

    def write_json(self, name: str, payload: Any) -> None:
        write_json_report(payload, self.run_dir / name)

    def _append_jsonl(self, name: str, payload: Dict[str, Any]) -> None:
        path = self.run_dir / name
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _atomic_text(self, name: str, value: str) -> None:
        path = self.run_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)


def load_episodes(path: Path) -> list[Episode]:
    episodes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            episodes.append(Episode.model_validate_json(line))
    return episodes


def list_run_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())
