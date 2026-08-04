from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from evoshift.benchmarks.base import (
    BenchmarkAdapter,
    ensure_unique_ids,
    select_samples,
)
from evoshift.errors import DatasetError
from evoshift.schemas import BenchmarkSample

_PROMPT_ALIASES = ("prompt", "input", "question", "query", "instruction", "text")
_REFERENCE_ALIASES = ("reference", "target", "answer", "output", "label")


def _take_first(data: Mapping[str, Any], names: tuple[str, ...]) -> tuple[Any, str]:
    for name in names:
        if name in data:
            return data[name], name
    raise KeyError(names[0])


class JSONLBenchmarkAdapter(BenchmarkAdapter):
    """Read canonical or common prompt/target JSONL records."""

    def __init__(
        self,
        path: Path,
        *,
        limit: int = 0,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        self.path = Path(path)
        self.limit = limit
        self.shuffle = shuffle
        self.seed = seed

    def load(self) -> list[BenchmarkSample]:
        if not self.path.exists():
            raise DatasetError(f"JSONL benchmark does not exist: {self.path}")
        if not self.path.is_file():
            raise DatasetError(f"JSONL benchmark path is not a file: {self.path}")

        samples: list[BenchmarkSample] = []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise DatasetError(f"cannot read JSONL benchmark {self.path}: {exc}") from exc

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(
                    f"invalid JSON in {self.path} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise DatasetError(
                    f"JSONL record in {self.path} at line {line_number} must be an object"
                )
            samples.append(self._normalize_record(record, line_number))

        if not samples:
            raise DatasetError(f"JSONL benchmark is empty: {self.path}")
        unique = ensure_unique_ids(samples)
        return select_samples(
            unique,
            limit=self.limit,
            shuffle=self.shuffle,
            seed=self.seed,
        )

    def _normalize_record(self, record: dict[str, Any], line_number: int) -> BenchmarkSample:
        consumed: set[str] = set()
        try:
            prompt, prompt_key = _take_first(record, _PROMPT_ALIASES)
            reference, reference_key = _take_first(record, _REFERENCE_ALIASES)
        except KeyError as exc:
            raise DatasetError(
                f"record in {self.path} at line {line_number} is missing "
                f"a prompt or reference field"
            ) from exc
        consumed.update({prompt_key, reference_key})

        sample_id = record.get("sample_id", record.get("id", f"{self.path.stem}:{line_number}"))
        consumed.update({"sample_id", "id"})
        for field in ("domain", "phase", "evaluator", "metadata"):
            if field in record:
                consumed.add(field)

        raw_metadata = record.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raise DatasetError(f"metadata in {self.path} at line {line_number} must be an object")
        metadata = dict(raw_metadata)
        metadata.setdefault("source_path", str(self.path))
        metadata.setdefault("source_line", line_number)
        extras = {key: value for key, value in record.items() if key not in consumed}
        if extras:
            metadata.setdefault("source_fields", extras)

        try:
            return BenchmarkSample(
                sample_id=str(sample_id),
                prompt=str(prompt),
                reference=reference,
                domain=str(record.get("domain", "default")),
                phase=str(record.get("phase", "stream")),
                evaluator=str(record.get("evaluator", "exact_match")),
                metadata=metadata,
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise DatasetError(
                f"invalid benchmark record in {self.path} at line {line_number}: {exc}"
            ) from exc
