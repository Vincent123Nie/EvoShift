from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from evoshift.benchmarks.base import BenchmarkAdapter, ensure_unique_ids, select_samples
from evoshift.errors import DatasetError
from evoshift.schemas import BenchmarkSample

_PROMPT_FIELDS = ("prompt", "input", "question", "query", "instruction", "text")
_REFERENCE_FIELDS = ("reference", "target", "answer", "answers", "output", "label")


class HuggingFaceBenchmarkAdapter(BenchmarkAdapter):
    """Generic adapter for datasets loadable through the optional `datasets` package."""

    def __init__(
        self,
        dataset_name: str,
        *,
        split: str = "test",
        subsets: Sequence[str] = (),
        revision: str = "",
        prompt_field: str | None = None,
        reference_field: str | None = None,
        domain_field: str | None = None,
        phase_field: str | None = None,
        evaluator: str = "exact_match",
        limit: int = 0,
        shuffle: bool = False,
        seed: int = 42,
        protected_phases: Sequence[str] = (),
    ) -> None:
        if not dataset_name.strip():
            raise DatasetError("Hugging Face dataset name cannot be empty")
        if not split.strip():
            raise DatasetError("Hugging Face split cannot be empty")
        self.dataset_name = dataset_name
        self.split = split
        self.subsets = tuple(subsets)
        self.revision = revision
        self.prompt_field = prompt_field
        self.reference_field = reference_field
        self.domain_field = domain_field
        self.phase_field = phase_field
        self.evaluator = evaluator
        self.limit = limit
        self.shuffle = shuffle
        self.seed = seed
        self.protected_phases = frozenset(protected_phases)

    def load(self) -> list[BenchmarkSample]:
        load_dataset = self._resolve_loader()
        configurations: tuple[str | None, ...] = tuple(self.subsets) if self.subsets else (None,)
        samples: list[BenchmarkSample] = []
        for subset_index, subset in enumerate(configurations):
            kwargs: dict[str, Any] = {
                "split": self.split,
                "trust_remote_code": False,
            }
            if self.revision:
                kwargs["revision"] = self.revision
            try:
                dataset = load_dataset(self.dataset_name, subset, **kwargs)
            except Exception as exc:
                subset_label = subset or "default"
                raise DatasetError(
                    f"cannot load Hugging Face dataset {self.dataset_name}/"
                    f"{subset_label}[{self.split}]: {exc}"
                ) from exc
            samples.extend(self._normalize_dataset(dataset, subset, subset_index))

        if not samples:
            raise DatasetError(f"Hugging Face dataset {self.dataset_name}[{self.split}] is empty")
        unique = ensure_unique_ids(samples)
        return select_samples(
            unique,
            limit=self.limit,
            shuffle=self.shuffle,
            seed=self.seed,
        )

    @staticmethod
    def _resolve_loader() -> Any:
        try:
            module = importlib.import_module("datasets")
        except (ImportError, ModuleNotFoundError) as exc:
            raise DatasetError(
                "Hugging Face benchmarks require the optional 'datasets' dependency. "
                "Install it with: pip install 'evoshift[hf]'"
            ) from exc
        loader = getattr(module, "load_dataset", None)
        if loader is None:
            raise DatasetError("installed 'datasets' package has no load_dataset function")
        return loader

    def _normalize_dataset(
        self,
        dataset: Any,
        subset: str | None,
        subset_index: int,
    ) -> list[BenchmarkSample]:
        rows = list(dataset)
        if not rows:
            return []
        first = rows[0]
        if not isinstance(first, Mapping):
            raise DatasetError("Hugging Face dataset rows must be mappings")
        prompt_field = self.prompt_field or self._infer_field(first, _PROMPT_FIELDS, "prompt")
        reference_field = self.reference_field or self._infer_field(
            first, _REFERENCE_FIELDS, "reference"
        )
        feature = self._feature_for(dataset, reference_field)
        subset_label = subset or "default"
        default_phase = f"phase_{subset_index:02d}_{_slug(subset_label)}"
        normalized: list[BenchmarkSample] = []

        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise DatasetError(f"Hugging Face row {index} is not a mapping")
            if prompt_field not in row or reference_field not in row:
                raise DatasetError(
                    f"Hugging Face row {index} lacks {prompt_field!r} or {reference_field!r}"
                )
            phase = (
                str(row[self.phase_field])
                if self.phase_field and self.phase_field in row
                else default_phase
            )
            domain = (
                str(row[self.domain_field])
                if self.domain_field and self.domain_field in row
                else subset_label
            )
            raw_id = row.get("sample_id", row.get("id", index))
            reference = self._decode_reference(row[reference_field], feature)
            normalized.append(
                BenchmarkSample(
                    sample_id=f"hf:{self.dataset_name}:{subset_label}:{raw_id}",
                    prompt=_as_text(row[prompt_field]),
                    reference=reference,
                    domain=domain,
                    phase=phase,
                    evaluator=self.evaluator,
                    metadata={
                        "dataset": self.dataset_name,
                        "subset": subset_label,
                        "split": self.split,
                        "revision": self.revision or "upstream-default",
                        "source_index": index,
                        "prompt_field": prompt_field,
                        "reference_field": reference_field,
                        "phase_index": subset_index,
                        "is_shift_boundary": subset_index > 0 and index == 0,
                        "protected": phase in self.protected_phases,
                    },
                )
            )
        return normalized

    @staticmethod
    def _infer_field(row: Mapping[str, Any], candidates: Sequence[str], role: str) -> str:
        for candidate in candidates:
            if candidate in row:
                return candidate
        available = ", ".join(sorted(str(key) for key in row))
        raise DatasetError(f"cannot infer Hugging Face {role} field; available fields: {available}")

    @staticmethod
    def _feature_for(dataset: Any, field: str) -> Any:
        features = getattr(dataset, "features", None)
        if isinstance(features, Mapping):
            return features.get(field)
        return None

    @staticmethod
    def _decode_reference(value: Any, feature: Any) -> Any:
        converter = getattr(feature, "int2str", None)
        if callable(converter) and isinstance(value, int) and not isinstance(value, bool):
            try:
                return converter(value)
            except (TypeError, ValueError):
                return value
        return value


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _slug(value: str) -> str:
    slug = "".join(character if character.isalnum() else "_" for character in value.lower())
    return slug.strip("_") or "default"


HFBenchmarkAdapter = HuggingFaceBenchmarkAdapter
