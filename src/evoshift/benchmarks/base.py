from __future__ import annotations

import hashlib
import json
import random
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence

from evoshift.errors import DatasetError
from evoshift.schemas import BenchmarkSample


class BenchmarkAdapter(ABC):
    """Synchronous, reproducible source of normalized benchmark samples."""

    @abstractmethod
    def load(self) -> list[BenchmarkSample]:
        """Load and normalize the benchmark stream."""

    def __iter__(self) -> Iterator[BenchmarkSample]:
        return iter(self.load())

    def fingerprint(self) -> str:
        """Return a stable content hash for experiment manifests."""

        return sample_fingerprint(self.load())


def select_samples(
    samples: Sequence[BenchmarkSample],
    *,
    limit: int = 0,
    shuffle: bool = False,
    seed: int = 42,
) -> list[BenchmarkSample]:
    """Apply deterministic selection without mutating the source sequence."""

    if limit < 0:
        raise DatasetError("benchmark limit cannot be negative")
    selected = list(samples)
    if shuffle:
        random.Random(seed).shuffle(selected)
    if limit:
        selected = selected[:limit]
    return selected


def ensure_unique_ids(samples: Iterable[BenchmarkSample]) -> list[BenchmarkSample]:
    normalized = list(samples)
    seen = set()
    duplicates = set()
    for sample in normalized:
        if sample.sample_id in seen:
            duplicates.add(sample.sample_id)
        seen.add(sample.sample_id)
    if duplicates:
        joined = ", ".join(sorted(duplicates))
        raise DatasetError(f"duplicate benchmark sample_id values: {joined}")
    return normalized


def sample_fingerprint(samples: Sequence[BenchmarkSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        payload = sample.model_dump(mode="json")
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def stable_seed(seed: int, namespace: str) -> int:
    payload = f"{seed}:{namespace}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")
