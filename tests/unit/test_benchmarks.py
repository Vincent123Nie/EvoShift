from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evoshift.benchmarks import (
    BBH_CANARY,
    DEFAULT_BBH_REVISION,
    BBHBenchmarkAdapter,
    JSONLBenchmarkAdapter,
    SyntheticShiftBenchmark,
    create_benchmark,
)
from evoshift.benchmarks.huggingface import HuggingFaceBenchmarkAdapter
from evoshift.config import BenchmarkConfig
from evoshift.errors import DatasetError


def test_jsonl_adapter_normalizes_aliases_and_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    records = [
        {"id": "one", "question": "1 + 1", "answer": "2", "extra": "kept"},
        {
            "sample_id": "two",
            "prompt": "2 + 2",
            "reference": "4",
            "domain": "math",
            "phase": "later",
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    first = JSONLBenchmarkAdapter(path, shuffle=True, seed=7).load()
    second = JSONLBenchmarkAdapter(path, shuffle=True, seed=7).load()

    assert first == second
    assert {sample.sample_id for sample in first} == {"one", "two"}
    aliased = next(sample for sample in first if sample.sample_id == "one")
    assert aliased.prompt == "1 + 1"
    assert aliased.reference == "2"
    assert aliased.metadata["source_fields"] == {"extra": "kept"}


def test_jsonl_adapter_reports_line_and_duplicate_ids(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text('{"sample_id": "x"}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="line 1"):
        JSONLBenchmarkAdapter(invalid).load()

    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        "\n".join(
            [
                '{"sample_id":"x","prompt":"p","reference":"a"}',
                '{"sample_id":"x","prompt":"q","reference":"b"}',
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="duplicate"):
        JSONLBenchmarkAdapter(duplicate).load()


def test_synthetic_shift_is_reproducible_and_has_protected_probes() -> None:
    adapter = SyntheticShiftBenchmark(seed=123, phase_size=4, protected_probes_per_phase=1)
    first = adapter.load()
    second = adapter.load()

    assert first == second
    assert len(first) == 4 * 4 + 3
    assert {sample.phase for sample in first} == {
        "phase_0_addition",
        "phase_1_affine",
        "phase_2_conditional",
        "phase_3_symbolic",
    }
    boundaries = [sample for sample in first if sample.metadata["is_shift_boundary"]]
    assert [sample.phase for sample in boundaries] == [
        "phase_1_affine",
        "phase_2_conditional",
        "phase_3_symbolic",
    ]
    probes = [sample for sample in first if sample.metadata["protected_probe"]]
    assert len(probes) == 3
    assert all(sample.domain == "arithmetic/addition" for sample in probes)
    assert all(sample.evaluator == "exact_match" for sample in first)
    assert (
        adapter.fingerprint()
        == SyntheticShiftBenchmark(
            seed=123, phase_size=4, protected_probes_per_phase=1
        ).fingerprint()
    )


def _fake_bbh_payload() -> bytes:
    return json.dumps(
        {
            "canary": BBH_CANARY,
            "examples": [
                {"input": "not True is", "target": "False"},
                {"input": "True and True is", "target": "True"},
            ],
        },
        separators=(",", ":"),
    ).encode("utf-8")


def test_bbh_download_is_pinned_manifested_and_cache_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _fake_bbh_payload()
    manifest: dict[str, dict[str, Any]] = {
        "boolean_expressions": {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    }
    adapter = BBHBenchmarkAdapter(
        tmp_path,
        revision="test-revision",
        subsets=["boolean_expressions"],
        phase_size=1,
        expected_manifest=manifest,
    )
    calls = []

    def fake_download(url: str) -> bytes:
        calls.append(url)
        return data

    monkeypatch.setattr(adapter, "_download_bytes", fake_download)
    samples = adapter.load()

    assert calls == [
        "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/"
        "test-revision/bbh/boolean_expressions.json"
    ]
    assert len(samples) == 1
    assert samples[0].reference == "False"
    assert samples[0].evaluator == "normalized_exact_match"
    disk_manifest = json.loads(adapter.manifest_path.read_text(encoding="utf-8"))
    assert disk_manifest["revision"] == "test-revision"
    assert (
        disk_manifest["files"]["boolean_expressions"]["sha256"]
        == manifest["boolean_expressions"]["sha256"]
    )

    monkeypatch.setattr(
        adapter,
        "_download_bytes",
        lambda _url: pytest.fail("verified cache should not redownload"),
    )
    assert adapter.load() == samples
    assert adapter.verify_cache()["schema_version"] == 1


def test_bbh_rejects_tampered_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _fake_bbh_payload()
    manifest = {
        "boolean_expressions": {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    }
    adapter = BBHBenchmarkAdapter(
        tmp_path,
        revision="test-revision",
        subsets=["boolean_expressions"],
        expected_manifest=manifest,
    )
    monkeypatch.setattr(adapter, "_download_bytes", lambda _url: data)
    adapter.load()
    (adapter.dataset_dir / "boolean_expressions.json").write_bytes(data + b"tampered")

    with pytest.raises(DatasetError, match="SHA-256 mismatch"):
        adapter.load()


def test_default_bbh_revision_and_factory_contract(tmp_path: Path) -> None:
    config = BenchmarkConfig(
        kind="bbh",
        path=str(tmp_path),
        subsets=["boolean_expressions"],
        phase_size=3,
    )
    adapter = create_benchmark(config, root=tmp_path, seed=9)
    assert isinstance(adapter, BBHBenchmarkAdapter)
    assert adapter.revision == DEFAULT_BBH_REVISION
    assert adapter.phase_size == 3


def test_factory_resolves_jsonl_path_from_root(tmp_path: Path) -> None:
    data_dir = tmp_path / "fixtures"
    data_dir.mkdir()
    path = data_dir / "sample.jsonl"
    path.write_text('{"prompt":"p","reference":"a"}\n', encoding="utf-8")

    config = BenchmarkConfig(kind="jsonl", path="fixtures/sample.jsonl")
    adapter = create_benchmark(config, root=tmp_path)

    assert isinstance(adapter, JSONLBenchmarkAdapter)
    assert adapter.path == path
    assert adapter.load()[0].reference == "a"


def test_huggingface_adapter_has_friendly_optional_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_module(_name: str) -> Any:
        raise ModuleNotFoundError("datasets")

    monkeypatch.setattr("evoshift.benchmarks.huggingface.importlib.import_module", missing_module)
    adapter = HuggingFaceBenchmarkAdapter("example/dataset")
    with pytest.raises(DatasetError, match=r"pip install 'evoshift\[hf\]'"):
        adapter.load()


def test_huggingface_adapter_normalizes_subsets_and_class_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ClassLabel:
        def int2str(self, value: int) -> str:
            return {0: "no", 1: "yes"}[value]

    class FakeDataset(list[dict[str, Any]]):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            super().__init__(rows)
            self.features = {"label": ClassLabel()}

    calls: list[tuple[str, str | None, dict[str, Any]]] = []

    def load_dataset(name: str, subset: str | None, **kwargs: Any) -> FakeDataset:
        calls.append((name, subset, kwargs))
        phase = "protected" if subset == "alpha" else "shifted"
        return FakeDataset(
            [
                {
                    "id": f"{subset}-1",
                    "question": ["Is", subset, "valid?"],
                    "label": 1,
                    "domain_name": f"domain-{subset}",
                    "phase_name": phase,
                }
            ]
        )

    monkeypatch.setattr(
        "evoshift.benchmarks.huggingface.importlib.import_module",
        lambda _name: SimpleNamespace(load_dataset=load_dataset),
    )
    adapter = HuggingFaceBenchmarkAdapter(
        "org/example",
        subsets=["alpha", "beta"],
        revision="fixed-revision",
        domain_field="domain_name",
        phase_field="phase_name",
        evaluator="normalized_exact_match",
        protected_phases=["protected"],
    )

    samples = adapter.load()

    assert [sample.reference for sample in samples] == ["yes", "yes"]
    assert samples[0].prompt == '["Is", "alpha", "valid?"]'
    assert samples[0].metadata["protected"] is True
    assert samples[1].metadata["is_shift_boundary"] is True
    assert samples[1].domain == "domain-beta"
    assert calls == [
        (
            "org/example",
            "alpha",
            {"split": "test", "trust_remote_code": False, "revision": "fixed-revision"},
        ),
        (
            "org/example",
            "beta",
            {"split": "test", "trust_remote_code": False, "revision": "fixed-revision"},
        ),
    ]


def test_huggingface_adapter_rejects_bad_shapes_and_loader_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(DatasetError, match="name cannot be empty"):
        HuggingFaceBenchmarkAdapter(" ")
    with pytest.raises(DatasetError, match="split cannot be empty"):
        HuggingFaceBenchmarkAdapter("org/data", split=" ")

    adapter = HuggingFaceBenchmarkAdapter("org/data")
    with pytest.raises(DatasetError, match="rows must be mappings"):
        adapter._normalize_dataset([1], None, 0)
    with pytest.raises(DatasetError, match="row 1 is not a mapping"):
        adapter._normalize_dataset([{"prompt": "p", "answer": "a"}, 2], None, 0)
    with pytest.raises(DatasetError, match="lacks"):
        adapter._normalize_dataset(
            [{"prompt": "p", "answer": "a"}, {"prompt": "q"}],
            None,
            0,
        )
    with pytest.raises(DatasetError, match=r"cannot infer.*prompt"):
        adapter._normalize_dataset([{"answer": "a"}], None, 0)

    monkeypatch.setattr(
        "evoshift.benchmarks.huggingface.importlib.import_module",
        lambda _name: SimpleNamespace(),
    )
    with pytest.raises(DatasetError, match="no load_dataset"):
        adapter.load()

    def failing_loader(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("offline")

    monkeypatch.setattr(
        "evoshift.benchmarks.huggingface.importlib.import_module",
        lambda _name: SimpleNamespace(load_dataset=failing_loader),
    )
    with pytest.raises(DatasetError, match="cannot load Hugging Face dataset"):
        adapter.load()

    monkeypatch.setattr(
        "evoshift.benchmarks.huggingface.importlib.import_module",
        lambda _name: SimpleNamespace(load_dataset=lambda *_args, **_kwargs: []),
    )
    with pytest.raises(DatasetError, match="is empty"):
        adapter.load()
