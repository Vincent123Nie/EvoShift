from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evoshift.benchmarks import (
    BBH_CANARY,
    BBH_FILE_MANIFEST,
    DEFAULT_BBH_REVISION,
    BBHBenchmarkAdapter,
    JSONLBenchmarkAdapter,
    PolicyShiftBenchmark,
    SyntheticShiftBenchmark,
    TauRetailPolicyShiftBenchmark,
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


def test_policy_shift_separates_oracle_from_observed_feedback() -> None:
    adapter = PolicyShiftBenchmark(
        seed=123,
        phase_size=8,
        feedback_noise_rate=1.0,
        feedback_attack_rate=1.0,
    )

    first = adapter.load()
    second = adapter.load()

    assert first == second
    assert len(first) == 24
    assert {sample.phase for sample in first} == {"phase_0", "phase_1", "phase_2"}
    assert {sample.metadata["policy_version"] for sample in first} == {"v1", "v2", "v3"}
    assert all(
        str(sample.metadata["policy_version"]) not in sample.prompt
        and str(sample.metadata["policy_version"]) not in sample.sample_id
        for sample in first
    )
    assert all(sample.metadata["feedback_reference"] != sample.reference for sample in first)
    assert {sample.metadata["feedback_kind"] for sample in first} == {"noise", "attack"}
    assert any(sample.metadata["protected"] for sample in first)
    assert any(sample.metadata["policy_changed_case"] for sample in first)
    assert any(sample.metadata["future_change_case"] for sample in first)

    for sample in first:
        tier = str(sample.metadata["customer_tier"])
        window = int(sample.metadata[f"{tier}_window"])
        expected = "APPROVE" if int(sample.metadata["request_day"]) <= window else "DENY"
        assert sample.reference == expected


def test_policy_shift_distinguishes_transition_protected_and_future_cases() -> None:
    samples = PolicyShiftBenchmark(
        seed=7,
        phase_size=24,
        feedback_noise_rate=0.0,
        feedback_attack_rate=0.0,
    ).load()
    by_phase = {
        phase: [sample for sample in samples if sample.phase == phase]
        for phase in ("phase_0", "phase_1", "phase_2")
    }

    assert not any(sample.metadata["transition_case"] for sample in by_phase["phase_0"])
    assert any(sample.metadata["future_change_case"] for sample in by_phase["phase_0"])
    assert any(sample.metadata["transition_case"] for sample in by_phase["phase_1"])
    assert any(sample.metadata["future_change_case"] for sample in by_phase["phase_1"])
    assert any(sample.metadata["transition_case"] for sample in by_phase["phase_2"])
    assert not any(sample.metadata["future_change_case"] for sample in by_phase["phase_2"])

    for sample in samples:
        transition = bool(sample.metadata["transition_case"])
        protected = bool(sample.metadata["protected"])
        future = bool(sample.metadata["future_change_case"])
        assert sum((transition, protected, future)) == 1
        assert sample.metadata["policy_changed_case"] is transition


def test_policy_shift_factory_uses_noise_configuration(tmp_path: Path) -> None:
    config = BenchmarkConfig(
        kind="policy_shift",
        path=None,
        phase_size=8,
        feedback_noise_rate=0.25,
        feedback_attack_rate=0.50,
        feedback_shared_source=True,
        feedback_shared_source_name="shared_portal",
        feedback_attack_burst_length=2,
        policy_schedule=["v1", "v2", "v1", "v2"],
    )

    adapter = create_benchmark(config, root=tmp_path, seed=9)

    assert isinstance(adapter, PolicyShiftBenchmark)
    assert adapter.feedback_noise_rate == 0.25
    assert adapter.feedback_attack_rate == 0.50
    assert adapter.feedback_shared_source is True
    assert adapter.feedback_shared_source_name == "shared_portal"
    assert adapter.feedback_attack_burst_length == 2
    assert adapter.policy_schedule == ("v1", "v2", "v1", "v2")


def test_policy_shift_supports_revocation_and_recurring_regimes() -> None:
    samples = PolicyShiftBenchmark(
        seed=7,
        phase_size=8,
        feedback_noise_rate=0.0,
        feedback_attack_rate=0.0,
        policy_schedule=["v1", "v2", "v1", "v2"],
    ).load()

    assert len(samples) == 32
    assert [
        next(sample for sample in samples if sample.phase == f"phase_{index}").metadata[
            "policy_version"
        ]
        for index in range(4)
    ] == ["v1", "v2", "v1", "v2"]
    reverted = [sample for sample in samples if sample.phase == "phase_2"]
    recurring = [sample for sample in samples if sample.phase == "phase_3"]
    assert all(sample.metadata["is_policy_reversion"] for sample in reverted + recurring)
    assert any(sample.metadata["transition_case"] for sample in reverted)
    assert any(sample.metadata["transition_case"] for sample in recurring)
    assert all("policy_v2" in sample.metadata["stale_memory_tags"] for sample in reverted)
    assert all("policy_v2" in sample.metadata["valid_memory_tags"] for sample in recurring)
    assert all(
        "policy_schedule" not in sample.prompt and "stale_memory_tags" not in sample.prompt
        for sample in samples
    )


def test_policy_shift_rejects_unknown_policy_versions() -> None:
    with pytest.raises(ValueError, match="supports only v1, v2, and v3"):
        BenchmarkConfig(kind="policy_shift", path=None, policy_schedule=["v1", "v4"])
    with pytest.raises(DatasetError, match="supports only v1, v2, and v3"):
        PolicyShiftBenchmark(policy_schedule=["v4"])


def test_policy_shift_shared_source_burst_creates_observable_contexts() -> None:
    samples = PolicyShiftBenchmark(
        seed=7,
        phase_size=24,
        feedback_noise_rate=0.0,
        feedback_attack_rate=0.0,
        feedback_shared_source=True,
        feedback_shared_source_name="shared_portal",
        feedback_attack_burst_length=2,
    ).load()

    assert {sample.metadata["feedback_source"] for sample in samples} == {"shared_portal"}
    early_attacks = [
        sample
        for sample in samples
        if sample.phase == "phase_0" and sample.metadata["feedback_kind"] == "attack"
    ]
    assert len(early_attacks) == 4
    assert all(
        sample.metadata["feedback_attack_goal"] == "premature_update" for sample in early_attacks
    )
    assert all(sample.metadata["future_change_case"] for sample in early_attacks)
    assert all(sample.metadata["feedback_context"] for sample in samples)


def _fake_tau_sources() -> tuple[bytes, bytes, dict[str, dict[str, Any]]]:
    policy = b"pinned retail policy clauses for test"
    tasks = json.dumps(
        [
            {
                "evaluation_criteria": {
                    "actions": [
                        {"name": "cancel_pending_order"},
                        {"name": "return_delivered_order_items"},
                    ]
                }
            }
        ],
        separators=(",", ":"),
    ).encode()
    manifest = {
        "policy": {
            "path": "policy.md",
            "upstream_path": "policy.md",
            "sha256": hashlib.sha256(policy).hexdigest(),
            "bytes": len(policy),
        },
        "tasks": {
            "path": "tasks.json",
            "upstream_path": "tasks.json",
            "sha256": hashlib.sha256(tasks).hexdigest(),
            "bytes": len(tasks),
        },
    }
    return policy, tasks, manifest


def test_tau_retail_policy_shift_is_pinned_multi_rule_and_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy, tasks, manifest = _fake_tau_sources()
    adapter = TauRetailPolicyShiftBenchmark(
        tmp_path,
        seed=7,
        phase_size=18,
        policy_schedule=["v1", "v2", "v3", "v2", "v1", "v2"],
        feedback_noise_rate=0.0,
        feedback_attack_burst_length=0,
        expected_manifest=manifest,
        expected_task_count=1,
        required_policy_snippets=(),
        required_task_actions=("cancel_pending_order", "return_delivered_order_items"),
    )

    def fake_download(url: str) -> bytes:
        return policy if url.endswith("policy.md") else tasks

    monkeypatch.setattr(adapter, "_download_bytes", fake_download)
    samples = adapter.load()

    assert len(samples) == 108
    assert len({sample.metadata["rule_family"] for sample in samples}) >= 6
    assert {sample.metadata["policy_version"] for sample in samples} == {"v1", "v2", "v3"}
    assert any(sample.metadata["transition_case"] for sample in samples)
    assert any(sample.metadata["future_change_case"] for sample in samples)
    assert any(sample.metadata["protected"] for sample in samples)
    assert all(sample.metadata["derived"] for sample in samples)
    assert all(sample.metadata["official_tau3_benchmark"] is False for sample in samples)
    assert all(
        sample.metadata["policy_version"] not in sample.prompt
        and "source_rule" not in sample.prompt
        and "feedback_reference" not in sample.prompt
        for sample in samples
    )
    assert all(len(sample.metadata["valid_memory_tags"]) == 1 for sample in samples)
    assert all(len(sample.metadata["stale_memory_tags"]) == 1 for sample in samples)
    assert adapter.verify_cache()["official_tau3_benchmark"] is False


def test_tau_retail_policy_shift_rejects_tampered_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy, tasks, manifest = _fake_tau_sources()
    adapter = TauRetailPolicyShiftBenchmark(
        tmp_path,
        phase_size=8,
        expected_manifest=manifest,
        expected_task_count=1,
        required_policy_snippets=(),
        required_task_actions=(),
    )
    monkeypatch.setattr(
        adapter,
        "_download_bytes",
        lambda url: policy if url.endswith("policy.md") else tasks,
    )
    adapter.load()
    (adapter.dataset_dir / "policy.md").write_bytes(policy + b"tampered")

    with pytest.raises(DatasetError, match="SHA-256 mismatch"):
        adapter.load()


def test_tau_retail_policy_shift_factory_contract(tmp_path: Path) -> None:
    config = BenchmarkConfig(
        kind="tau3_retail_policy_shift",
        path="data/tau3",
        phase_size=12,
        policy_schedule=["v1", "v2"],
    )

    adapter = create_benchmark(config, root=tmp_path, seed=9)

    assert isinstance(adapter, TauRetailPolicyShiftBenchmark)
    assert adapter.dataset_dir == tmp_path / "data/tau3/v1.0.1"
    assert adapter.phase_size == 12
    assert adapter.policy_schedule == ("v1", "v2")


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
    assert samples[0].evaluator == "binary_choice"
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


def test_bbh_manifest_uses_complete_large_upstream_files() -> None:
    assert BBH_FILE_MANIFEST["causal_judgement"] == {
        "sha256": "6e8bca44dd402008ed41c07a3d0a1d5e7b03db7001a7e002fcdf4c60985ea081",
        "bytes": 202943,
    }
    assert BBH_FILE_MANIFEST["salient_translation_error_detection"] == {
        "sha256": "becf3eb8dd53c821a555ecd5c1334ecd93eb25f980e63be963c518e7138de5da",
        "bytes": 286443,
    }
    assert BBH_FILE_MANIFEST["tracking_shuffled_objects_seven_objects"] == {
        "sha256": "b59fd05b850c37cf57cb514ae1e6975c00b1638ade2291722056b2af3c54084d",
        "bytes": 214906,
    }


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
