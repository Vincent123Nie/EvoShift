from __future__ import annotations

from pathlib import Path

from evoshift.benchmarks.base import BenchmarkAdapter
from evoshift.benchmarks.bbh import DEFAULT_BBH_REVISION, BBHBenchmarkAdapter
from evoshift.benchmarks.huggingface import HuggingFaceBenchmarkAdapter
from evoshift.benchmarks.jsonl import JSONLBenchmarkAdapter
from evoshift.benchmarks.policy_shift import PolicyShiftBenchmark
from evoshift.benchmarks.synthetic import SyntheticShiftBenchmark
from evoshift.benchmarks.tau_retail_policy_shift import TauRetailPolicyShiftBenchmark
from evoshift.config import BenchmarkConfig
from evoshift.errors import DatasetError
from evoshift.schemas import BenchmarkSample


def create_benchmark(
    config: BenchmarkConfig,
    *,
    root: Path | None = None,
    seed: int = 42,
    cache_dir: Path | None = None,
) -> BenchmarkAdapter:
    """Construct a benchmark adapter from the project's strict BenchmarkConfig."""

    project_root = root or Path.cwd()
    kind = config.kind.strip().lower().replace("-", "_")
    if kind in {"jsonl", "json_lines"}:
        if not config.path:
            raise DatasetError("JSONL benchmark requires benchmark.path")
        jsonl_path = _resolve_path(Path(config.path), project_root)
        return JSONLBenchmarkAdapter(
            jsonl_path,
            limit=config.limit,
            shuffle=config.shuffle,
            seed=seed,
        )
    if kind in {"synthetic", "synthetic_shift"}:
        return SyntheticShiftBenchmark(
            seed=seed,
            phase_size=config.phase_size or 12,
            protected_phases=config.protected_phases or ["phase_0_addition"],
            protected_probes_per_phase=2,
            shuffle_within_phase=config.shuffle,
            limit=config.limit,
        )
    if kind in {"policy_shift", "enterprise_policy_shift"}:
        return PolicyShiftBenchmark(
            seed=seed,
            phase_size=config.phase_size or 24,
            feedback_noise_rate=config.feedback_noise_rate,
            feedback_attack_rate=config.feedback_attack_rate,
            feedback_shared_source=config.feedback_shared_source,
            feedback_shared_source_name=config.feedback_shared_source_name,
            feedback_attack_burst_length=config.feedback_attack_burst_length,
            feedback_warmup_attack_observations=config.feedback_warmup_attack_observations,
            feedback_warmup_attack_burst_length=config.feedback_warmup_attack_burst_length,
            prompt_style=config.policy_prompt_style,
            policy_schedule=config.policy_schedule or None,
            shuffle_within_phase=config.shuffle,
            limit=config.limit,
        )
    if kind in {
        "tau3_retail_policy_shift",
        "tau_retail_policy_shift",
        "public_retail_policy_shift",
    }:
        configured_cache = cache_dir or Path(
            config.path or "data/benchmarks/tau3_retail_policy_shift"
        )
        resolved_cache = _resolve_path(configured_cache, project_root)
        return TauRetailPolicyShiftBenchmark(
            resolved_cache,
            seed=seed,
            phase_size=config.phase_size or 24,
            feedback_noise_rate=config.feedback_noise_rate,
            feedback_attack_rate=config.feedback_attack_rate,
            feedback_shared_source=config.feedback_shared_source,
            feedback_shared_source_name=config.feedback_shared_source_name,
            feedback_attack_burst_length=config.feedback_attack_burst_length,
            policy_schedule=config.policy_schedule or None,
            coverage_balanced=config.coverage_balanced,
            coverage_min_per_slice=config.coverage_min_per_slice,
            shuffle_within_phase=config.shuffle,
            limit=config.limit,
        )
    if kind in {"bbh", "big_bench_hard"}:
        configured_cache = cache_dir or Path(config.path or "data/benchmarks/bbh")
        resolved_cache = _resolve_path(configured_cache, project_root)
        return BBHBenchmarkAdapter(
            resolved_cache,
            revision=config.revision or DEFAULT_BBH_REVISION,
            subsets=config.subsets,
            limit=config.limit,
            phase_size=config.phase_size,
            shuffle=config.shuffle,
            seed=seed,
            protected_phases=config.protected_phases,
        )
    if kind in {"hf", "huggingface", "hugging_face"}:
        return HuggingFaceBenchmarkAdapter(
            config.name,
            split=config.split,
            subsets=config.subsets,
            revision=config.revision,
            limit=config.limit,
            shuffle=config.shuffle,
            seed=seed,
            protected_phases=config.protected_phases,
        )
    raise DatasetError(f"unsupported benchmark kind: {config.kind!r}")


def load_benchmark(
    config: BenchmarkConfig,
    *,
    root: Path | None = None,
    seed: int = 42,
    cache_dir: Path | None = None,
) -> list[BenchmarkSample]:
    return create_benchmark(
        config,
        root=root,
        seed=seed,
        cache_dir=cache_dir,
    ).load()


def _resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path
