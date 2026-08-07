from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from evoshift import __version__
from evoshift.config import EvoShiftConfig, load_config
from evoshift.evaluation import (
    infer_shift_indices,
    paired_bootstrap_ci,
    post_shift_gain,
    render_markdown_report,
    usage_metrics,
    write_json_report,
)
from evoshift.providers.factory import create_client
from evoshift.runtime.artifacts import load_episodes
from evoshift.schemas import GenerationRequest, MemoryStatus
from evoshift.storage import SQLiteStore

app = typer.Typer(
    name="evoshift",
    no_args_is_help=True,
    help="Verified test-time self-evolution for API-only LLM agents.",
)
provider_app = typer.Typer(no_args_is_help=True, help="Provider diagnostics.")
data_app = typer.Typer(no_args_is_help=True, help="Public benchmark data management.")
benchmark_app = typer.Typer(no_args_is_help=True, help="Benchmark discovery.")
memory_app = typer.Typer(no_args_is_help=True, help="Inspect versioned memory state.")
app.add_typer(provider_app, name="provider")
app.add_typer(data_app, name="data")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(memory_app, name="memory")
console = Console()


def _config(path: Optional[Path], overrides: List[str]) -> EvoShiftConfig:
    selected = path
    if selected is None:
        default = Path("configs/default.yaml")
        selected = default if default.exists() else None
    return load_config(selected, overrides)


def _fail(exc: Exception) -> None:
    console.print(f"[red]Error:[/red] {exc}")
    raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print the installed EvoShift version."""

    console.print(__version__)


@app.command()
def doctor(
    config: Optional[Path] = typer.Option(None, exists=False, help="Configuration YAML."),
) -> None:
    """Run local checks only; this command never calls a paid API."""

    try:
        resolved = _config(config, [])
        connection = sqlite3.connect(":memory:")
        fts5 = bool(
            connection.execute("SELECT sqlite_compileoption_used('ENABLE_FTS5')").fetchone()[0]
        )
        connection.close()
        table = Table(title="EvoShift doctor")
        table.add_column("Check")
        table.add_column("Result")
        table.add_row("Version", __version__)
        table.add_row("Python", sys.version.split()[0])
        table.add_row("SQLite", sqlite3.sqlite_version)
        table.add_row("SQLite FTS5", "available" if fts5 else "not required / unavailable")
        table.add_row("Provider kind", resolved.provider.kind)
        table.add_row("Model", resolved.provider.resolved_model())
        key_status = "configured" if os.getenv(resolved.provider.api_key_env) else "missing"
        table.add_row("API key", key_status)
        table.add_row("Config hash", resolved.fingerprint()[:16])
        console.print(table)
    except Exception as exc:
        _fail(exc)


@provider_app.command("smoke")
def provider_smoke(
    config: Optional[Path] = typer.Option(None, help="Configuration YAML."),
    set_value: List[str] = typer.Option([], "--set", help="Override dotted key=value."),
) -> None:
    """Make one explicit, paid Responses API request and expect exactly OK."""

    async def smoke() -> None:
        resolved = _config(config, set_value)
        client = create_client(resolved.provider)
        try:
            response = await client.generate(
                GenerationRequest(
                    model=resolved.provider.resolved_model(),
                    messages=[{"role": "user", "content": "Reply with exactly OK"}],
                    max_output_tokens=32,
                    reasoning_effort=resolved.provider.reasoning_effort,
                    metadata={"purpose": "provider_smoke"},
                )
            )
        finally:
            await client.aclose()
        table = Table(title="Provider smoke")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("Result", "PASS" if response.text.strip() == "OK" else "RESPONSE_RECEIVED")
        table.add_row("Model", response.model)
        table.add_row("Output", response.text[:100])
        table.add_row("Tokens", str(response.usage.total_tokens))
        table.add_row("Latency ms", f"{response.usage.latency_ms:.1f}")
        console.print(table)

    try:
        asyncio.run(smoke())
    except Exception as exc:
        _fail(exc)


@benchmark_app.command("list")
def benchmark_list() -> None:
    """List built-in and optional benchmark adapters."""

    table = Table(title="Benchmarks")
    table.add_column("Kind")
    table.add_column("Purpose")
    table.add_column("Network")
    rows = [
        ("synthetic_shift", "Deterministic CI and shift/rollback validation", "no"),
        ("policy_shift", "Policy updates with noisy/adversarial feedback", "no"),
        (
            "tau3_retail_policy_shift",
            "Pinned tau3 retail policy with explicit derived version overlays",
            "first pull",
        ),
        ("jsonl", "User or exported public dataset in normalized JSONL", "no"),
        ("bbh", "Pinned BIG-Bench Hard task-family shift stream", "first pull"),
        (
            "longmemeval_s",
            "Pinned public session-level long-term-memory retrieval evaluation",
            "first pull",
        ),
        ("huggingface", "Generic Hugging Face dataset field mapper", "first pull"),
    ]
    for row in rows:
        table.add_row(*row)
    console.print(table)


@data_app.command("pull")
def data_pull(
    dataset: str = typer.Argument(
        "bbh", help="Dataset kind: bbh, tau3-retail-policy, or longmemeval."
    ),
    config: Optional[Path] = typer.Option(None, help="Configuration YAML."),
    set_value: List[str] = typer.Option([], "--set", help="Override dotted key=value."),
) -> None:
    """Download and integrity-check a configured public benchmark."""

    try:
        normalized = dataset.lower().replace("_", "-")
        if normalized == "bbh":
            benchmark_kind = "bbh"
        elif normalized in {"tau3-retail-policy", "tau3-retail-policy-shift"}:
            benchmark_kind = "tau3_retail_policy_shift"
        elif normalized in {"longmemeval", "longmemeval-s"}:
            from evoshift.benchmarks.longmemeval import (
                default_longmemeval_path,
                download_longmemeval,
            )

            integrity = download_longmemeval(default_longmemeval_path(Path.cwd()))
            console.print("[green]Verified pinned LongMemEval_S file.[/green]")
            console.print(
                f"{integrity.path} bytes={integrity.size_bytes} sha256={integrity.sha256}"
            )
            return
        else:
            raise ValueError(
                "data pull currently supports 'bbh', 'tau3-retail-policy', and "
                "'longmemeval'; generic HF downloads occur on run"
            )
        resolved = _config(config, [f"benchmark.kind={benchmark_kind}", *set_value])
        from evoshift.benchmarks import create_benchmark

        adapter = create_benchmark(
            resolved.benchmark, root=Path.cwd(), seed=resolved.evaluation.seed
        )
        download = getattr(adapter, "download", None)
        if download is None:
            raise ValueError("configured benchmark has no explicit download operation")
        files = download()
        label = "BBH files" if benchmark_kind == "bbh" else "source files"
        console.print(f"[green]Verified {len(files)} {label}.[/green]")
        console.print(str(getattr(adapter, "manifest_path", "")))
    except Exception as exc:
        _fail(exc)


@benchmark_app.command("retrieval-eval")
def benchmark_retrieval_eval(
    dataset: Optional[Path] = typer.Option(
        None,
        "--dataset",
        help="Pinned LongMemEval_S JSON; defaults to data/benchmarks after data pull.",
    ),
    method: str = typer.Option(
        "bm25",
        help="Retrieval system: bm25, bm25_llm_rerank, or bm25_llm_rerank_fused.",
    ),
    config: Optional[Path] = typer.Option(None, help="Provider/budget configuration YAML."),
    set_value: List[str] = typer.Option([], "--set", help="Override dotted key=value."),
    output_root: Path = typer.Option(
        Path("runs/retrieval"), help="Root for immutable retrieval artifacts."
    ),
    limit: int = typer.Option(0, min=0, help="Maximum selected questions; zero means all."),
    max_per_type: int = typer.Option(
        0,
        min=0,
        help="Take the deterministic first N non-abstention questions per type.",
    ),
    concurrency: int = typer.Option(4, min=1, max=32, help="Maximum concurrent LLM calls."),
    candidate_k: int = typer.Option(20, min=1, max=50, help="BM25 rerank candidate pool."),
    output_k: int = typer.Option(10, min=1, max=50, help="Requested reranker prefix length."),
    max_candidate_chars: int = typer.Option(
        2_500, min=256, help="Per-session character bound in the reranker prompt."
    ),
    bm25_rank_weight: float = typer.Option(
        0.4,
        min=0.0,
        max=1.0,
        help="BM25 rank weight for bm25_llm_rerank_fused.",
    ),
) -> None:
    """Evaluate session retrieval on pinned LongMemEval without online mutation."""

    async def execute() -> None:
        from evoshift.benchmarks.longmemeval import default_longmemeval_path
        from evoshift.evaluation.longmemeval_retrieval import (
            run_longmemeval_retrieval,
            write_retrieval_artifacts,
        )
        from evoshift.runtime.artifacts import git_state
        from evoshift.runtime.budget import BudgetLedger

        resolved = _config(config, set_value)
        selected_path = dataset or default_longmemeval_path(Path.cwd())
        normalized_method = method.strip().lower().replace("-", "_")
        client = None
        budget = None
        if normalized_method in {"bm25_llm_rerank", "bm25_llm_rerank_fused"}:
            budget = BudgetLedger.from_config(resolved.budget)
            client = create_client(resolved.provider, budget=budget)
        try:
            evaluation = await run_longmemeval_retrieval(
                selected_path,
                method=normalized_method,
                client=client,
                provider=resolved.provider if client is not None else None,
                limit=limit,
                max_per_type=max_per_type,
                concurrency=concurrency,
                candidate_k=candidate_k,
                output_k=output_k,
                max_candidate_chars=max_candidate_chars,
                bm25_rank_weight=bm25_rank_weight,
            )
        finally:
            if client is not None:
                await client.aclose()
        commit, dirty = git_state(Path.cwd())
        budget_payload = budget.snapshot().as_dict() if budget is not None else None
        run_dir = write_retrieval_artifacts(
            evaluation,
            output_root,
            config_hash=resolved.fingerprint(),
            git_commit=commit,
            git_dirty=dirty,
            budget=budget_payload,
        )
        systems = evaluation.report["systems"]
        selected_system = systems.get(normalized_method, systems["bm25"])
        macro = selected_system["macro"]
        console.print(f"[green]Retrieval evaluation complete:[/green] {run_dir.name}")
        console.print(str(run_dir))
        console.print(
            f"n={evaluation.report['n_questions']} "
            f"recall_all@5={macro['recall_all@5']:.4f} "
            f"recall_all@10={macro['recall_all@10']:.4f} "
            f"ndcg_any@10={macro['ndcg_any@10']:.4f} mrr={macro['mrr']:.4f}"
        )

    try:
        asyncio.run(execute())
    except Exception as exc:
        _fail(exc)


@app.command("run")
def run_experiment(
    config: Path = typer.Option(
        Path("configs/experiments/offline_demo.yaml"), help="Experiment YAML."
    ),
    set_value: List[str] = typer.Option([], "--set", help="Override dotted key=value."),
) -> None:
    """Run a prequential benchmark and emit an immutable artifact directory."""

    async def execute() -> None:
        resolved = _config(config, set_value)
        from evoshift.benchmarks import create_benchmark
        from evoshift.runner import EvoShiftRunner

        adapter = create_benchmark(
            resolved.benchmark, root=Path.cwd(), seed=resolved.evaluation.seed
        )
        result = await EvoShiftRunner(resolved, adapter).run()
        console.print(f"[green]Run complete:[/green] {result.run_id}")
        console.print(str(result.run_dir))
        console.print(
            f"score={result.metrics['overall']['mean_score']:.4f} "
            f"success={result.metrics['overall']['success_rate']:.4f} "
            f"policy=v{result.final_policy.version}"
        )

    try:
        asyncio.run(execute())
    except Exception as exc:
        _fail(exc)


@app.command("audit")
def audit_experiment(
    source_run: Path = typer.Option(
        ...,
        "--source-run",
        exists=True,
        file_okay=False,
        help="Completed prequential run whose evolved state will be frozen.",
    ),
    config: Path = typer.Option(
        Path("configs/experiments/audit_bbh_heldout.yaml"),
        help="Held-out benchmark configuration YAML.",
    ),
    set_value: List[str] = typer.Option([], "--set", help="Override dotted key=value."),
) -> None:
    """Evaluate evolved state on a distinct stream with all mutation disabled."""

    async def execute() -> None:
        from evoshift.audit import load_evolved_state, require_same_model
        from evoshift.benchmarks import create_benchmark
        from evoshift.runner import EvoShiftRunner

        state = load_evolved_state(source_run)
        resolved = _config(config, set_value)
        require_same_model(state, resolved.provider.resolved_model())
        adapter = create_benchmark(
            resolved.benchmark,
            root=Path.cwd(),
            seed=resolved.evaluation.seed,
        )
        result = await EvoShiftRunner(
            resolved,
            adapter,
            initial_memories=state.memories,
            initial_policy=state.policy,
            frozen_audit=True,
            source_run_id=state.source_run_id,
            source_state_hash=state.fingerprint,
            source_dataset_hash=state.source_dataset_hash,
        ).run()
        console.print(f"[green]Frozen audit complete:[/green] {result.run_id}")
        console.print(str(result.run_dir))
        console.print(
            f"score={result.metrics['overall']['mean_score']:.4f} "
            f"source={state.source_run_id} state={state.fingerprint[:12]}"
        )

    try:
        asyncio.run(execute())
    except Exception as exc:
        _fail(exc)


@app.command("audit-memories")
def audit_memories_experiment(
    source_run: Path = typer.Option(
        ...,
        "--source-run",
        exists=True,
        file_okay=False,
        help="Completed prequential run whose evolved state will be frozen.",
    ),
    config: Path = typer.Option(
        Path("configs/experiments/audit_bbh_heldout.yaml"),
        help="Held-out benchmark configuration YAML.",
    ),
    set_value: List[str] = typer.Option([], "--set", help="Override dotted key=value."),
) -> None:
    """Audit the held-out contribution of every active memory card."""

    async def execute() -> None:
        from evoshift.ablation import run_memory_ablation_audit
        from evoshift.audit import load_evolved_state, require_same_model
        from evoshift.benchmarks import create_benchmark

        state = load_evolved_state(source_run)
        resolved = _config(config, set_value)
        require_same_model(state, resolved.provider.resolved_model())
        adapter = create_benchmark(
            resolved.benchmark,
            root=Path.cwd(),
            seed=resolved.evaluation.seed,
        )
        audit = await run_memory_ablation_audit(
            resolved,
            adapter,
            state,
            workdir=Path.cwd(),
        )
        summary = audit.report["summary"]
        console.print(
            f"[green]Memory ablation complete:[/green] {audit.report['full_state_run_id']}"
        )
        console.print(str(audit.markdown_path))
        console.print(
            f"cards={audit.report['n_cards_tested']} "
            f"useful={summary['useful_cards']} "
            f"harmful={summary['harmful_cards']} "
            f"inconclusive={summary['inconclusive_cards']}"
        )

    try:
        asyncio.run(execute())
    except Exception as exc:
        _fail(exc)


@app.command("sweep")
def run_experiment_sweep(
    spec: Path = typer.Option(
        Path("configs/sweeps/demo_matrix.yaml"), help="Sweep specification YAML."
    ),
) -> None:
    """Run a bounded algorithm/seed/hyperparameter matrix sequentially."""

    async def execute() -> None:
        from evoshift.sweep import expand_sweep, load_sweep_spec, run_sweep

        resolved = load_sweep_spec(spec, Path.cwd())
        run_count = len(expand_sweep(resolved))
        console.print(f"Executing {run_count} bounded sweep runs...")
        destination = await run_sweep(resolved, Path.cwd())
        console.print(f"[green]Sweep complete:[/green] {destination}")

    try:
        asyncio.run(execute())
    except Exception as exc:
        _fail(exc)


@app.command("compare")
def compare_runs(
    baseline: Path = typer.Argument(..., exists=True, file_okay=False),
    candidate: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """Compare same-sample runs with paired confidence intervals and cost deltas."""

    try:
        baseline_episodes = load_episodes(baseline / "predictions.jsonl")
        candidate_episodes = load_episodes(candidate / "predictions.jsonl")
        baseline_by_id = {episode.sample.sample_id: episode for episode in baseline_episodes}
        aligned_baseline = [
            baseline_by_id[episode.sample.sample_id] for episode in candidate_episodes
        ]
        if len(aligned_baseline) != len(baseline_episodes):
            raise ValueError("runs do not contain the same unique sample ids")
        baseline_scores = [episode.score.primary for episode in aligned_baseline]
        candidate_scores = [episode.score.primary for episode in candidate_episodes]
        deltas = [new - old for old, new in zip(baseline_scores, candidate_scores)]
        ci_low, ci_high = paired_bootstrap_ci(deltas)
        shifts = infer_shift_indices(candidate_episodes)
        baseline_usage = usage_metrics(aligned_baseline)
        candidate_usage = usage_metrics(candidate_episodes)
        baseline_manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
        candidate_manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
        baseline_total = json.loads((baseline / "costs.json").read_text(encoding="utf-8"))
        candidate_total = json.loads((candidate / "costs.json").read_text(encoding="utf-8"))
        baseline_config = load_config(baseline / "resolved_config.yaml")
        candidate_config = load_config(candidate / "resolved_config.yaml")
        same_model = baseline_manifest.get("model") == candidate_manifest.get("model")
        same_dataset = baseline_manifest.get("dataset_hash") == candidate_manifest.get(
            "dataset_hash"
        )
        same_git_commit = baseline_manifest.get("git_commit") == candidate_manifest.get(
            "git_commit"
        )
        clean_worktrees = not bool(baseline_manifest.get("git_dirty")) and not bool(
            candidate_manifest.get("git_dirty")
        )
        provenance_comparable = same_git_commit and clean_worktrees
        same_cache_policy = (
            baseline_config.storage.cache_enabled == candidate_config.storage.cache_enabled
        )
        cache_disabled_for_both = (
            not baseline_config.storage.cache_enabled and not candidate_config.storage.cache_enabled
        )
        accuracy_comparable = same_model and same_dataset
        resource_comparable = (
            accuracy_comparable and cache_disabled_for_both and provenance_comparable
        )
        comparison_warnings = []
        if not same_model:
            comparison_warnings.append("model identifiers differ")
        if not same_dataset:
            comparison_warnings.append("dataset hashes differ")
        if not same_git_commit:
            comparison_warnings.append("Git commits differ")
        if not clean_worktrees:
            comparison_warnings.append(
                "at least one run used a dirty worktree; formal reproducibility claims are invalid"
            )
        if not cache_disabled_for_both:
            comparison_warnings.append(
                "LLM cache was enabled for at least one run; cost and latency claims are invalid"
            )
        if baseline_usage["cached_episodes"] != candidate_usage["cached_episodes"]:
            comparison_warnings.append(
                "foreground cache-hit counts differ, so run order affected measured resources"
            )
        incremental_tokens = float(candidate_total.get("total_tokens", 0)) - float(
            baseline_total.get("total_tokens", 0)
        )
        mean_gain = sum(deltas) / len(deltas)
        report = {
            "n": len(deltas),
            "baseline_run": baseline.name,
            "candidate_run": candidate.name,
            "baseline_mean": sum(baseline_scores) / len(baseline_scores),
            "candidate_mean": sum(candidate_scores) / len(candidate_scores),
            "mean_gain": mean_gain,
            "paired_bootstrap_95_ci": [ci_low, ci_high],
            "post_shift_gain": post_shift_gain(candidate_episodes, aligned_baseline, shifts),
            "shift_indices": shifts,
            "baseline_resources": baseline_usage,
            "candidate_resources": candidate_usage,
            "baseline_total_budget": baseline_total,
            "candidate_total_budget": candidate_total,
            "incremental_total_tokens": incremental_tokens,
            "accuracy_gain_per_1m_incremental_tokens": (
                mean_gain / (incremental_tokens / 1_000_000.0) if incremental_tokens > 0 else None
            ),
            "same_model": same_model,
            "same_dataset_hash": same_dataset,
            "same_git_commit": same_git_commit,
            "clean_worktrees": clean_worktrees,
            "provenance_comparable": provenance_comparable,
            "same_cache_policy": same_cache_policy,
            "cache_disabled_for_both": cache_disabled_for_both,
            "accuracy_comparable": accuracy_comparable,
            "resource_comparable": resource_comparable,
            "comparison_warnings": comparison_warnings,
            "comparable_for_claims": resource_comparable,
            "same_sample_comparison": True,
        }
        write_json_report(report, candidate / f"compare_vs_{baseline.name}.json")
        markdown = [
            "# EvoShift paired run comparison",
            "",
            f"- Baseline: `{baseline.name}`",
            f"- Candidate: `{candidate.name}`",
            f"- Samples: {len(deltas)}",
            f"- Mean gain: {report['mean_gain']:.4f}",
            f"- Paired bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]",
            f"- Post-shift gain: {report['post_shift_gain']:.4f}",
            f"- Same model: {same_model}",
            f"- Same dataset hash: {same_dataset}",
            f"- Same Git commit: {same_git_commit}",
            f"- Clean worktrees: {clean_worktrees}",
            f"- Baseline cached episodes: {baseline_usage['cached_episodes']}",
            f"- Candidate cached episodes: {candidate_usage['cached_episodes']}",
            f"- Accuracy comparable: {accuracy_comparable}",
            f"- Resource comparable: {resource_comparable}",
            f"- Incremental total tokens: {incremental_tokens:.0f}",
            "",
            *(
                ["## Warnings", "", *[f"- {item}" for item in comparison_warnings], ""]
                if comparison_warnings
                else []
            ),
            "Different models, prompts, budgets, cache protocols, or sample sets must not "
            "be presented as a SOTA claim.",
            "",
        ]
        (candidate / f"compare_vs_{baseline.name}.md").write_text(
            "\n".join(markdown), encoding="utf-8"
        )
        console.print(json.dumps(report, indent=2, ensure_ascii=False))
    except Exception as exc:
        _fail(exc)


@app.command("report")
def regenerate_report(run_dir: Path = typer.Argument(..., exists=True, file_okay=False)) -> None:
    """Regenerate report.md from metrics.json."""

    try:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        (run_dir / "report.md").write_text(
            render_markdown_report(metrics, title=f"EvoShift run {run_dir.name}"),
            encoding="utf-8",
        )
        console.print(str(run_dir / "report.md"))
    except Exception as exc:
        _fail(exc)


@memory_app.command("inspect")
def memory_inspect(
    database: Path = typer.Option(..., exists=True, dir_okay=False),
    status: str = typer.Option("active", help="active, shadow, rejected, retired, or all"),
) -> None:
    """Inspect typed memories without printing provider secrets or raw prompts."""

    try:
        store = SQLiteStore(database)
        try:
            statuses = None if status == "all" else [MemoryStatus(status)]
            memories = store.list_memories(statuses)
        finally:
            store.close()
        table = Table(title=f"Memories: {status}")
        for column in ("ID", "Version", "Status", "Utility", "Uses", "Trigger", "Directive"):
            table.add_column(column)
        for item in memories:
            table.add_row(
                item.memory_id,
                str(item.version),
                item.status.value,
                f"{item.posterior_utility:.3f}",
                str(item.use_count),
                item.trigger[:60],
                item.directive[:80],
            )
        console.print(table)
    except Exception as exc:
        _fail(exc)


if __name__ == "__main__":
    app()
