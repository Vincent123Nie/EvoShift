"""Independent held-out, leave-one-memory-out audits.

The prequential learner produces one immutable evolved state.  This module
evaluates that state on a fixed held-out stream, then repeats the same frozen
evaluation after removing exactly one active memory card at a time.  All
comparisons are paired by ``sample_id`` so the result is a contribution audit,
not a second training run.
"""

from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from evoshift.audit import EvolvedState, state_fingerprint
from evoshift.benchmarks.base import BenchmarkAdapter, sample_fingerprint
from evoshift.config import EvoShiftConfig
from evoshift.evaluation.reporting import write_json_report
from evoshift.evaluation.statistics import paired_bootstrap_ci
from evoshift.runtime.artifacts import load_episodes
from evoshift.schemas import BenchmarkSample, MemoryItem, MemoryStatus


class FixedBenchmarkAdapter(BenchmarkAdapter):
    """Replay one normalized held-out sample list byte-for-byte across runs."""

    def __init__(self, samples: Sequence[BenchmarkSample]):
        copied = tuple(sample.model_copy(deep=True) for sample in samples)
        if not copied:
            raise ValueError("held-out benchmark must contain at least one sample")
        ids = [sample.sample_id for sample in copied]
        if len(ids) != len(set(ids)):
            raise ValueError("held-out benchmark contains duplicate sample ids")
        self._samples = copied

    def load(self) -> list[BenchmarkSample]:
        return [sample.model_copy(deep=True) for sample in self._samples]


@dataclass(frozen=True)
class MemoryAblationCard:
    memory_id: str
    version: int
    trigger: str
    directive: str
    tags: tuple[str, ...]
    source_domains: tuple[str, ...]

    @classmethod
    def from_item(cls, item: MemoryItem) -> "MemoryAblationCard":
        return cls(
            memory_id=item.memory_id,
            version=item.version,
            trigger=item.trigger,
            directive=item.directive,
            tags=tuple(item.tags),
            source_domains=tuple(item.source_domains),
        )

    @property
    def key(self) -> tuple[str, int]:
        return self.memory_id, self.version


@dataclass(frozen=True)
class MemoryAblationAudit:
    """JSON-ready audit output plus the locations of all frozen runs."""

    report_path: Path
    markdown_path: Path
    report: dict[str, Any]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _episode_map(episodes: Sequence[Any], *, label: str) -> dict[str, Any]:
    mapped = {episode.sample.sample_id: episode for episode in episodes}
    if len(mapped) != len(episodes):
        raise ValueError(f"{label} contains duplicate sample ids")
    return mapped


def _slice_metrics(
    full: Sequence[Any], ablated: Sequence[Any], predicate: Callable[[Any], bool]
) -> dict[str, Any]:
    indices = [index for index, episode in enumerate(full) if predicate(episode)]
    if not indices:
        return {
            "n": 0,
            "full_mean_score": None,
            "ablated_mean_score": None,
            "score_delta_full_minus_ablated": None,
            "full_success_rate": None,
            "ablated_success_rate": None,
            "success_delta_full_minus_ablated": None,
            "paired_bootstrap_95_ci": None,
        }
    full_scores = [float(full[index].score.primary) for index in indices]
    ablated_scores = [float(ablated[index].score.primary) for index in indices]
    deltas = [new - old for old, new in zip(ablated_scores, full_scores)]
    ci_low, ci_high = paired_bootstrap_ci(deltas)
    return {
        "n": len(indices),
        "full_mean_score": statistics.fmean(full_scores),
        "ablated_mean_score": statistics.fmean(ablated_scores),
        "score_delta_full_minus_ablated": statistics.fmean(deltas),
        "full_success_rate": statistics.fmean(
            float(full[index].score.success) for index in indices
        ),
        "ablated_success_rate": statistics.fmean(
            float(ablated[index].score.success) for index in indices
        ),
        "success_delta_full_minus_ablated": statistics.fmean(
            float(full[index].score.success) - float(ablated[index].score.success)
            for index in indices
        ),
        "paired_bootstrap_95_ci": [ci_low, ci_high],
    }


def _card_result(
    card: MemoryAblationCard,
    full: Sequence[Any],
    ablated: Sequence[Any],
) -> dict[str, Any]:
    selected = sum(card.memory_id in episode.selected_memory_ids for episode in full)
    applied = sum(card.memory_id in episode.output.applied_memory_ids for episode in full)
    coverage = selected / len(full) if full else 0.0
    application = applied / len(full) if full else 0.0
    all_slice = _slice_metrics(full, ablated, lambda _: True)
    protected = _slice_metrics(
        full,
        ablated,
        lambda episode: bool(episode.sample.metadata.get("protected")),
    )
    changed = _slice_metrics(
        full,
        ablated,
        lambda episode: bool(
            episode.sample.metadata.get(
                "transition_case", episode.sample.metadata.get("policy_changed_case")
            )
        ),
    )
    future = _slice_metrics(
        full,
        ablated,
        lambda episode: bool(episode.sample.metadata.get("future_change_case")),
    )
    protected_delta = protected["score_delta_full_minus_ablated"]
    mean_delta = all_slice["score_delta_full_minus_ablated"]
    protected_regression = (
        max(0.0, -float(protected_delta)) if protected_delta is not None else None
    )
    useful = bool(
        coverage > 0.0
        and application > 0.0
        and mean_delta is not None
        and float(mean_delta) > 0.0
        and all_slice["paired_bootstrap_95_ci"] is not None
        and float(all_slice["paired_bootstrap_95_ci"][0]) > 0.0
        and (protected_regression is None or protected_regression <= 0.0)
    )
    if coverage <= 0.0:
        reason = "not_tested:no_full_state_retrieval"
    elif application <= 0.0:
        reason = "not_tested:no_full_state_application"
    elif useful:
        reason = "useful:positive_paired_gain_without_protected_regression"
    elif mean_delta is not None and float(mean_delta) < 0.0:
        reason = "harmful:leave_one_out_improves_mean_score"
    else:
        reason = "inconclusive:paired_gain_gate_not_met"
    return {
        "memory_id": card.memory_id,
        "version": card.version,
        "trigger": card.trigger,
        "directive": card.directive,
        "tags": list(card.tags),
        "source_domains": list(card.source_domains),
        "retrieval_count_full_state": selected,
        "application_count_full_state": applied,
        "heldout_retrieval_coverage": coverage,
        "heldout_application_coverage": application,
        "overall": all_slice,
        "protected": protected,
        "changed": changed,
        "future_change": future,
        "protected_regression": protected_regression,
        "useful_card": useful,
        "decision": reason,
    }


def _render_ablation_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# EvoShift held-out memory ablation",
        "",
        f"- Source run: `{report['source_run_id']}`",
        f"- Full-state run: `{report['full_state_run_id']}`",
        f"- Held-out dataset hash: `{report['heldout_dataset_hash']}`",
        f"- Cards tested: {report['n_cards_tested']}",
        f"- Useful cards: {report['summary']['useful_cards']}",
        f"- Harmful cards: {report['summary']['harmful_cards']}",
        f"- Inconclusive cards: {report['summary']['inconclusive_cards']}",
        "",
        "| Memory | Ver. | Coverage | Full | LOO | delta | 95% CI | Decision |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for card in report["cards"]:
        overall = card["overall"]
        ci = overall["paired_bootstrap_95_ci"]
        ci_text = "N/A" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"

        def fmt(value: Any) -> str:
            return "N/A" if value is None else f"{float(value):.4f}"

        lines.append(
            "| `{}` | {} | {:.3f} | {} | {} | {} | {} | {} |".format(
                card["memory_id"],
                card["version"],
                card["heldout_application_coverage"],
                fmt(overall["full_mean_score"]),
                fmt(overall["ablated_mean_score"]),
                fmt(overall["score_delta_full_minus_ablated"]),
                ci_text,
                card["decision"],
            )
        )
    lines.extend(
        [
            "",
            "The full state is the paired control for every leave-one-out run. "
            "A positive Δ means the removed card helped the frozen held-out score.",
        ]
    )
    return "\n".join(lines) + "\n"


async def run_memory_ablation_audit(
    config: EvoShiftConfig,
    adapter: BenchmarkAdapter,
    state: EvolvedState,
    *,
    workdir: Path | None = None,
    client_factory: Callable[[], Any] | None = None,
) -> MemoryAblationAudit:
    """Run full-state and leave-one-memory-out frozen audits.

    ``client_factory`` is intended for deterministic tests (for example, a new
    demo client per condition). Production callers normally leave it unset so
    each runner creates the configured provider client itself.
    """

    if config.storage.cache_enabled:
        raise ValueError("memory ablation requires storage.cache_enabled=false")
    root = (workdir or Path.cwd()).resolve()
    samples = adapter.load()
    fixed_adapter = FixedBenchmarkAdapter(samples)
    heldout_hash = sample_fingerprint(samples)
    cards = [
        MemoryAblationCard.from_item(item)
        for item in sorted(state.memories, key=lambda item: (item.memory_id, item.version))
        if item.status == MemoryStatus.ACTIVE
    ]
    if not cards:
        raise ValueError("source evolved state has no active memory cards")

    from evoshift.runner import EvoShiftRunner

    def make_runner(
        memories: Sequence[MemoryItem],
        *,
        variant: str,
        excluded: MemoryAblationCard | None,
        expected_hash: str,
    ) -> EvoShiftRunner:
        return EvoShiftRunner(
            config,
            fixed_adapter,
            workdir=root,
            client=client_factory() if client_factory is not None else None,
            initial_memories=memories,
            initial_policy=state.policy,
            frozen_audit=True,
            source_run_id=state.source_run_id,
            source_state_hash=state.fingerprint,
            source_dataset_hash=state.source_dataset_hash,
            expected_state_hash=expected_hash,
            audit_variant=variant,
            excluded_memory_id=excluded.memory_id if excluded else "",
            excluded_memory_version=excluded.version if excluded else None,
        )

    full_result = await make_runner(
        state.memories,
        variant="full_state",
        excluded=None,
        expected_hash=state.fingerprint,
    ).run()
    full_episodes = load_episodes(full_result.run_dir / "predictions.jsonl")
    full_ids = [episode.sample.sample_id for episode in full_episodes]
    if full_ids != [sample.sample_id for sample in samples]:
        raise ValueError("full-state audit changed held-out sample ordering")

    card_results: list[dict[str, Any]] = []
    for card in cards:
        ablated_memories = tuple(
            item for item in state.memories if (item.memory_id, item.version) != card.key
        )
        expected_hash = state_fingerprint(state.policy, ablated_memories)
        result = await make_runner(
            ablated_memories,
            variant="leave_one_memory_out",
            excluded=card,
            expected_hash=expected_hash,
        ).run()
        ablated_episodes = load_episodes(result.run_dir / "predictions.jsonl")
        ablated_map = _episode_map(ablated_episodes, label=f"ablation {card.memory_id}")
        full_map = _episode_map(full_episodes, label="full-state audit")
        if set(full_map) != set(ablated_map):
            raise ValueError(f"sample ids differ for memory ablation {card.memory_id}")
        aligned_full = [full_map[sample.sample_id] for sample in samples]
        aligned_ablated = [ablated_map[sample.sample_id] for sample in samples]
        card_results.append(
            {
                **_card_result(card, aligned_full, aligned_ablated),
                "run_id": result.run_id,
                "run_dir": str(result.run_dir),
                "state_fingerprint_without_card": expected_hash,
                "state_unchanged": bool(result.metrics["audit"]["state_unchanged"]),
                "sample_ids_aligned": True,
            }
        )

    useful = sum(bool(card["useful_card"]) for card in card_results)
    harmful = sum(card["decision"].startswith("harmful:") for card in card_results)
    inconclusive = len(card_results) - useful - harmful
    report = {
        "protocol": "heldout_per_memory_leave_one_out_v1",
        "source_run_id": state.source_run_id,
        "source_state_hash": state.fingerprint,
        "source_dataset_hash": state.source_dataset_hash,
        "heldout_dataset_hash": heldout_hash,
        "model": config.provider.resolved_model(),
        "cache_enabled": config.storage.cache_enabled,
        "full_state_run_id": full_result.run_id,
        "full_state_run_dir": str(full_result.run_dir),
        "n_cards_source": len(state.memories),
        "n_cards_tested": len(card_results),
        "sample_count": len(samples),
        "sample_ids_aligned": True,
        "all_frozen_states_unchanged": all(card["state_unchanged"] for card in card_results)
        and bool(full_result.metrics["audit"]["state_unchanged"]),
        "cards": card_results,
        "summary": {
            "useful_cards": useful,
            "harmful_cards": harmful,
            "inconclusive_cards": inconclusive,
            "tested_card_precision": useful / len(card_results) if card_results else 0.0,
            "mean_useful_card_delta": (
                statistics.fmean(
                    card["overall"]["score_delta_full_minus_ablated"]
                    for card in card_results
                    if card["useful_card"]
                )
                if useful
                else None
            ),
        },
    }
    report_root = _resolve(root, config.storage.runs_dir)
    report_dir = report_root / (
        "memory-ablation-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    report_dir.mkdir(parents=True, exist_ok=False)
    report_path = write_json_report(report, report_dir / "memory_ablation_report.json")
    markdown_path = report_dir / "memory_ablation_report.md"
    markdown_path.write_text(_render_ablation_markdown(report), encoding="utf-8")
    return MemoryAblationAudit(report_path, markdown_path, report)


__all__ = [
    "FixedBenchmarkAdapter",
    "MemoryAblationAudit",
    "MemoryAblationCard",
    "run_memory_ablation_audit",
]
