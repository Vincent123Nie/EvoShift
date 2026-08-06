from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from evoshift.agents import MemoryAgent
from evoshift.config import EvolutionConfig
from evoshift.evaluation import PromotionGate, score_feedback_sample
from evoshift.memory.retriever import jaccard_similarity, tokenize
from evoshift.schemas import Episode, MemoryItem, PolicyGenome, PromotionDecision


def _cost_proxy(tokens: int, cost_usd: float) -> float:
    # Dollar prices are often unknown for a private reverse proxy. Falling back
    # to tokens preserves an efficiency gate instead of silently treating calls
    # as free.
    return cost_usd if cost_usd > 0.0 else float(tokens)


def _attach_replay_trace(
    decision: PromotionDecision,
    episodes: Sequence[Episode],
    protected_mask: Sequence[bool],
) -> PromotionDecision:
    """Persist stable replay provenance alongside the statistical decision."""

    decision.result.replay_sample_ids = [episode.sample.sample_id for episode in episodes]
    decision.result.replay_episode_indices = [episode.index for episode in episodes]
    decision.result.replay_protected_mask = list(protected_mask)
    return decision


def _observable_replay_context(episode: Episode, context_field: str) -> tuple[str, str]:
    source = str(episode.sample.metadata.get("feedback_source", "")).strip() or "unspecified"
    configured = str(episode.sample.metadata.get(context_field, "")).strip()
    context = (
        configured.casefold() if configured else " ".join(episode.sample.prompt.casefold().split())
    )
    return source, context


def select_replay_buffer(
    episodes: Sequence[Episode],
    window: int,
    *,
    query: str = "",
    protected_phases: Sequence[str] = (),
    min_feedback_trust: float = 0.0,
    regime_start_index: int | None = None,
    historical_context_anchors_enabled: bool = False,
    historical_context_anchor_fraction: float = 1.0 / 3.0,
    observable_context_field: str = "feedback_context",
    excluded_historical_contexts: Sequence[tuple[str, str]] = (),
) -> Tuple[List[Episode], List[bool]]:
    """Build a deterministic related/observable-compatibility replay mixture.

    The safety quota first honors explicitly configured protected phases. Any
    remaining slots may hold recent representatives of distinct, learner-visible
    historical contexts. Context anchors are ordinary replay examples, not
    oracle-labelled invariant cases.
    """

    if window < 1:
        raise ValueError("replay window must be positive")
    if not 0.0 < historical_context_anchor_fraction <= 0.5:
        raise ValueError("historical context anchor fraction must be in (0, 0.5]")
    if observable_context_field != "feedback_context":
        raise ValueError(
            "observable replay context must use the typed learner-visible feedback_context field"
        )
    anchor_quota = math.floor(window * historical_context_anchor_fraction)
    if historical_context_anchors_enabled and anchor_quota < 1:
        raise ValueError("historical replay anchors require window * anchor fraction >= 1")
    if not episodes:
        return [], []
    eligible = [episode for episode in episodes if episode.feedback_trust >= min_feedback_trust]
    if not eligible:
        return [], []
    protected_set = set(protected_phases)
    protected = [episode for episode in eligible if episode.sample.phase in protected_set]
    protected_ids = {episode.episode_id for episode in protected}
    ordinary = [
        episode
        for episode in eligible
        if episode.episode_id not in protected_ids
        and (regime_start_index is None or episode.index >= regime_start_index)
    ]
    historical_context_anchors: list[Episode] = []
    remaining_historical: list[Episode] = []
    if historical_context_anchors_enabled and regime_start_index is not None:
        excluded_contexts = frozenset(excluded_historical_contexts)
        latest_by_context: dict[tuple[str, str], Episode] = {}
        historical: list[Episode] = []
        for episode in eligible:
            if episode.episode_id in protected_ids or episode.index >= regime_start_index:
                continue
            context_key = _observable_replay_context(episode, observable_context_field)
            if context_key in excluded_contexts:
                continue
            historical.append(episode)
            previous = latest_by_context.get(context_key)
            if previous is None or episode.index > previous.index:
                latest_by_context[context_key] = episode
        historical_context_anchors = sorted(
            latest_by_context.values(),
            key=lambda episode: episode.index,
            reverse=True,
        )
        anchor_ids = {episode.episode_id for episode in historical_context_anchors}
        remaining_historical = sorted(
            [episode for episode in historical if episode.episode_id not in anchor_ids],
            key=lambda episode: episode.index,
            reverse=True,
        )
    query_tokens = tokenize(query)

    def relevance(episode: Episode) -> Tuple[float, int]:
        score = jaccard_similarity(query_tokens, tokenize(episode.sample.prompt)) if query else 0.0
        return score, episode.index

    ordinary = sorted(ordinary, key=relevance, reverse=True)
    protected = sorted(protected, key=lambda episode: episode.index, reverse=True)
    safety_quota = anchor_quota
    if protected and safety_quota < 1:
        safety_quota = 1
    selected_protected = protected[:safety_quota]
    anchor_slots = max(0, safety_quota - len(selected_protected))
    selected_anchors = historical_context_anchors[:anchor_slots]
    related_quota = max(0, window - len(selected_protected) - len(selected_anchors))
    selected = ordinary[:related_quota] + selected_protected + selected_anchors
    available_by_id = {
        episode.episode_id: episode
        for episode in ordinary + protected + historical_context_anchors + remaining_historical
    }
    available = list(available_by_id.values())
    if len(selected) < min(window, len(available)):
        selected_ids = {episode.episode_id for episode in selected}
        remaining = sorted(available, key=lambda episode: episode.index, reverse=True)
        selected.extend(episode for episode in remaining if episode.episode_id not in selected_ids)
        selected = selected[: min(window, len(available))]
    selected.sort(key=lambda episode: episode.index)
    mask = [episode.sample.phase in protected_set for episode in selected]
    return selected, mask


class ReplayVerifier:
    def __init__(
        self,
        agent: MemoryAgent,
        config: EvolutionConfig,
        *,
        protected_phases: Sequence[str] = (),
    ):
        self.agent = agent
        self.config = config
        self.protected_phases = tuple(protected_phases)
        self.gate = PromotionGate(config)

    def memory_buffer(
        self,
        candidate: MemoryItem,
        episodes: Sequence[Episode],
        *,
        regime_start_index: int | None = None,
    ) -> Tuple[List[Episode], List[bool]]:
        provenance_ids = frozenset(candidate.provenance_episode_ids)
        excluded_historical_contexts = {
            _observable_replay_context(
                episode,
                self.config.dynamic_feedback_context_field,
            )
            for episode in episodes
            if episode.episode_id in provenance_ids
        }
        return select_replay_buffer(
            episodes,
            self.config.validation_window,
            query=f"{candidate.trigger} {candidate.scope}",
            protected_phases=self.protected_phases,
            min_feedback_trust=self.config.min_feedback_trust_for_replay,
            regime_start_index=(
                regime_start_index if self.config.replay_current_regime_only else None
            ),
            historical_context_anchors_enabled=(
                self.config.replay_historical_context_anchors_enabled
            ),
            historical_context_anchor_fraction=(
                self.config.replay_historical_context_anchor_fraction
            ),
            observable_context_field=self.config.dynamic_feedback_context_field,
            excluded_historical_contexts=tuple(sorted(excluded_historical_contexts)),
        )

    def policy_buffer(
        self,
        episodes: Sequence[Episode],
        *,
        regime_start_index: int | None = None,
    ) -> Tuple[List[Episode], List[bool]]:
        return select_replay_buffer(
            episodes,
            self.config.validation_window,
            query="",
            protected_phases=self.protected_phases,
            min_feedback_trust=self.config.min_feedback_trust_for_replay,
            regime_start_index=(
                regime_start_index if self.config.replay_current_regime_only else None
            ),
            historical_context_anchors_enabled=(
                self.config.replay_historical_context_anchors_enabled
            ),
            historical_context_anchor_fraction=(
                self.config.replay_historical_context_anchor_fraction
            ),
            observable_context_field=self.config.dynamic_feedback_context_field,
        )

    async def validate_memory(
        self,
        candidate: MemoryItem,
        episodes: Sequence[Episode],
        policy: PolicyGenome,
        *,
        regime_start_index: int | None = None,
    ) -> PromotionDecision:
        buffer, protected_mask = self.memory_buffer(
            candidate,
            episodes,
            regime_start_index=regime_start_index,
        )
        control_scores: List[float] = []
        candidate_scores: List[float] = []
        control_costs: List[float] = []
        candidate_costs: List[float] = []
        for episode in buffer:
            if self.config.paired_replay:
                control_prediction = await self.agent.solve(episode.sample, policy)
                control_score = score_feedback_sample(
                    episode.sample, control_prediction.output.answer
                ).primary
                control_usage = control_prediction.usage
            else:
                control_score = episode.adaptation_score.primary
                control_usage = episode.usage
            challenger_prediction = await self.agent.solve(
                episode.sample, policy, extra_memories=[candidate]
            )
            challenger_score = score_feedback_sample(
                episode.sample, challenger_prediction.output.answer
            ).primary
            control_scores.append(control_score)
            candidate_scores.append(challenger_score)
            control_costs.append(_cost_proxy(control_usage.total_tokens, control_usage.cost_usd))
            candidate_costs.append(
                _cost_proxy(
                    challenger_prediction.usage.total_tokens,
                    challenger_prediction.usage.cost_usd,
                )
            )
        return _attach_replay_trace(
            self.gate.decide(
                candidate_id=f"{candidate.memory_id}@v{candidate.version}",
                candidate_type="memory",
                control_scores=control_scores,
                candidate_scores=candidate_scores,
                control_costs=control_costs,
                candidate_costs=candidate_costs,
                protected_mask=protected_mask,
            ),
            buffer,
            protected_mask,
        )

    async def validate_policy(
        self,
        candidate: PolicyGenome,
        champion: PolicyGenome,
        episodes: Sequence[Episode],
        *,
        regime_start_index: int | None = None,
    ) -> PromotionDecision:
        buffer, protected_mask = self.policy_buffer(
            episodes,
            regime_start_index=regime_start_index,
        )
        control_scores: List[float] = []
        candidate_scores: List[float] = []
        control_costs: List[float] = []
        candidate_costs: List[float] = []
        for episode in buffer:
            control_prediction = await self.agent.solve(episode.sample, champion)
            challenger_prediction = await self.agent.solve(episode.sample, candidate)
            control_scores.append(
                score_feedback_sample(episode.sample, control_prediction.output.answer).primary
            )
            candidate_scores.append(
                score_feedback_sample(episode.sample, challenger_prediction.output.answer).primary
            )
            control_costs.append(
                _cost_proxy(
                    control_prediction.usage.total_tokens, control_prediction.usage.cost_usd
                )
            )
            candidate_costs.append(
                _cost_proxy(
                    challenger_prediction.usage.total_tokens,
                    challenger_prediction.usage.cost_usd,
                )
            )
        return _attach_replay_trace(
            self.gate.decide(
                candidate_id=f"policy-v{candidate.version}",
                candidate_type="policy",
                control_scores=control_scores,
                candidate_scores=candidate_scores,
                control_costs=control_costs,
                candidate_costs=candidate_costs,
                protected_mask=protected_mask,
            ),
            buffer,
            protected_mask,
        )
