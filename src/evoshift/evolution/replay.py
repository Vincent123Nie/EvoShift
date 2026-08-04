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


def select_replay_buffer(
    episodes: Sequence[Episode],
    window: int,
    *,
    query: str = "",
    protected_phases: Sequence[str] = (),
) -> Tuple[List[Episode], List[bool]]:
    """Build a deterministic related/protected replay mixture.

    Roughly two thirds of the buffer are relevance-ranked recent episodes and
    the remainder protects old phases. This is intentionally explicit so the
    promotion result cannot be cherry-picked by an LLM.
    """

    if window < 1:
        raise ValueError("replay window must be positive")
    if not episodes:
        return [], []
    protected_set = set(protected_phases)
    protected = [
        episode
        for episode in episodes
        if episode.sample.phase in protected_set or bool(episode.sample.metadata.get("protected"))
    ]
    ordinary = [episode for episode in episodes if episode not in protected]
    query_tokens = tokenize(query)

    def relevance(episode: Episode) -> Tuple[float, int]:
        score = jaccard_similarity(query_tokens, tokenize(episode.sample.prompt)) if query else 0.0
        return score, episode.index

    ordinary = sorted(ordinary, key=relevance, reverse=True)
    protected = sorted(protected, key=lambda episode: episode.index, reverse=True)
    protected_quota = min(len(protected), max(1, math.floor(window / 3)))
    related_quota = max(0, window - protected_quota)
    selected = ordinary[:related_quota] + protected[:protected_quota]
    if len(selected) < min(window, len(episodes)):
        selected_ids = {episode.episode_id for episode in selected}
        remaining = sorted(episodes, key=lambda episode: episode.index, reverse=True)
        selected.extend(episode for episode in remaining if episode.episode_id not in selected_ids)
        selected = selected[:window]
    selected.sort(key=lambda episode: episode.index)
    mask = [
        episode.sample.phase in protected_set or bool(episode.sample.metadata.get("protected"))
        for episode in selected
    ]
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

    async def validate_memory(
        self,
        candidate: MemoryItem,
        episodes: Sequence[Episode],
        policy: PolicyGenome,
    ) -> PromotionDecision:
        buffer, protected_mask = select_replay_buffer(
            episodes,
            self.config.validation_window,
            query=f"{candidate.trigger} {candidate.scope}",
            protected_phases=self.protected_phases,
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
        return self.gate.decide(
            candidate_id=f"{candidate.memory_id}@v{candidate.version}",
            candidate_type="memory",
            control_scores=control_scores,
            candidate_scores=candidate_scores,
            control_costs=control_costs,
            candidate_costs=candidate_costs,
            protected_mask=protected_mask,
        )

    async def validate_policy(
        self,
        candidate: PolicyGenome,
        champion: PolicyGenome,
        episodes: Sequence[Episode],
    ) -> PromotionDecision:
        buffer, protected_mask = select_replay_buffer(
            episodes,
            self.config.validation_window,
            query="",
            protected_phases=self.protected_phases,
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
        return self.gate.decide(
            candidate_id=f"policy-v{candidate.version}",
            candidate_type="policy",
            control_scores=control_scores,
            candidate_scores=candidate_scores,
            control_costs=control_costs,
            candidate_costs=candidate_costs,
            protected_mask=protected_mask,
        )
