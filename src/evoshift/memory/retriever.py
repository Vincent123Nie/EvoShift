from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set

from evoshift.schemas import MemoryItem, PolicyGenome, RetrievedMemory

TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.IGNORECASE)


def tokenize(text: str) -> List[str]:
    """Tokenize without model weights; add CJK bigrams for useful local retrieval."""

    raw = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    tokens = list(raw)
    cjk_run: List[str] = []
    for token in [*raw, "<boundary>"]:
        if len(token) == 1 and "\u4e00" <= token <= "\u9fff":
            cjk_run.append(token)
            continue
        if len(cjk_run) > 1:
            tokens.extend(a + b for a, b in zip(cjk_run, cjk_run[1:]))
        cjk_run = []
    return tokens


def memory_document(item: MemoryItem) -> str:
    return " ".join(
        part
        for part in [
            item.trigger,
            item.scope,
            " ".join(item.tags),
            item.directive,
            item.anti_pattern,
        ]
        if part
    )


def jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def normalize_domain(domain: str) -> str:
    """Canonicalize dataset domain labels without knowing a benchmark taxonomy."""

    return "_".join(tokenize(domain))


def domain_matches(item: MemoryItem, domain: str) -> bool:
    """Return whether a query belongs to one of a memory's provenance domains."""

    query_domain = normalize_domain(domain)
    if not query_domain or not item.source_domains:
        return False
    return query_domain in {normalize_domain(value) for value in item.source_domains}


class BM25MemoryRetriever:
    """Sparse retrieval combined with online utility/UCB and MMR diversity."""

    def score(
        self,
        query: str,
        items: Sequence[MemoryItem],
        policy: PolicyGenome,
        *,
        domain: str = "",
    ) -> List[RetrievedMemory]:
        eligible_items = [
            item
            for item in items
            if (
                not domain
                or not item.source_domains
                or policy.allow_cross_domain_transfer
                or domain_matches(item, domain)
            )
        ]
        if not eligible_items:
            return []
        documents = [tokenize(memory_document(item)) for item in eligible_items]
        query_terms = list(dict.fromkeys(tokenize(query)))
        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(set(document))
        average_length = sum(len(document) for document in documents) / max(1, len(documents))
        total_uses = sum(item.use_count for item in eligible_items)

        raw_scores: List[float] = []
        for document in documents:
            frequencies = Counter(document)
            length_norm = (
                1.0 - policy.bm25_b + policy.bm25_b * (len(document) / max(1.0, average_length))
            )
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                frequency_docs = document_frequency[term]
                inverse_document_frequency = math.log(
                    1.0 + (len(documents) - frequency_docs + 0.5) / (frequency_docs + 0.5)
                )
                score += (
                    inverse_document_frequency
                    * (frequency * (policy.bm25_k1 + 1.0))
                    / (frequency + policy.bm25_k1 * length_norm)
                )
            raw_scores.append(score)

        max_raw = max(raw_scores, default=0.0)
        results: List[RetrievedMemory] = []
        for item, raw_score in zip(eligible_items, raw_scores):
            if raw_score <= 0.0 and not domain_matches(item, domain):
                continue
            relevance = raw_score / max_raw if max_raw > 0.0 else 0.0
            utility = item.posterior_utility
            exploration = min(
                1.0,
                math.sqrt(2.0 * math.log(total_uses + 2.0) / (item.use_count + 1.0)) / 2.0,
            )
            final = (
                policy.relevance_weight * relevance
                + policy.utility_weight * utility
                + policy.exploration_weight * exploration
            )
            results.append(
                RetrievedMemory(
                    item=item,
                    relevance=relevance,
                    utility=utility,
                    exploration=exploration,
                    final_score=final,
                )
            )
        return sorted(results, key=lambda result: (-result.final_score, result.item.memory_id))

    def retrieve(
        self,
        query: str,
        items: Sequence[MemoryItem],
        policy: PolicyGenome,
        *,
        domain: str = "",
    ) -> List[RetrievedMemory]:
        candidates = self.score(query, items, policy, domain=domain)
        if policy.top_k <= 0:
            return []
        selected: List[RetrievedMemory] = []
        candidate_tokens: Dict[str, Set[str]] = {
            candidate.item.memory_id: set(tokenize(memory_document(candidate.item)))
            for candidate in candidates
        }
        remaining = list(candidates)
        while remaining and len(selected) < policy.top_k:
            best: Optional[RetrievedMemory] = None
            best_mmr = float("-inf")
            for candidate in remaining:
                redundancy = 0.0
                if selected:
                    redundancy = max(
                        jaccard_similarity(
                            candidate_tokens[candidate.item.memory_id],
                            candidate_tokens[chosen.item.memory_id],
                        )
                        for chosen in selected
                    )
                mmr_score = (
                    policy.mmr_lambda * candidate.final_score
                    - (1.0 - policy.mmr_lambda) * redundancy
                )
                if mmr_score > best_mmr:
                    best, best_mmr = candidate, mmr_score
            if best is None:
                break
            selected.append(best)
            remaining.remove(best)
        return selected

    def novelty(
        self,
        query: str,
        items: Sequence[MemoryItem],
        policy: PolicyGenome,
        *,
        domain: str = "",
    ) -> float:
        scored = self.score(query, items, policy, domain=domain)
        if not scored:
            return 1.0
        return max(0.0, min(1.0, 1.0 - max(result.relevance for result in scored)))


def render_memory_context(retrieved: Sequence[RetrievedMemory], token_budget: int) -> str:
    """Render auditable memory cards under a conservative character token estimate."""

    if token_budget <= 0 or not retrieved:
        return ""
    max_characters = token_budget * 4
    sections: List[str] = []
    used = 0
    for result in retrieved:
        item = result.item
        section = (
            f"[experience:{item.memory_id}@v{item.version}]\n"
            f"When: {item.trigger}\n"
            f"Do: {item.directive}\n"
        )
        if item.anti_pattern:
            section += f"Avoid: {item.anti_pattern}\n"
        if used + len(section) > max_characters:
            break
        sections.append(section.rstrip())
        used += len(section)
    return "\n\n".join(sections)
