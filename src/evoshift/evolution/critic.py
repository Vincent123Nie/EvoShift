from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional

from evoshift.config import EvolutionConfig, ProviderConfig
from evoshift.evolution.candidates import observable_candidate_cluster_key
from evoshift.schemas import (
    Episode,
    FailureRecord,
    FailureType,
    GenerationRequest,
    MemoryItem,
    MemoryKind,
    MemoryStatus,
    ShiftReport,
)

JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> Dict[str, Any]:
    fenced = JSON_FENCE.search(text)
    candidates = [fenced.group(1)] if fenced else []
    start = text.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("model output does not contain a valid JSON object")


class ExperienceCritic:
    """Turns outcome feedback into a typed, bounded experience candidate."""

    def __init__(self, client: Any, provider: ProviderConfig, evolution: EvolutionConfig):
        self.client = client
        self.provider = provider
        self.evolution = evolution

    async def analyze(
        self,
        episode: Episode,
        active_memories: List[MemoryItem],
        shift: Optional[ShiftReport] = None,
    ) -> FailureRecord:
        visible_feedback = self._visible_feedback(episode)
        memory_summary = [
            {
                "id": item.memory_id,
                "trigger": item.trigger,
                "scope": item.scope,
                "directive": item.directive,
                "tags": item.tags,
            }
            for item in active_memories[:8]
        ]
        payload = {
            "task": episode.sample.prompt,
            "domain": episode.sample.domain,
            "agent_answer": episode.output.answer,
            "confidence": episode.output.confidence,
            "rationale_summary": episode.output.rationale_summary,
            "reward": episode.adaptation_score.primary,
            "feedback": visible_feedback,
            "retrieved_memory_ids": episode.selected_memory_ids,
            "active_memories": memory_summary,
            "shift": shift.model_dump(mode="json") if shift else None,
        }
        system = (
            "You are EvoShift's failure attribution and experience distillation module. "
            "Return one compact JSON object only. Never output hidden chain-of-thought. "
            "Infer a reusable procedural lesson, not the answer to this individual task. "
            "Do not propose code or tool execution. Allowed failure_type values: "
            + ", ".join(item.value for item in FailureType)
            + ". Required schema: "
            '{"failure_type":"...","signature":"...","evidence":"...",'
            '"confidence":0.0,"memory":{"kind":"procedural","trigger":"...",'
            '"scope":"...","directive":"...","anti_pattern":"...",'
            '"evidence":"...","tags":["..."],'
            '"supersedes_memory_ids":["active-memory-id"]}}. '
            "List a superseded ID only when the new rule explicitly replaces or fully "
            "subsumes that active rule; otherwise return an empty list."
        )
        response = await self.client.generate(
            GenerationRequest(
                model=self.provider.resolved_model(),
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": "<EVOSHIFT_CRITIQUE>\n"
                        + json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    },
                ],
                max_output_tokens=min(700, self.provider.max_output_tokens),
                reasoning_effort=self.provider.reasoning_effort,
                metadata={"purpose": "experience_critic", "episode_id": episode.episode_id},
            )
        )
        try:
            return self._parse(response.text, episode, active_memories)
        except (ValueError, TypeError, KeyError):
            return self._fallback(episode)

    def _visible_feedback(self, episode: Episode) -> str:
        if self.evolution.feedback_mode == "reward_only":
            return "success" if episode.adaptation_score.success else "failure"
        if self.evolution.feedback_mode == "grader_feedback":
            return episode.adaptation_score.feedback
        return json.dumps(
            {
                "grader_feedback": episode.adaptation_score.feedback,
                "reference": episode.sample.reference,
            },
            ensure_ascii=False,
        )

    def _parse(
        self,
        text: str,
        episode: Episode,
        active_memories: List[MemoryItem],
    ) -> FailureRecord:
        data = extract_json_object(text)
        memory_data = data["memory"]
        confidence = float(data.get("confidence", memory_data.get("confidence", 0.5)))
        try:
            failure_type = FailureType(str(data.get("failure_type", "unknown")))
        except ValueError:
            failure_type = FailureType.UNKNOWN
        try:
            kind = MemoryKind(str(memory_data.get("kind", "procedural")))
        except ValueError:
            kind = MemoryKind.PROCEDURAL
        tags = [str(tag)[:80] for tag in memory_data.get("tags", [])[:20]]
        learner_visible_tag = str(
            episode.sample.metadata.get("learner_visible_memory_tag", "")
        ).strip()
        if learner_visible_tag:
            tags.append(learner_visible_tag[:80])
        memory = MemoryItem(
            memory_id="",
            kind=kind,
            status=MemoryStatus.SHADOW,
            trigger=str(memory_data["trigger"]).strip(),
            scope=str(memory_data.get("scope", episode.sample.domain)).strip(),
            directive=str(memory_data["directive"]).strip(),
            anti_pattern=str(memory_data.get("anti_pattern", "")).strip(),
            evidence=str(memory_data.get("evidence", data.get("evidence", ""))).strip(),
            tags=list(dict.fromkeys(tags))[:20],
            source_domains=[episode.sample.domain],
            provenance_episode_ids=[episode.episode_id],
            evidence_cluster_key=self._cluster_key(episode),
            supersedes_memory_ids=[
                str(memory_id)
                for memory_id in memory_data.get("supersedes_memory_ids", [])[:20]
                if str(memory_id) in {item.memory_id for item in active_memories}
                and str(memory_id) != str(memory_data.get("memory_id", ""))
            ],
            valid_from_episode_id=episode.episode_id,
            valid_from_index=episode.index,
            confidence=max(0.0, min(1.0, confidence)),
        )
        return FailureRecord(
            failure_id=f"fail-{uuid.uuid4().hex[:16]}",
            episode_id=episode.episode_id,
            failure_type=failure_type,
            signature=str(data.get("signature", "unspecified failure"))[:1200],
            evidence=str(data.get("evidence", ""))[:1200],
            proposed_memory=memory,
            confidence=max(0.0, min(1.0, confidence)),
        )

    def _fallback(self, episode: Episode) -> FailureRecord:
        candidate = MemoryItem(
            memory_id="",
            status=MemoryStatus.SHADOW,
            trigger=f"Tasks similar to {episode.sample.domain}",
            scope=episode.sample.domain,
            directive="Re-check the task constraints and verify the final answer format.",
            anti_pattern="Do not repeat an answer without validating it against the prompt.",
            evidence="Fallback because the critic response was not valid structured JSON.",
            source_domains=[episode.sample.domain],
            provenance_episode_ids=[episode.episode_id],
            evidence_cluster_key=self._cluster_key(episode),
            valid_from_episode_id=episode.episode_id,
            valid_from_index=episode.index,
            confidence=0.2,
        )
        return FailureRecord(
            failure_id=f"fail-{uuid.uuid4().hex[:16]}",
            episode_id=episode.episode_id,
            failure_type=FailureType.UNKNOWN,
            signature="critic_parse_failure",
            evidence="Structured critic output could not be parsed.",
            proposed_memory=candidate,
            confidence=0.2,
        )

    def _cluster_key(self, episode: Episode) -> str:
        """Derive provenance from fields visible to the learner only."""

        if not self.evolution.shadow_hierarchical_eprocess_enabled:
            return ""

        metadata = episode.sample.metadata
        source = str(metadata.get("feedback_source", "")).strip() or "unspecified"
        context = str(metadata.get("feedback_context", "")).strip()
        if not context:
            context = " ".join(episode.sample.prompt.casefold().split())
        raw_signal = metadata.get(
            "feedback_reference",
            "success" if episode.adaptation_score.success else "failure",
        )
        if isinstance(raw_signal, str):
            signal = " ".join(raw_signal.casefold().split())
        else:
            signal = json.dumps(
                raw_signal,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        return observable_candidate_cluster_key(
            domain=episode.sample.domain,
            feedback_source=source,
            feedback_context=context,
            feedback_signal=signal,
        )
