from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from evoshift.errors import ProviderError
from evoshift.providers.base import ModelT, validate_json_text
from evoshift.runtime.budget import BudgetLedger
from evoshift.schemas import GenerationRequest, GenerationResponse, LLMUsage

_EXPERIENCE_ID = re.compile(r"\[experience:([^@\]]+)@v\d+\]")
_ADDITION = re.compile(r"Compute\s+(-?\d+)\s*\+\s*(-?\d+)\.", re.I)
_AFFINE = re.compile(r"Compute\s+F\((-?\d+),\s*(-?\d+)\)", re.I)
_CONDITIONAL = re.compile(r"Given\s+x=(-?\d+)\s+and\s+y=(-?\d+)", re.I)
_SYMBOLIC = re.compile(r"for\s+x=(-?\d+),\s*y=(-?\d+)", re.I)
_REFUND = re.compile(
    r"customer_tier=(STANDARD|PREMIUM);\s*request_day=(\d+)",
    re.I,
)

_REFUND_V2_DIRECTIVE = (
    "For refund decisions, approve both standard and premium customers when "
    "request_day is at most 14; deny later requests."
)
_REFUND_V3_DIRECTIVE = (
    "For refund decisions, premium customers are approved through request_day 30, "
    "while the standard-customer limit remains 14 days."
)


@dataclass(frozen=True)
class _TaskRule:
    name: str
    trigger: str
    scope: str
    directive: str
    anti_pattern: str
    tags: tuple[str, ...]


_RULES: dict[str, _TaskRule] = {
    "addition": _TaskRule(
        name="addition",
        trigger="integer arithmetic addition Compute x + y",
        scope="arithmetic/addition",
        directive="Add the two signed integers and return only their integer sum.",
        anti_pattern="Do not change the requested operation or output format.",
        tags=("arithmetic", "addition"),
    ),
    "affine": _TaskRule(
        name="affine",
        trigger="affine F(x, y) 2*x - y Compute integer",
        scope="arithmetic/affine",
        directive="For affine tasks, evaluate exactly 2*x - y; do not substitute addition.",
        anti_pattern="Do not compute x+y when the prompt defines F(x,y)=2*x-y.",
        tags=("arithmetic", "affine", "operation-shift"),
    ),
    "conditional": _TaskRule(
        name="conditional",
        trigger="conditional arithmetic x even otherwise x+y x-y integer",
        scope="arithmetic/conditional",
        directive="Branch on x parity: use x+y when x is even, otherwise use x-y.",
        anti_pattern="Do not always add; evaluate whether x is even first.",
        tags=("arithmetic", "conditional", "control-flow"),
    ),
    "symbolic": _TaskRule(
        name="symbolic",
        trigger="symbolic output 3*x + y POS ZERO NEG sign",
        scope="arithmetic/symbolic_output",
        directive="Compute z=3*x+y, then map its sign to POS, ZERO, or NEG exactly.",
        anti_pattern="Do not return the numeric z when a symbolic sign label is required.",
        tags=("arithmetic", "symbolic-output", "format-shift"),
    ),
}


class HeuristicDemoClient:
    """Deterministic client for the SyntheticShift end-to-end demo only.

    It intentionally starts with common errors on shifted phases. A shifted
    task is solved correctly only when its matching experience card is in the
    task payload; the self-refine path is an explicit corrective baseline.
    This behavior is a pipeline demonstration, not an empirical research
    result or a model-quality simulation.
    """

    def __init__(
        self,
        *,
        model: str = "demo-heuristic",
        budget: BudgetLedger | None = None,
    ) -> None:
        self.model = model
        self._budget = budget
        self.calls: list[GenerationRequest] = []
        self._closed = False

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self._closed:
            raise ProviderError("demo LLM client is closed")
        self.calls.append(request.model_copy(deep=True))
        purpose = str(request.metadata.get("purpose", ""))
        if purpose == "experience_critic" or self._has_marker(request, "<EVOSHIFT_CRITIQUE>"):
            output = self._critic_output(self._payload(request, "<EVOSHIFT_CRITIQUE>"))
        elif purpose == "self_refine" or self._has_marker(request, "<EVOSHIFT_SELF_REFINE>"):
            output = self._solver_output(
                self._payload(request, "<EVOSHIFT_SELF_REFINE>"),
                force_correct=True,
            )
        elif purpose == "solve" or self._has_marker(request, "<EVOSHIFT_TASK>"):
            output = self._solver_output(
                self._payload(request, "<EVOSHIFT_TASK>"),
                force_correct=False,
            )
        else:
            raise ProviderError("demo client received an unsupported request purpose")

        text = json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        input_characters = sum(len(str(message.get("content", ""))) for message in request.messages)
        input_tokens = max(1, (input_characters + 3) // 4)
        output_tokens = max(1, (len(text) + 3) // 4)
        usage = LLMUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        if self._budget is not None:
            reservation = self._budget.reserve(estimated_tokens=usage.total_tokens)
            self._budget.reconcile(reservation, usage)
        return GenerationResponse(
            text=text,
            model=request.model or self.model,
            usage=usage,
            response_id=f"demo-{len(self.calls)}",
            raw={"demo": True, "purpose": purpose or "inferred"},
        )

    async def generate_json(
        self,
        request: GenerationRequest,
        schema: type[ModelT],
    ) -> ModelT:
        response = await self.generate(request)
        return validate_json_text(response.text, schema)

    def _solver_output(self, payload: Mapping[str, Any], *, force_correct: bool) -> dict[str, Any]:
        task = str(payload.get("task", ""))
        refund = _REFUND.search(task)
        if refund:
            return self._refund_solver_output(
                payload,
                tier=refund.group(1).lower(),
                request_day=int(refund.group(2)),
                force_correct=force_correct,
            )
        rule_name, x, y = self._parse_task(task)
        cards = str(payload.get("experience_cards", ""))
        learned = self._has_matching_card(rule_name, cards)
        correct = rule_name == "addition" or learned or force_correct
        answer = self._answer(rule_name, x, y, correct=correct)
        applied_ids = self._matching_memory_ids(rule_name, cards) if learned else []
        return {
            "answer": answer,
            "confidence": 0.96 if correct else 0.35,
            "rationale_summary": (
                "Applied the matching experience rule and checked the requested format."
                if learned
                else "Applied the demo client's current deterministic rule."
            ),
            "applied_memory_ids": applied_ids,
        }

    def _critic_output(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        task = str(payload.get("task", ""))
        if _REFUND.search(task):
            return self._refund_critic_output(payload)
        rule_name, _, _ = self._parse_task(task)
        rule = _RULES[rule_name]
        failure_type = "format_error" if rule_name == "symbolic" else "reasoning_error"
        return {
            "failure_type": failure_type,
            "signature": f"synthetic_{rule_name}_rule_not_applied",
            "evidence": "The deterministic first pass did not follow the task-specific rule.",
            "confidence": 0.98,
            "memory": {
                "kind": "procedural",
                "trigger": rule.trigger,
                "scope": rule.scope,
                "directive": rule.directive,
                "anti_pattern": rule.anti_pattern,
                "evidence": "Distilled from a failed SyntheticShift episode.",
                "tags": list(rule.tags),
            },
        }

    def _refund_solver_output(
        self,
        payload: Mapping[str, Any],
        *,
        tier: str,
        request_day: int,
        force_correct: bool,
    ) -> dict[str, Any]:
        cards = str(payload.get("experience_cards", ""))
        phase = str(payload.get("phase", "phase_0"))
        has_v2 = _REFUND_V2_DIRECTIVE.casefold() in cards.casefold()
        has_v3 = _REFUND_V3_DIRECTIVE.casefold() in cards.casefold()
        standard_window = 14 if has_v2 or has_v3 else 7
        premium_window = 30 if has_v3 else standard_window
        if force_correct:
            standard_window = 14 if phase != "phase_0" else 7
            premium_window = 30 if phase == "phase_2" else standard_window
        window = premium_window if tier == "premium" else standard_window
        answer = "APPROVE" if request_day <= window else "DENY"
        applied: list[str] = []
        if has_v2:
            applied.extend(self._memory_ids_with_directive(_REFUND_V2_DIRECTIVE, cards))
        if has_v3 and tier == "premium":
            applied.extend(self._memory_ids_with_directive(_REFUND_V3_DIRECTIVE, cards))
        return {
            "answer": answer,
            "confidence": 0.94 if has_v2 or has_v3 or phase == "phase_0" else 0.55,
            "rationale_summary": "Applied the currently available refund-policy memory.",
            "applied_memory_ids": list(dict.fromkeys(applied)),
        }

    @staticmethod
    def _refund_critic_output(payload: Mapping[str, Any]) -> dict[str, Any]:
        phase = str(payload.get("phase", "phase_0"))
        if phase == "phase_2":
            directive = _REFUND_V3_DIRECTIVE
            trigger = "A premium-customer refund request is made after day 14 but by day 30."
            anti_pattern = "Do not apply the 14-day standard limit to premium customers."
            tags = ["refund_policy", "premium_exception", "policy_v3"]
        else:
            directive = _REFUND_V2_DIRECTIVE
            trigger = "A refund request is made after day 7 but no later than day 14."
            anti_pattern = "Do not keep applying the superseded 7-day refund window."
            tags = ["refund_policy", "expanded_window", "policy_v2"]
        return {
            "failure_type": "reasoning_error",
            "signature": f"refund_policy_not_applied_{phase}",
            "evidence": "The observed feedback disagreed with the current refund decision.",
            "confidence": 0.96,
            "memory": {
                "kind": "procedural",
                "trigger": trigger,
                "scope": "customer_support/refund_policy",
                "directive": directive,
                "anti_pattern": anti_pattern,
                "evidence": "Distilled from an observed PolicyShift feedback event.",
                "tags": tags,
            },
        }

    @staticmethod
    def _parse_task(task: str) -> tuple[str, int, int]:
        match = _ADDITION.search(task)
        if match:
            return "addition", int(match.group(1)), int(match.group(2))
        match = _AFFINE.search(task)
        if match:
            return "affine", int(match.group(1)), int(match.group(2))
        match = _CONDITIONAL.search(task)
        if match:
            return "conditional", int(match.group(1)), int(match.group(2))
        match = _SYMBOLIC.search(task)
        if match and "3*x + y" in task:
            return "symbolic", int(match.group(1)), int(match.group(2))
        raise ProviderError("demo client supports only SyntheticShift arithmetic prompts")

    @staticmethod
    def _answer(rule_name: str, x: int, y: int, *, correct: bool) -> str:
        if rule_name == "addition":
            return str(x + y)
        if rule_name == "affine":
            return str(2 * x - y if correct else x + y)
        if rule_name == "conditional":
            return str((x + y if x % 2 == 0 else x - y) if correct else x + y)
        z = 3 * x + y
        if not correct:
            return str(z)
        return "POS" if z > 0 else ("ZERO" if z == 0 else "NEG")

    @staticmethod
    def _has_matching_card(rule_name: str, cards: str) -> bool:
        if rule_name == "addition":
            return True
        directive = _RULES[rule_name].directive
        return directive.casefold() in cards.casefold()

    @staticmethod
    def _matching_memory_ids(rule_name: str, cards: str) -> list[str]:
        directive = _RULES[rule_name].directive.casefold()
        return HeuristicDemoClient._memory_ids_with_directive(directive, cards)

    @staticmethod
    def _memory_ids_with_directive(directive: str, cards: str) -> list[str]:
        directive = directive.casefold()
        matches = list(_EXPERIENCE_ID.finditer(cards))
        selected: list[str] = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(cards)
            if directive in cards[match.start() : end].casefold():
                selected.append(match.group(1))
        return list(dict.fromkeys(selected))

    @staticmethod
    def _has_marker(request: GenerationRequest, marker: str) -> bool:
        return any(
            isinstance(message.get("content"), str) and marker in str(message["content"])
            for message in request.messages
        )

    @staticmethod
    def _payload(request: GenerationRequest, marker: str) -> Mapping[str, Any]:
        for message in reversed(request.messages):
            content = message.get("content")
            if not isinstance(content, str) or marker not in content:
                continue
            raw = content.split(marker, 1)[1].strip()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                raise ProviderError("demo client received malformed task JSON") from None
            if isinstance(payload, Mapping):
                return payload
        raise ProviderError(f"demo client request is missing marker {marker}")

    async def aclose(self) -> None:
        self._closed = True

    async def __aenter__(self) -> HeuristicDemoClient:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        await self.aclose()
