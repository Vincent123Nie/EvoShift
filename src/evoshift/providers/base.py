from __future__ import annotations

import json
import re
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

from evoshift.errors import ProviderError
from evoshift.schemas import GenerationRequest, GenerationResponse

ModelT = TypeVar("ModelT", bound=BaseModel)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal asynchronous interface used by agents and evaluators."""

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate one response for a normalized request."""

    async def generate_json(
        self,
        request: GenerationRequest,
        schema: type[ModelT],
    ) -> ModelT:
        """Generate and validate a JSON object against ``schema``."""

    async def aclose(self) -> None:
        """Release resources owned by the client."""


def prompt_text(request: GenerationRequest) -> str:
    """Return a readable prompt representation for deterministic fake handlers."""

    chunks: list[str] = []
    for message in request.messages:
        content = message.get("content", "")
        if isinstance(content, str):
            chunks.append(content)
        else:
            chunks.append(json.dumps(content, ensure_ascii=False, sort_keys=True, default=str))
    return "\n".join(chunks)


def validate_json_text(text: str, schema: type[ModelT]) -> ModelT:
    """Parse plain or fenced JSON without leaking model output in errors."""

    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL | re.I)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        value: Any = json.loads(candidate)
    except json.JSONDecodeError:
        # Gateways occasionally prepend a short sentence despite an explicit
        # JSON-only prompt. A conservative object extraction keeps the helper
        # useful while deliberately refusing ambiguous multi-object output.
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise ProviderError("provider returned invalid structured JSON") from None
        try:
            value = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            raise ProviderError("provider returned invalid structured JSON") from None

    try:
        return schema.model_validate(value)
    except ValidationError:
        raise ProviderError("provider JSON did not match the requested schema") from None
