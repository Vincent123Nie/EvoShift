from __future__ import annotations

import inspect
import json
from collections import deque
from collections.abc import Awaitable, Iterable, Mapping
from typing import Any, Callable, Union

from evoshift.errors import ProviderError
from evoshift.providers.base import ModelT, validate_json_text
from evoshift.schemas import GenerationRequest, GenerationResponse, LLMUsage

FakeValue = Union[str, GenerationResponse, Mapping[str, Any], BaseException]
FakeResult = Union[FakeValue, Awaitable[FakeValue]]
PromptHandler = Callable[[GenerationRequest], FakeResult]
ScriptItem = Union[FakeValue, PromptHandler]


class FakeLLMClient:
    """Deterministic offline client driven by a handler or response sequence."""

    def __init__(
        self,
        responses: Iterable[ScriptItem] | None = None,
        *,
        handler: PromptHandler | None = None,
        model: str = "fake",
        default_response: str = "FAKE_RESPONSE",
    ) -> None:
        if responses is not None and handler is not None:
            raise ValueError("provide either responses or handler, not both")
        self.model = model
        self.handler = handler
        self._scripted = responses is not None
        self._responses: deque[ScriptItem] = deque(responses or [])
        self.default_response = default_response
        self.calls: list[GenerationRequest] = []
        self._closed = False

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self._closed:
            raise ProviderError("fake LLM client is closed")
        self.calls.append(request.model_copy(deep=True))
        call_number = len(self.calls)

        if self.handler is not None:
            value: Any = self.handler(request)
        elif self._scripted:
            if not self._responses:
                raise ProviderError("scripted fake responses exhausted")
            item = self._responses.popleft()
            value = item(request) if callable(item) else item
        else:
            value = self.default_response

        if inspect.isawaitable(value):
            value = await value
        if isinstance(value, BaseException):
            raise value
        return self._coerce_response(value, request=request, call_number=call_number)

    async def generate_json(
        self,
        request: GenerationRequest,
        schema: type[ModelT],
    ) -> ModelT:
        response = await self.generate(request)
        return validate_json_text(response.text, schema)

    def _coerce_response(
        self,
        value: Any,
        *,
        request: GenerationRequest,
        call_number: int,
    ) -> GenerationResponse:
        if isinstance(value, GenerationResponse):
            return value.model_copy(deep=True)
        model = request.model or self.model
        if isinstance(value, str):
            text = value
        elif isinstance(value, Mapping):
            if "text" in value:
                try:
                    return GenerationResponse.model_validate(value)
                except Exception:
                    raise ProviderError("invalid scripted GenerationResponse mapping") from None
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            raise ProviderError(f"unsupported fake response type: {type(value).__name__}")
        return GenerationResponse(
            text=text,
            model=model,
            usage=LLMUsage(),
            response_id=f"fake-{call_number}",
            raw={"fake": True},
        )

    @property
    def remaining(self) -> int | None:
        return len(self._responses) if self._scripted else None

    async def aclose(self) -> None:
        self._closed = True

    async def __aenter__(self) -> FakeLLMClient:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        await self.aclose()


class ScriptedLLMClient(FakeLLMClient):
    def __init__(self, responses: Iterable[ScriptItem], *, model: str = "fake") -> None:
        super().__init__(responses=responses, model=model)


FakeClient = FakeLLMClient
