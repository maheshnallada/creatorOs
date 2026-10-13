import logging
from typing import Any, Literal, TypeVar

import httpx
import instructor
from instructor.core.exceptions import InstructorRetryException
from openai import APIConnectionError, APIError, APIStatusError, OpenAI
from pydantic import BaseModel, ConfigDict, Field

from app.config import MESH_BASE_URL, Settings

OutputModel = TypeVar("OutputModel", bound=BaseModel)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str


class LLMRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0, le=300)
    temperature: float | None = Field(default=None, ge=0, le=2)
    tools: list[dict[str, Any]] | None = None
    response_format: dict[str, Any] | None = None


class LLMUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResult(BaseModel):
    content: str | None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: LLMUsage = Field(default_factory=LLMUsage)
    request_id: str | None = None
    model: str
    finish_reason: str | None = None


class MeshRequestError(RuntimeError):
    """A sanitized failure returned by the Mesh gateway."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class MissingMeshApiKey(RuntimeError):
    """Raised when a model request is attempted without Mesh credentials."""


class MeshStructuredOutputError(RuntimeError):
    """Raised when Instructor exhausts the configured Pydantic validation retries."""


class _MeshSecretFilter(logging.Filter):
    def __init__(self, secret: str | None):
        super().__init__()
        self._secret = secret

    def filter(self, record: logging.LogRecord) -> bool:
        if self._secret:
            message = record.getMessage()
            if self._secret in message:
                record.msg = message.replace(self._secret, "[REDACTED]")
                record.args = ()
        return True


class MeshClient:
    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.Client | None = None,
    ):
        self._api_key = (
            settings.mesh_api_key.get_secret_value() if settings.mesh_api_key is not None else None
        )
        self._instructor_logger = logging.getLogger("instructor.v2.retry")
        self._secret_filter = _MeshSecretFilter(self._api_key)
        self._instructor_logger.addFilter(self._secret_filter)
        self._client = OpenAI(
            base_url=MESH_BASE_URL,
            api_key=self._api_key or "missing-local-mesh-key",
            timeout=settings.request_timeout_seconds,
            http_client=http_client,
        )

    def complete(self, request: LLMRequest | dict[str, Any]) -> LLMResult:
        if self._api_key is None:
            raise MissingMeshApiKey("MESH_API_KEY must be configured before making model calls")

        normalized_request = LLMRequest.model_validate(request)
        try:
            client = (
                self._client.with_options(timeout=normalized_request.timeout_seconds)
                if normalized_request.timeout_seconds is not None
                else self._client
            )
            response = client.chat.completions.create(
                model=normalized_request.model,
                messages=[message.model_dump() for message in normalized_request.messages],
                max_tokens=normalized_request.max_tokens,
                temperature=normalized_request.temperature,
                tools=normalized_request.tools,
                response_format=normalized_request.response_format,
            )
        except APIStatusError as error:
            raise MeshRequestError(
                f"Mesh API request failed with HTTP {error.status_code}",
                retryable=error.status_code in {408, 409, 429}
                or (error.status_code is not None and error.status_code >= 500),
            ) from error
        except APIConnectionError as error:
            raise MeshRequestError("Mesh API connection failed", retryable=True) from error
        except APIError as error:
            raise MeshRequestError("Mesh API request failed") from error

        choice = response.choices[0]
        usage = response.usage
        return LLMResult(
            content=choice.message.content,
            tool_calls=[tool_call.model_dump() for tool_call in choice.message.tool_calls or []],
            usage=LLMUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
            request_id=response.id,
            model=response.model,
            finish_reason=choice.finish_reason,
        )

    def complete_structured(
        self,
        request: LLMRequest | dict[str, Any],
        *,
        response_model: type[OutputModel],
        max_retries: int,
    ) -> OutputModel:
        if self._api_key is None:
            raise MissingMeshApiKey("MESH_API_KEY must be configured before making model calls")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")

        normalized_request = LLMRequest.model_validate(request)
        client = (
            self._client.with_options(timeout=normalized_request.timeout_seconds)
            if normalized_request.timeout_seconds is not None
            else self._client
        )
        instructor_client = instructor.from_openai(client, mode=instructor.Mode.JSON)
        try:
            return instructor_client.chat.completions.create(
                model=normalized_request.model,
                messages=[message.model_dump() for message in normalized_request.messages],
                max_tokens=normalized_request.max_tokens,
                temperature=normalized_request.temperature,
                response_model=response_model,
                max_retries=max_retries,
            )
        except APIStatusError as error:
            raise MeshRequestError(
                f"Mesh API request failed with HTTP {error.status_code}",
                retryable=error.status_code in {408, 409, 429}
                or (error.status_code is not None and error.status_code >= 500),
            ) from None
        except APIConnectionError:
            raise MeshRequestError("Mesh API connection failed", retryable=True) from None
        except APIError:
            raise MeshRequestError("Mesh API request failed") from None
        except InstructorRetryException as error:
            provider_error = error.__cause__
            if isinstance(provider_error, APIStatusError):
                raise MeshRequestError(
                    f"Mesh API request failed with HTTP {provider_error.status_code}",
                    retryable=provider_error.status_code in {408, 409, 429}
                    or (
                        provider_error.status_code is not None and provider_error.status_code >= 500
                    ),
                ) from None
            if isinstance(provider_error, APIConnectionError):
                raise MeshRequestError("Mesh API connection failed", retryable=True) from None
            if isinstance(provider_error, APIError):
                raise MeshRequestError("Mesh API request failed") from None
            raise MeshStructuredOutputError(
                "structured output remained invalid after correction limit"
            ) from None

    def close(self) -> None:
        self._client.close()
        self._instructor_logger.removeFilter(self._secret_filter)
