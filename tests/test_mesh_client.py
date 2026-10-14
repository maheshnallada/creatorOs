import json

import httpx
import pytest
from pydantic import BaseModel, SecretStr


def load_mesh_client():
    try:
        from app.llm.mesh_client import (
            LLMRequest,
            MeshClient,
            MeshRequestError,
            MeshStructuredOutputError,
            MissingMeshApiKey,
        )
    except ModuleNotFoundError as error:
        pytest.fail(f"Mesh client implementation is missing: {error.name}")
    return (
        LLMRequest,
        MeshClient,
        MeshRequestError,
        MeshStructuredOutputError,
        MissingMeshApiKey,
    )


class StructuredCaption(BaseModel):
    caption: str


def mesh_settings(api_key: str | None = "test-mesh-key"):
    from app.config import Settings

    return Settings(
        llm_provider="mesh",
        mesh_api_key=SecretStr(api_key) if api_key else None,
        _env_file=None,
    )


def groq_settings(api_key: str | None = "test-groq-key", model: str | None = "test-groq-model"):
    from app.config import Settings

    return Settings(
        llm_provider="groq",
        groq_api_key=SecretStr(api_key) if api_key else None,
        groq_model=model,
        _env_file=None,
    )


def test_llm_client_posts_to_groq_and_normalizes_result():
    from app.llm.mesh_client import LLMClient, LLMRequest

    observed = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["url"] = str(request.url)
        observed["authorization"] = request.headers.get("Authorization")
        observed["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-groq",
                "object": "chat.completion",
                "created": 1,
                "model": "test-groq-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"ok":true}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    llm_client = LLMClient(groq_settings(), http_client=http_client)
    result = llm_client.complete(
        LLMRequest(
            model="test-groq-model",
            messages=[{"role": "user", "content": "Write a post."}],
        )
    )
    llm_client.close()

    assert observed["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert observed["authorization"] == "Bearer test-groq-key"
    assert observed["body"]["model"] == "test-groq-model"
    assert result.content == '{"ok":true}'
    assert result.usage.total_tokens == 10
    assert result.request_id == "chatcmpl-groq"


def test_llm_client_resolves_models_for_selected_provider():
    from app.llm.mesh_client import LLMClient

    mesh_client = LLMClient(mesh_settings())
    groq_client = LLMClient(groq_settings())
    try:
        assert mesh_client.resolve_model("mesh-agent-model") == "mesh-agent-model"
        assert groq_client.resolve_model("mesh-agent-model") == "test-groq-model"
    finally:
        mesh_client.close()
        groq_client.close()


def test_llm_client_requires_mesh_credentials_when_resolving_model():
    from app.llm.mesh_client import LLMClient, MissingProviderApiKey

    settings = mesh_settings(api_key=None)
    assert settings.llm_provider == "mesh"
    llm_client = LLMClient(settings)
    with pytest.raises(MissingProviderApiKey, match="MESH_API_KEY"):
        llm_client.resolve_model("mesh-agent-model")
    llm_client.close()


def test_llm_client_rejects_missing_groq_credentials_before_network():
    from app.llm.mesh_client import LLMClient, MissingProviderApiKey

    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: pytest.fail(f"unexpected request: {request}"))
    )
    llm_client = LLMClient(groq_settings(api_key=None), http_client=http_client)
    with pytest.raises(MissingProviderApiKey, match="GROQ_API_KEY"):
        llm_client.complete(
            {
                "model": "test-groq-model",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
    llm_client.close()


def test_llm_client_rejects_missing_groq_model_before_network():
    from app.llm.mesh_client import LLMClient, LLMConfigurationError

    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: pytest.fail(f"unexpected request: {request}"))
    )
    llm_client = LLMClient(groq_settings(model=None), http_client=http_client)
    with pytest.raises(LLMConfigurationError, match="GROQ_MODEL"):
        llm_client.complete(
            {
                "model": "mesh-specific-model",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
    llm_client.close()


def test_llm_client_structured_groq_retries_invalid_json_within_limit():
    from app.llm.mesh_client import LLMClient, LLMRequest

    responses = ["not JSON", '{"caption":"Groq corrected"}']
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        content = responses.pop(0)
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-groq-{len(requests)}",
                "object": "chat.completion",
                "created": 1,
                "model": "test-groq-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    llm_client = LLMClient(groq_settings(), http_client=http_client)
    result = llm_client.complete_structured(
        LLMRequest(
            model=llm_client.resolve_model("mesh-specific-model"),
            messages=[{"role": "user", "content": "Write a caption."}],
        ),
        response_model=StructuredCaption,
        max_retries=1,
    )
    llm_client.close()

    assert result.caption == "Groq corrected"
    assert len(requests) == 2
    assert all(request.url.host == "api.groq.com" for request in requests)
    assert all(request.headers["Authorization"] == "Bearer test-groq-key" for request in requests)


def test_llm_client_does_not_fallback_and_sanitizes_groq_errors(caplog):
    from app.llm.mesh_client import LLMClient, LLMProviderRequestError

    secret = "test-groq-key"
    urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(401, json={"error": f"invalid key {secret}"})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    llm_client = LLMClient(groq_settings(secret), http_client=http_client)
    with pytest.raises(LLMProviderRequestError) as error:
        llm_client.complete(
            {
                "model": "test-groq-model",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
    llm_client.close()

    assert "Groq" in str(error.value)
    assert "401" in str(error.value)
    assert error.value.provider == "groq"
    assert secret not in str(error.value)
    assert secret not in caplog.text
    assert urls == ["https://api.groq.com/openai/v1/chat/completions"]


def test_mesh_client_posts_to_mesh_and_normalizes_result():
    request_type, client_type, _, _, _ = load_mesh_client()
    observed = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["url"] = str(request.url)
        observed["authorization"] = request.headers.get("Authorization")
        observed["body"] = json.loads(request.content)
        observed["timeout"] = request.extensions.get("timeout")
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "openai/gpt-oss-120b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"ok":true}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    mesh_client = client_type(mesh_settings(), http_client=http_client)
    result = mesh_client.complete(
        request_type(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "Create a post."}],
            max_tokens=100,
            timeout_seconds=3.0,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "format_platform_schema",
                        "description": "Validate post structure",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            response_format={"type": "json_object"},
        )
    )
    mesh_client.close()

    assert observed["url"] == "https://api.meshapi.ai/v1/chat/completions"
    assert observed["authorization"] == "Bearer test-mesh-key"
    assert observed["body"]["model"] == "openai/gpt-oss-120b"
    assert observed["body"]["tools"][0]["function"]["name"] == "format_platform_schema"
    assert observed["body"]["response_format"] == {"type": "json_object"}
    assert observed["timeout"]["read"] == 3.0
    assert result.content == '{"ok":true}'
    assert result.usage.total_tokens == 14
    assert result.request_id == "chatcmpl-test"


def test_mesh_client_redacts_provider_error_details():
    _, client_type, error_type, _, _ = load_mesh_client()
    secret = "test-mesh-key"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": f"invalid key {secret}"})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    mesh_client = client_type(mesh_settings(secret), http_client=http_client)

    with pytest.raises(error_type) as error:
        mesh_client.complete(
            {
                "model": "openai/gpt-oss-120b",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
    mesh_client.close()

    assert secret not in str(error.value)
    assert "401" in str(error.value)


def test_mesh_client_refuses_calls_without_configured_key():
    request_type, client_type, _, _, missing_key_type = load_mesh_client()
    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: pytest.fail("network call"))
    )
    mesh_client = client_type(mesh_settings(None), http_client=http_client)

    with pytest.raises(missing_key_type, match="MESH_API_KEY"):
        mesh_client.complete(
            request_type(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": "hello"}],
            )
        )
    mesh_client.close()


def test_mesh_client_uses_instructor_and_mesh_for_structured_response():
    request_type, client_type, _, _, _ = load_mesh_client()
    observed = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["url"] = str(request.url)
        observed["authorization"] = request.headers.get("Authorization")
        observed["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-structured",
                "object": "chat.completion",
                "created": 1,
                "model": "openai/gpt-oss-120b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"caption":"Mesh output"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    mesh_client = client_type(mesh_settings(), http_client=http_client)
    result = mesh_client.complete_structured(
        request_type(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "Write a caption."}],
            max_tokens=80,
            timeout_seconds=4.0,
        ),
        response_model=StructuredCaption,
        max_retries=1,
    )
    mesh_client.close()

    assert isinstance(result, StructuredCaption)
    assert result.caption == "Mesh output"
    assert observed["url"] == "https://api.meshapi.ai/v1/chat/completions"
    assert observed["authorization"] == "Bearer test-mesh-key"
    assert observed["body"]["response_format"] == {"type": "json_object"}


def test_mesh_client_instructor_retries_validation_only_to_limit():
    request_type, client_type, _, _, _ = load_mesh_client()
    responses = ["not JSON", '{"caption":"corrected"}']
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = responses.pop(0) if responses else "still invalid"
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-{calls}",
                "object": "chat.completion",
                "created": 1,
                "model": "openai/gpt-oss-120b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    mesh_client = client_type(mesh_settings(), http_client=http_client)
    result = mesh_client.complete_structured(
        request_type(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "Write a caption."}],
        ),
        response_model=StructuredCaption,
        max_retries=1,
    )
    mesh_client.close()

    assert result.caption == "corrected"
    assert calls == 2


def test_mesh_client_instructor_stops_at_validation_retry_limit():
    request_type, client_type, _, structured_error_type, _ = load_mesh_client()
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-{calls}",
                "object": "chat.completion",
                "created": 1,
                "model": "openai/gpt-oss-120b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "not JSON"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    mesh_client = client_type(mesh_settings(), http_client=http_client)
    with pytest.raises(structured_error_type, match="correction limit"):
        mesh_client.complete_structured(
            request_type(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": "Write a caption."}],
            ),
            response_model=StructuredCaption,
            max_retries=1,
        )
    mesh_client.close()

    assert calls == 2


def test_mesh_client_instructor_zero_retry_budget_makes_one_call():
    request_type, client_type, _, structured_error_type, _ = load_mesh_client()
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-{calls}",
                "object": "chat.completion",
                "created": 1,
                "model": "openai/gpt-oss-120b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "not JSON"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    mesh_client = client_type(mesh_settings(), http_client=http_client)
    with pytest.raises(structured_error_type, match="correction limit"):
        mesh_client.complete_structured(
            request_type(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": "Write a caption."}],
            ),
            response_model=StructuredCaption,
            max_retries=0,
        )
    mesh_client.close()

    assert calls == 1


def test_mesh_client_sanitizes_structured_provider_errors(caplog):
    _, client_type, request_error_type, _, _ = load_mesh_client()
    secret = "test-mesh-key"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": f"invalid key {secret}"})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    mesh_client = client_type(mesh_settings(secret), http_client=http_client)
    with pytest.raises(request_error_type) as error:
        mesh_client.complete_structured(
            {
                "model": "openai/gpt-oss-120b",
                "messages": [{"role": "user", "content": "Write a caption."}],
            },
            response_model=StructuredCaption,
            max_retries=1,
        )
    mesh_client.close()

    assert "401" in str(error.value)
    assert secret not in str(error.value)
    assert secret not in caplog.text
