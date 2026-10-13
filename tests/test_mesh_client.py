import json

import httpx
import pytest
from pydantic import SecretStr


def load_mesh_client():
    try:
        from app.llm.mesh_client import (
            LLMRequest,
            MeshClient,
            MeshRequestError,
            MissingMeshApiKey,
        )
    except ModuleNotFoundError as error:
        pytest.fail(f"Mesh client implementation is missing: {error.name}")
    return LLMRequest, MeshClient, MeshRequestError, MissingMeshApiKey


def mesh_settings(api_key: str | None = "test-mesh-key"):
    from app.config import Settings

    return Settings(mesh_api_key=SecretStr(api_key) if api_key else None, _env_file=None)


def test_mesh_client_posts_to_mesh_and_normalizes_result():
    request_type, client_type, _, _ = load_mesh_client()
    observed = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["url"] = str(request.url)
        observed["authorization"] = request.headers.get("Authorization")
        observed["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "openai/gpt-4o-mini",
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
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Create a post."}],
            max_tokens=100,
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
    assert observed["body"]["model"] == "openai/gpt-4o-mini"
    assert observed["body"]["tools"][0]["function"]["name"] == "format_platform_schema"
    assert observed["body"]["response_format"] == {"type": "json_object"}
    assert result.content == '{"ok":true}'
    assert result.usage.total_tokens == 14
    assert result.request_id == "chatcmpl-test"


def test_mesh_client_redacts_provider_error_details():
    _, client_type, error_type, _ = load_mesh_client()
    secret = "test-mesh-key"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": f"invalid key {secret}"})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    mesh_client = client_type(mesh_settings(secret), http_client=http_client)

    with pytest.raises(error_type) as error:
        mesh_client.complete(
            {
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
    mesh_client.close()

    assert secret not in str(error.value)
    assert "401" in str(error.value)


def test_mesh_client_refuses_calls_without_configured_key():
    request_type, client_type, _, missing_key_type = load_mesh_client()
    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: pytest.fail("network call"))
    )
    mesh_client = client_type(mesh_settings(None), http_client=http_client)

    with pytest.raises(missing_key_type, match="MESH_API_KEY"):
        mesh_client.complete(
            request_type(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "hello"}],
            )
        )
    mesh_client.close()
