import json
from pathlib import Path

import pytest
from pydantic import ValidationError


def load_config_types():
    try:
        from app.config import MESH_BASE_URL, PlatformAgentConfig, Settings
    except ModuleNotFoundError as error:
        pytest.fail(f"configuration implementation is missing: {error.name}")
    return MESH_BASE_URL, PlatformAgentConfig, Settings


def test_settings_read_mesh_key_and_use_mesh_endpoint(monkeypatch):
    mesh_base_url, _, settings_type = load_config_types()
    monkeypatch.setenv("MESH_API_KEY", "test-key")

    settings = settings_type(_env_file=None)

    assert settings.mesh_api_key.get_secret_value() == "test-key"
    assert settings.mesh_base_url == mesh_base_url
    assert str(settings.mesh_base_url) == "https://api.meshapi.ai/v1"


def test_local_settings_allow_missing_mesh_key(monkeypatch):
    _, _, settings_type = load_config_types()
    monkeypatch.delenv("MESH_API_KEY", raising=False)

    settings = settings_type(_env_file=None)

    assert settings.app_env == "local"
    assert settings.mesh_api_key is None


def test_production_settings_require_mesh_key():
    _, _, settings_type = load_config_types()
    with pytest.raises(ValidationError, match="MESH_API_KEY"):
        settings_type(app_env="production", mesh_api_key=None, _env_file=None)


def test_platform_agent_config_accepts_allowlisted_tools():
    _, config_type, _ = load_config_types()
    config = config_type(
        platform="linkedin",
        agent_id="writer",
        model="openai/gpt-4o-mini",
        system_prompt_id="linkedin.writer.system",
        instruction_prompt_id="linkedin.writer.instructions",
        mcp_tool_ids=["get_brand_voice_profile"],
        llm_tool_ids=["format_platform_schema"],
    )

    assert config.max_tokens > 0
    assert config.timeout_seconds > 0
    assert config.max_attempts > 0


def test_platform_agent_config_rejects_unknown_tool_ids():
    _, config_type, _ = load_config_types()
    with pytest.raises(ValidationError, match="unknown_tool"):
        config_type(
            platform="linkedin",
            agent_id="writer",
            model="openai/gpt-4o-mini",
            system_prompt_id="linkedin.writer.system",
            instruction_prompt_id="linkedin.writer.instructions",
            mcp_tool_ids=["unknown_tool"],
        )


def test_sample_platform_config_references_existing_prompts():
    from app.prompts.loader import PromptLoader

    project_root = Path(__file__).resolve().parents[1]
    config_data = json.loads(
        (project_root / "platform_configs" / "linkedin.writer.json").read_text(encoding="utf-8")
    )
    config_type = load_config_types()[1]
    config = config_type.model_validate(config_data)
    prompt_loader = PromptLoader(project_root / "agent_prompts")

    prompt_loader.render(config.system_prompt_id, {})
    prompt_loader.render(
        config.instruction_prompt_id,
        {"brief": "brief", "strategy": "strategy", "brand_voice_profile": "voice"},
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_tokens": 0},
        {"timeout_seconds": 0},
        {"max_attempts": 0},
    ],
)
def test_platform_agent_config_rejects_nonpositive_limits(overrides):
    _, config_type, _ = load_config_types()
    values = {
        "platform": "linkedin",
        "agent_id": "writer",
        "model": "openai/gpt-4o-mini",
        "system_prompt_id": "linkedin.writer.system",
        "instruction_prompt_id": "linkedin.writer.instructions",
    }
    values.update(overrides)

    with pytest.raises(ValidationError):
        config_type(**values)
