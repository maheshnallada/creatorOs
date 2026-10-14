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


def test_settings_default_to_mesh_provider(monkeypatch):
    _, _, settings_type = load_config_types()
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    settings = settings_type(_env_file=None)

    assert settings.llm_provider == "mesh"
    assert settings.groq_api_key is None
    assert settings.groq_model is None


def test_settings_can_select_groq_for_local_development(monkeypatch):
    _, _, settings_type = load_config_types()
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("GROQ_MODEL", "test-groq-model")

    settings = settings_type(_env_file=None)

    assert settings.llm_provider == "groq"
    assert settings.groq_api_key.get_secret_value() == "test-groq-key"
    assert settings.groq_model == "test-groq-model"


def test_production_settings_reject_groq_provider():
    _, _, settings_type = load_config_types()

    with pytest.raises(ValidationError, match="Groq.*local or test"):
        settings_type(
            app_env="production",
            llm_provider="groq",
            mesh_api_key="mesh-test-key",
            _env_file=None,
        )


def test_settings_default_to_local_sqlite_database():
    settings_type = load_config_types()[2]

    settings = settings_type(_env_file=None)

    assert settings.database_url == "sqlite:///./creatoros.db"


def test_production_settings_require_mesh_key():
    _, _, settings_type = load_config_types()
    with pytest.raises(ValidationError, match="MESH_API_KEY"):
        settings_type(
            app_env="production",
            llm_provider="mesh",
            mesh_api_key=None,
            _env_file=None,
        )


def test_platform_agent_config_accepts_allowlisted_tools():
    _, config_type, _ = load_config_types()
    config = config_type(
        platform="linkedin",
        agent_id="writer",
        model="openai/gpt-oss-120b",
        system_prompt_id="linkedin.writer.system",
        instruction_prompt_id="linkedin.writer.instructions",
        mcp_tool_ids=["get_brand_voice_profile"],
        llm_tool_ids=["format_platform_schema"],
    )

    assert config.max_tokens > 0
    assert config.timeout_seconds > 0
    assert config.max_attempts > 0


def test_platform_agent_config_has_bounded_output_and_editor_budgets():
    _, config_type, _ = load_config_types()

    config = config_type(
        platform="linkedin",
        agent_id="writer",
        model="openai/gpt-oss-120b",
        system_prompt_id="linkedin.writer.system",
        instruction_prompt_id="linkedin.writer.instructions",
    )

    assert config.max_output_corrections == 1
    assert config.max_editor_revisions == 2
    assert config.editor_quality_threshold == 7.0


@pytest.mark.parametrize(
    "override",
    [
        {"max_output_corrections": 4},
        {"max_editor_revisions": 6},
        {"editor_quality_threshold": 10.1},
    ],
)
def test_platform_agent_config_rejects_unbounded_loop_settings(override):
    _, config_type, _ = load_config_types()
    values = {
        "platform": "linkedin",
        "agent_id": "writer",
        "model": "openai/gpt-oss-120b",
        "system_prompt_id": "linkedin.writer.system",
        "instruction_prompt_id": "linkedin.writer.instructions",
    }
    values.update(override)

    with pytest.raises(ValidationError):
        config_type(**values)


def test_platform_agent_config_rejects_unknown_tool_ids():
    _, config_type, _ = load_config_types()
    with pytest.raises(ValidationError, match="unknown_tool"):
        config_type(
            platform="linkedin",
            agent_id="writer",
            model="openai/gpt-oss-120b",
            system_prompt_id="linkedin.writer.system",
            instruction_prompt_id="linkedin.writer.instructions",
            mcp_tool_ids=["unknown_tool"],
        )


def test_all_linkedin_agent_configs_reference_existing_prompts():
    from app.prompts.loader import PromptLoader

    project_root = Path(__file__).resolve().parents[1]
    config_type = load_config_types()[1]
    prompt_loader = PromptLoader(project_root / "agent_prompts")
    expected_agents = {
        "safety",
        "strategist",
        "writer",
        "editor",
        "brand_voice",
        "reflection",
    }
    config_files = sorted((project_root / "platform_configs").glob("linkedin.*.json"))
    actual_agents = set()
    instruction_variables = {
        "safety": {"context": "initial brief", "text": "sample brief"},
        "strategist": {"brief_json": "{}", "brand_memory_json": "{}"},
        "writer": {
            "brief_json": "{}",
            "strategy_json": "{}",
            "critique_json": "{}",
            "brand_memory_json": "{}",
        },
        "editor": {"brief_json": "{}", "strategy_json": "{}", "draft_json": "{}"},
        "brand_voice": {"samples_json": "[]"},
        "reflection": {"feedback_json": "{}", "prior_playbook_json": "{}"},
    }

    for config_file in config_files:
        config_data = json.loads(config_file.read_text(encoding="utf-8"))
        config = config_type.model_validate(config_data)
        actual_agents.add(config.agent_id)
        prompt_loader.render(config.system_prompt_id, {})
        prompt_loader.render(
            config.instruction_prompt_id,
            instruction_variables[config.agent_id],
        )

    assert actual_agents == expected_agents


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
        "model": "openai/gpt-oss-120b",
        "system_prompt_id": "linkedin.writer.system",
        "instruction_prompt_id": "linkedin.writer.instructions",
    }
    values.update(overrides)

    with pytest.raises(ValidationError):
        config_type(**values)
