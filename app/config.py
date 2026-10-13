from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MESH_BASE_URL = "https://api.meshapi.ai/v1"

ALLOWED_MCP_TOOL_IDS = frozenset(
    {
        "get_brand_voice_profile",
        "query_top_performing_patterns",
        "score_content_rubric",
        "log_decision",
        "fetch_post_metrics",
        "update_brand_playbook",
        "publish_post",
    }
)
ALLOWED_LLM_TOOL_IDS = frozenset(
    {
        "format_platform_schema",
        "generate_image_mesh",
    }
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["local", "test", "production"] = "local"
    mesh_api_key: SecretStr | None = None
    mesh_base_url: Literal["https://api.meshapi.ai/v1"] = MESH_BASE_URL
    default_model: str = "openai/gpt-4o-mini"
    database_url: str = "sqlite:///./creatoros.db"
    request_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    default_max_attempts: int = Field(default=3, ge=1, le=10)

    @model_validator(mode="after")
    def require_mesh_key_in_production(self):
        if self.app_env == "production" and self.mesh_api_key is None:
            raise ValueError("MESH_API_KEY is required in production")
        return self


class PlatformAgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    max_attempts: int = Field(default=3, ge=1, le=10)
    max_output_corrections: int = Field(default=1, ge=0, le=3)
    max_editor_revisions: int = Field(default=2, ge=0, le=5)
    editor_quality_threshold: float = Field(default=7.0, ge=0, le=10)
    mcp_tool_ids: list[str] = Field(default_factory=list)
    llm_tool_ids: list[str] = Field(default_factory=list)
    system_prompt_id: str = Field(min_length=1)
    instruction_prompt_id: str = Field(min_length=1)

    @field_validator("mcp_tool_ids")
    @classmethod
    def validate_mcp_tool_ids(cls, tool_ids: list[str]) -> list[str]:
        unknown = set(tool_ids) - ALLOWED_MCP_TOOL_IDS
        if unknown:
            raise ValueError(f"unknown MCP tool IDs: {', '.join(sorted(unknown))}")
        return tool_ids

    @field_validator("llm_tool_ids")
    @classmethod
    def validate_llm_tool_ids(cls, tool_ids: list[str]) -> list[str]:
        unknown = set(tool_ids) - ALLOWED_LLM_TOOL_IDS
        if unknown:
            raise ValueError(f"unknown LLM tool IDs: {', '.join(sorted(unknown))}")
        return tool_ids
