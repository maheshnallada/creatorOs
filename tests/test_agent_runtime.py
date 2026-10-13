from pathlib import Path

import pytest
from pydantic import BaseModel


class CaptionOutput(BaseModel):
    caption: str


def load_runtime_types():
    try:
        from app.agents.runtime import AgentRuntime, StructuredOutputError
    except ModuleNotFoundError as error:
        pytest.fail(f"structured agent runtime is missing: {error.name}")
    return AgentRuntime, StructuredOutputError


class FakeMeshClient:
    def __init__(self, contents: list[str | None]):
        self.contents = list(contents)
        self.requests = []
        self.retry_limits = []

    def complete_structured(self, request, response_model, max_retries):
        self.requests.append(request)
        self.retry_limits.append(max_retries)
        for _ in range(max_retries + 1):
            content = self.contents.pop(0)
            try:
                return response_model.model_validate_json(content or "")
            except ValueError:
                continue
        from app.agents.runtime import StructuredOutputError

        raise StructuredOutputError("structured output correction limit reached")


def build_runtime(
    tmp_path: Path,
    contents: list[str | None],
    max_output_corrections=1,
    timeout_seconds=30,
    max_tokens=2048,
):
    from app.config import PlatformAgentConfig
    from app.prompts.loader import PromptLoader

    system_path = tmp_path / "linkedin" / "writer" / "system.md"
    instruction_path = tmp_path / "linkedin" / "writer" / "instructions.md"
    system_path.parent.mkdir(parents=True)
    system_path.write_text("Trusted writer role.", encoding="utf-8")
    instruction_path.write_text("Write about {topic}.", encoding="utf-8")
    config = PlatformAgentConfig(
        platform="linkedin",
        agent_id="writer",
        model="openai/gpt-4o-mini",
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        max_output_corrections=max_output_corrections,
        system_prompt_id="linkedin.writer.system",
        instruction_prompt_id="linkedin.writer.instructions",
    )
    mesh_client = FakeMeshClient(contents)
    runtime_type, _ = load_runtime_types()
    return runtime_type(mesh_client, PromptLoader(tmp_path)), config, mesh_client


def test_runtime_parses_valid_json_through_mesh_response_format(tmp_path: Path):
    runtime, config, mesh_client = build_runtime(tmp_path, ['{"caption":"A useful post"}'])

    output = runtime.generate_json(config, {}, {"topic": "AI safety"}, CaptionOutput)

    assert output == CaptionOutput(caption="A useful post")
    assert mesh_client.retry_limits == [1]
    assert mesh_client.requests[0].messages[0].role == "system"
    assert mesh_client.requests[0].messages[1].role == "user"


def test_runtime_applies_agent_timeout_and_token_budget(tmp_path: Path):
    runtime, config, mesh_client = build_runtime(
        tmp_path,
        ['{"caption":"A useful post"}'],
        timeout_seconds=11,
        max_tokens=333,
    )

    runtime.generate_json(config, {}, {"topic": "AI safety"}, CaptionOutput)

    assert mesh_client.requests[0].timeout_seconds == 11
    assert mesh_client.requests[0].max_tokens == 333


def test_runtime_corrects_malformed_json_with_bounded_feedback(tmp_path: Path):
    runtime, config, mesh_client = build_runtime(tmp_path, ["not json", '{"caption":"corrected"}'])

    output = runtime.generate_json(config, {}, {"topic": "AI safety"}, CaptionOutput)

    assert output.caption == "corrected"
    assert len(mesh_client.requests) == 1
    assert mesh_client.retry_limits == [1]


def test_runtime_stops_after_configured_correction_budget(tmp_path: Path):
    runtime, config, mesh_client = build_runtime(
        tmp_path, ["not json", "still not json"], max_output_corrections=1
    )
    runtime_type, error_type = load_runtime_types()
    assert isinstance(runtime, runtime_type)

    with pytest.raises(error_type, match="correction limit"):
        runtime.generate_json(config, {}, {"topic": "AI safety"}, CaptionOutput)

    assert len(mesh_client.requests) == 1
    assert mesh_client.retry_limits == [1]


def test_runtime_corrects_schema_violation(tmp_path: Path):
    runtime, config, mesh_client = build_runtime(
        tmp_path, ['{"unexpected":"field"}', '{"caption":"valid"}']
    )

    output = runtime.generate_json(config, {}, {"topic": "AI safety"}, CaptionOutput)

    assert output.caption == "valid"
    assert len(mesh_client.requests) == 1
    assert mesh_client.retry_limits == [1]


def test_runtime_corrects_empty_model_content(tmp_path: Path):
    runtime, config, mesh_client = build_runtime(tmp_path, [None, '{"caption":"recovered"}'])

    output = runtime.generate_json(config, {}, {"topic": "AI safety"}, CaptionOutput)

    assert output.caption == "recovered"
    assert len(mesh_client.requests) == 1
    assert mesh_client.retry_limits == [1]
