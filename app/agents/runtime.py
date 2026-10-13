from typing import TypeVar

from pydantic import BaseModel

from app.config import PlatformAgentConfig
from app.llm.mesh_client import LLMRequest, MeshClient, MeshStructuredOutputError
from app.llm.retry import RetryPolicy
from app.prompts.loader import PromptLoader

OutputModel = TypeVar("OutputModel", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """Raised when a model response remains invalid after bounded correction."""


class AgentRuntime:
    def __init__(self, mesh_client: MeshClient, prompt_loader: PromptLoader):
        self._mesh_client = mesh_client
        self._prompt_loader = prompt_loader

    def generate_json(
        self,
        agent_config: PlatformAgentConfig,
        system_prompt_vars: dict[str, str],
        instruction_prompt_vars: dict[str, str],
        output_model: type[OutputModel],
    ) -> OutputModel:
        system_prompt = self._prompt_loader.render(
            agent_config.system_prompt_id, system_prompt_vars
        )
        instruction_prompt = self._prompt_loader.render(
            agent_config.instruction_prompt_id, instruction_prompt_vars
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction_prompt},
        ]
        retry_policy = RetryPolicy(max_attempts=agent_config.max_attempts)

        request = LLMRequest(
            model=agent_config.model,
            messages=messages,
            max_tokens=agent_config.max_tokens,
            timeout_seconds=agent_config.timeout_seconds,
        )
        try:
            return retry_policy.run(
                lambda: self._mesh_client.complete_structured(
                    request,
                    response_model=output_model,
                    max_retries=agent_config.max_output_corrections,
                )
            )
        except MeshStructuredOutputError as error:
            raise StructuredOutputError(str(error)) from error
