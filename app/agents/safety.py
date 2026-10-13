from app.agents.contracts import SafetyVerdict
from app.agents.runtime import AgentRuntime
from app.config import PlatformAgentConfig


class SafetyAgent:
    def __init__(self, runtime: AgentRuntime, config: PlatformAgentConfig):
        self._runtime = runtime
        self._config = config

    def check(self, text: str, context: str) -> SafetyVerdict:
        return self._runtime.generate_json(
            self._config,
            system_prompt_vars={},
            instruction_prompt_vars={"context": context, "text": text},
            output_model=SafetyVerdict,
        )
