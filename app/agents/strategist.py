import json

from app.agents.contracts import ContentBrief, StrategyDecision
from app.agents.runtime import AgentRuntime
from app.config import PlatformAgentConfig


class StrategistAgent:
    def __init__(self, runtime: AgentRuntime, config: PlatformAgentConfig):
        self._runtime = runtime
        self._config = config

    def plan(self, brief: ContentBrief) -> StrategyDecision:
        return self._runtime.generate_json(
            self._config,
            system_prompt_vars={},
            instruction_prompt_vars={
                "brief_json": json.dumps(brief.model_dump(), ensure_ascii=False)
            },
            output_model=StrategyDecision,
        )
