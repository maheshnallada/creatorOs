import json

from app.agents.contracts import (
    ContentBrief,
    DraftContent,
    EditorCritique,
    StrategyDecision,
)
from app.agents.runtime import AgentRuntime
from app.config import PlatformAgentConfig


class EditorAgent:
    def __init__(self, runtime: AgentRuntime, config: PlatformAgentConfig):
        self._runtime = runtime
        self._config = config

    def review(
        self,
        brief: ContentBrief,
        strategy: StrategyDecision,
        draft: DraftContent,
    ) -> EditorCritique:
        return self._runtime.generate_json(
            self._config,
            system_prompt_vars={},
            instruction_prompt_vars={
                "brief_json": json.dumps(brief.model_dump(), ensure_ascii=False),
                "strategy_json": json.dumps(strategy.model_dump(), ensure_ascii=False),
                "draft_json": json.dumps(draft.model_dump(), ensure_ascii=False),
            },
            output_model=EditorCritique,
        )
