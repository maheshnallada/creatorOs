import json

from app.agents.contracts import (
    BrandMemoryContext,
    ContentBrief,
    DraftContent,
    EditorCritique,
    StrategyDecision,
)
from app.agents.runtime import AgentRuntime
from app.config import PlatformAgentConfig


class WriterAgent:
    def __init__(self, runtime: AgentRuntime, config: PlatformAgentConfig):
        self._runtime = runtime
        self._config = config

    def write(
        self,
        brief: ContentBrief,
        strategy: StrategyDecision,
        critique: EditorCritique | None = None,
        memory: BrandMemoryContext | None = None,
    ) -> DraftContent:
        critique_data = critique.model_dump() if critique is not None else {}
        memory_context = memory or BrandMemoryContext()
        return self._runtime.generate_json(
            self._config,
            system_prompt_vars={},
            instruction_prompt_vars={
                "brief_json": json.dumps(brief.model_dump(), ensure_ascii=False),
                "strategy_json": json.dumps(strategy.model_dump(), ensure_ascii=False),
                "critique_json": json.dumps(critique_data, ensure_ascii=False),
                "brand_memory_json": json.dumps(
                    memory_context.model_dump(mode="json"), ensure_ascii=False
                ),
            },
            output_model=DraftContent,
        )
