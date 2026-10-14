import json

from pydantic import BaseModel, ConfigDict, Field

from app.agents.contracts import PlaybookRule
from app.agents.runtime import AgentRuntime
from app.config import PlatformAgentConfig
from app.models.performance import PerformanceFeedback, PlaybookVersion


class LearningInsight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rationale: str = Field(min_length=1, max_length=1500)
    rules: list[PlaybookRule] = Field(min_length=1, max_length=20)


class ReflectionAgent:
    def __init__(self, runtime: AgentRuntime, config: PlatformAgentConfig):
        self._runtime = runtime
        self._config = config

    def analyze(
        self,
        feedback: PerformanceFeedback,
        prior_playbook: PlaybookVersion | None,
    ) -> LearningInsight:
        feedback_data = {
            "feedback_id": feedback.id,
            "brand_id": feedback.brand_id,
            "source": feedback.source,
            "metrics": feedback.metrics,
            "engagement_rate": feedback.engagement_rate,
            "published_at": feedback.published_at.isoformat(),
        }
        prior_data = (
            {
                "version": prior_playbook.version,
                "rules": prior_playbook.rules,
            }
            if prior_playbook is not None
            else {}
        )
        return self._runtime.generate_json(
            self._config,
            system_prompt_vars={},
            instruction_prompt_vars={
                "feedback_json": json.dumps(feedback_data, ensure_ascii=False),
                "prior_playbook_json": json.dumps(prior_data, ensure_ascii=False),
            },
            output_model=LearningInsight,
        )
