from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.agents.reflection import LearningInsight
from app.models.performance import PerformanceMetrics
from app.repositories.performance import PerformanceRepository
from app.repositories.runs import RunNotFoundError, RunRepository


class FeedbackInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["user_entered", "simulated"]
    metrics: PerformanceMetrics
    published_at: datetime


class FeedbackResult(BaseModel):
    feedback_id: str
    event_id: str
    source: Literal["user_entered", "simulated"]
    engagement_rate: float | None
    playbook_version: int | None


class LearningServiceError(RuntimeError):
    """A sanitized failure while deriving a playbook update."""


class BrandAssociationRequiredError(ValueError):
    """Raised when a run has no brand ID to associate with performance feedback."""


class LearningService:
    def __init__(
        self,
        run_repository: RunRepository,
        performance_repository: PerformanceRepository,
        reflection_agent,
    ):
        self._run_repository = run_repository
        self._performance_repository = performance_repository
        self._reflection_agent = reflection_agent

    def record_feedback(self, run_id: str, feedback_input: FeedbackInput) -> FeedbackResult:
        run = self._run_repository.get(run_id)
        if run is None:
            raise RunNotFoundError(run_id)
        brand_id = run.brief.get("brand_id")
        if not brand_id:
            raise BrandAssociationRequiredError("content run has no brand association")

        feedback = self._performance_repository.record_feedback(
            run_id,
            feedback_input.source,
            feedback_input.metrics,
            feedback_input.published_at,
        )
        self._run_repository.append_event(
            run_id,
            "performance_recorded",
            feedback_input.source,
            "Recorded explicitly labeled performance feedback.",
            {
                "feedback_id": feedback.id,
                "source": feedback.source,
                "metrics": feedback.metrics,
                "engagement_rate": feedback.engagement_rate,
            },
        )

        active_playbook = self._performance_repository.get_active_playbook(brand_id)
        if feedback.engagement_rate is None:
            skipped_event = self._run_repository.append_event(
                run_id,
                "learning_skipped",
                "reflection",
                "Learning was skipped because impressions were zero.",
                {"feedback_id": feedback.id},
            )
            return FeedbackResult(
                feedback_id=feedback.id,
                event_id=skipped_event.id,
                source=feedback.source,
                engagement_rate=None,
                playbook_version=active_playbook.version if active_playbook else None,
            )

        try:
            insight: LearningInsight = self._reflection_agent.analyze(feedback, active_playbook)
        except Exception as error:  # noqa: BLE001
            self._run_repository.append_event(
                run_id,
                "learning_failed",
                "reflection",
                "Reflection failed; the active playbook was retained.",
                {"feedback_id": feedback.id, "failure_type": type(error).__name__},
            )
            raise LearningServiceError(
                "performance feedback was recorded but learning failed"
            ) from None

        self._run_repository.append_event(
            run_id,
            "learning_proposed",
            "reflection",
            insight.rationale,
            {
                "feedback_id": feedback.id,
                "rules": [rule.model_dump() for rule in insight.rules],
            },
        )
        playbook = self._performance_repository.save_playbook_version(
            brand_id,
            insight.rules,
            [feedback.id],
            insight.rationale,
        )
        updated_event = self._run_repository.append_event(
            run_id,
            "playbook_updated",
            "reflection",
            insight.rationale,
            {
                "feedback_id": feedback.id,
                "playbook_version": playbook.version,
                "evidence_feedback_ids": playbook.evidence_feedback_ids,
            },
        )
        return FeedbackResult(
            feedback_id=feedback.id,
            event_id=updated_event.id,
            source=feedback.source,
            engagement_rate=feedback.engagement_rate,
            playbook_version=playbook.version,
        )
