from datetime import datetime
from typing import Literal

from sqlalchemy import func, select, update
from sqlalchemy.orm import sessionmaker

from app.agents.contracts import PlaybookRule
from app.models.performance import (
    PerformanceFeedback,
    PerformanceMetrics,
    PlaybookVersion,
)
from app.models.run import ContentRun


class PerformanceRepository:
    def __init__(self, session_factory: sessionmaker):
        self._session_factory = session_factory

    def record_feedback(
        self,
        run_id: str,
        source: Literal["user_entered", "simulated"],
        metrics: PerformanceMetrics,
        published_at: datetime,
    ) -> PerformanceFeedback:
        if source not in {"user_entered", "simulated"}:
            raise ValueError("unsupported performance feedback source")

        with self._session_factory.begin() as session:
            run = session.get(ContentRun, run_id)
            if run is None:
                raise LookupError(f"unknown content run: {run_id}")
            brand_id = run.brief.get("brand_id")
            if not brand_id:
                raise ValueError("content run must reference brand_id before recording feedback")
            feedback = PerformanceFeedback(
                run_id=run_id,
                brand_id=brand_id,
                source=source,
                metrics=metrics.model_dump(),
                engagement_rate=metrics.engagement_rate,
                published_at=published_at,
            )
            session.add(feedback)
        return feedback

    def list_feedback(self, brand_id: str) -> list[PerformanceFeedback]:
        with self._session_factory() as session:
            return list(
                session.scalars(
                    select(PerformanceFeedback)
                    .where(PerformanceFeedback.brand_id == brand_id)
                    .order_by(PerformanceFeedback.created_at, PerformanceFeedback.id)
                )
            )

    def save_playbook_version(
        self,
        brand_id: str,
        rules: list[PlaybookRule],
        evidence_feedback_ids: list[str],
        rationale: str,
    ) -> PlaybookVersion:
        unique_feedback_ids = list(dict.fromkeys(evidence_feedback_ids))
        if not unique_feedback_ids:
            raise ValueError("playbook versions require feedback evidence")
        if not rationale.strip():
            raise ValueError("playbook rationale must not be empty")

        with self._session_factory.begin() as session:
            evidence = list(
                session.scalars(
                    select(PerformanceFeedback).where(
                        PerformanceFeedback.brand_id == brand_id,
                        PerformanceFeedback.id.in_(unique_feedback_ids),
                    )
                )
            )
            if len(evidence) != len(unique_feedback_ids):
                raise ValueError("evidence feedback must belong to the requested brand")

            current_version = (
                session.scalar(
                    select(func.max(PlaybookVersion.version)).where(
                        PlaybookVersion.brand_id == brand_id
                    )
                )
                or 0
            )
            session.execute(
                update(PlaybookVersion)
                .where(PlaybookVersion.brand_id == brand_id)
                .values(is_active=False)
            )
            version = PlaybookVersion(
                brand_id=brand_id,
                version=current_version + 1,
                rules=[rule.model_dump() for rule in rules],
                evidence_feedback_ids=unique_feedback_ids,
                rationale=rationale.strip(),
                is_active=True,
            )
            session.add(version)
        return version

    def get_active_playbook(self, brand_id: str) -> PlaybookVersion | None:
        with self._session_factory() as session:
            return session.scalar(
                select(PlaybookVersion).where(
                    PlaybookVersion.brand_id == brand_id,
                    PlaybookVersion.is_active.is_(True),
                )
            )

    def list_playbook_versions(self, brand_id: str) -> list[PlaybookVersion]:
        with self._session_factory() as session:
            return list(
                session.scalars(
                    select(PlaybookVersion)
                    .where(PlaybookVersion.brand_id == brand_id)
                    .order_by(PlaybookVersion.version)
                )
            )
