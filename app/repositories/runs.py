from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from app.models.run import ContentRun, DecisionEvent


class RunNotFoundError(LookupError):
    """Raised when a run-linked write targets an unknown run."""


class RunRepository:
    def __init__(self, session_factory: sessionmaker):
        self._session_factory = session_factory

    def create(self, run_id: str, brief: dict[str, Any]) -> ContentRun:
        run = ContentRun(id=run_id, brief=brief)
        with self._session_factory.begin() as session:
            session.add(run)
        return run

    def get(self, run_id: str) -> ContentRun | None:
        with self._session_factory() as session:
            return session.get(ContentRun, run_id)

    def save_state(self, run_id: str, state: dict[str, Any], status: str) -> ContentRun:
        with self._session_factory.begin() as session:
            run = session.scalar(
                select(ContentRun).where(ContentRun.id == run_id).with_for_update()
            )
            if run is None:
                raise RunNotFoundError(run_id)
            run.workflow_state = state
            run.status = status
            run.updated_at = datetime.now(UTC)
        return run

    def append_event(
        self,
        run_id: str,
        event_type: str,
        agent_id: str,
        reason: str,
        payload: dict[str, Any],
    ) -> DecisionEvent:
        with self._session_factory.begin() as session:
            run = session.scalar(
                select(ContentRun).where(ContentRun.id == run_id).with_for_update()
            )
            if run is None:
                raise RunNotFoundError(run_id)
            last_sequence = session.scalar(
                select(func.max(DecisionEvent.sequence)).where(DecisionEvent.run_id == run_id)
            )
            event = DecisionEvent(
                run_id=run_id,
                sequence=(last_sequence or 0) + 1,
                event_type=event_type,
                agent_id=agent_id,
                reason=reason,
                payload=payload,
            )
            session.add(event)
        return event

    def list_events(self, run_id: str) -> list[DecisionEvent]:
        with self._session_factory() as session:
            return list(
                session.scalars(
                    select(DecisionEvent)
                    .where(DecisionEvent.run_id == run_id)
                    .order_by(DecisionEvent.sequence)
                )
            )
