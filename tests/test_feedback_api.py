from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


class FakeLearningService:
    def __init__(self, feedback_result=None, failure=None):
        self.feedback_result = feedback_result
        self.failure = failure
        self.calls = []

    def record_feedback(self, run_id, feedback_input):
        self.calls.append((run_id, feedback_input))
        if self.failure is not None:
            raise self.failure
        return self.feedback_result


def make_client(tmp_path: Path, learning_service, *, raise_server_exceptions=True):
    from app.config import Settings
    from app.db import Database
    from app.repositories.performance import PerformanceRepository
    from app.repositories.runs import RunRepository

    database_url = f"sqlite:///{tmp_path / 'feedback-api.db'}"
    database = Database(database_url)
    database.create_schema()
    run_repository = RunRepository(database.session_factory)
    run_repository.create("run-1", {"brand_id": "brand-1", "topic": "AI safety"})
    performance_repository = PerformanceRepository(database.session_factory)
    app = _load_create_app()(
        Settings(app_env="test", database_url=database_url, _env_file=None),
        creation_service=object(),
        run_repository=run_repository,
        performance_repository=performance_repository,
        database=database,
        learning_service=learning_service,
    )
    return (
        TestClient(app, raise_server_exceptions=raise_server_exceptions),
        database,
        run_repository,
    )


def _load_create_app():
    try:
        from app.main import create_app
    except ModuleNotFoundError as error:
        pytest.fail(f"feedback API implementation is missing: {error.name}")
    return create_app


def test_feedback_endpoint_returns_provenance_and_updated_version(tmp_path: Path):
    from app.services.learning import FeedbackResult

    learning_service = FakeLearningService(
        FeedbackResult(
            feedback_id="feedback-1",
            event_id="event-3",
            source="user_entered",
            engagement_rate=0.1,
            playbook_version=2,
        )
    )
    client, database, _ = make_client(tmp_path, learning_service)
    try:
        response = client.post(
            "/api/v1/runs/run-1/feedback",
            json={
                "source": "user_entered",
                "published_at": datetime(2026, 10, 13, tzinfo=UTC).isoformat(),
                "metrics": {"impressions": 100, "likes": 10},
            },
        )

        assert response.status_code == 200
        assert response.json() == {
            "feedback_id": "feedback-1",
            "event_id": "event-3",
            "source": "user_entered",
            "engagement_rate": 0.1,
            "playbook_version": 2,
        }
        assert learning_service.calls[0][0] == "run-1"
        assert learning_service.calls[0][1].source == "user_entered"
    finally:
        client.close()
        database.close()


def test_feedback_endpoint_rejects_invalid_source_and_unknown_runs(tmp_path: Path):
    from app.services.learning import FeedbackResult

    client, database, _ = make_client(
        tmp_path,
        FakeLearningService(
            FeedbackResult(
                feedback_id="feedback-1",
                event_id="event-3",
                source="simulated",
                engagement_rate=None,
                playbook_version=1,
            )
        ),
    )
    try:
        invalid_source = client.post(
            "/api/v1/runs/run-1/feedback",
            json={
                "source": "platform_fetched",
                "published_at": "2026-10-13T00:00:00Z",
                "metrics": {"impressions": 100},
            },
        )
        unknown = client.post(
            "/api/v1/runs/unknown/feedback",
            json={
                "source": "simulated",
                "published_at": "2026-10-13T00:00:00Z",
                "metrics": {"impressions": 100},
            },
        )

        assert invalid_source.status_code == 422
        assert unknown.status_code == 404
    finally:
        client.close()
        database.close()


def test_feedback_endpoint_sanitizes_unexpected_service_errors(tmp_path: Path):
    client, database, _ = make_client(
        tmp_path,
        FakeLearningService(failure=RuntimeError("private database detail")),
        raise_server_exceptions=False,
    )
    try:
        response = client.post(
            "/api/v1/runs/run-1/feedback",
            json={
                "source": "simulated",
                "published_at": "2026-10-14T12:00:00Z",
                "metrics": {"impressions": 100, "likes": 10},
            },
        )

        assert response.status_code == 500
        assert "private database detail" not in response.text
    finally:
        client.close()
        database.close()
