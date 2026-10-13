from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


def load_create_app():
    try:
        from app.main import create_app
    except ModuleNotFoundError as error:
        pytest.fail(f"runs API implementation is missing: {error.name}")
    return create_app


class FakeCreationService:
    def __init__(self, repository, status="needs_review"):
        self.repository = repository
        self.status = status

    def create_run(self, brief):
        run_id = str(uuid4())
        brief_data = brief.model_dump()
        state = {
            "status": self.status,
            "strategy": {
                "angle": "practical lessons",
                "audience": "engineers",
                "hook": "Safety needs feedback",
                "rationale": "Concrete lessons serve the brief.",
            },
            "draft": {"text": "Feedback makes safety testable."},
            "draft_history": [
                {"text": "AI safety matters."},
                {"text": "Feedback makes safety testable."},
            ],
            "latest_critique": {
                "score": 8.0,
                "approved": True,
                "critique": "Clear and specific.",
                "revision_instructions": [],
            },
            "editor_passed": True,
            "revision_count": 1,
        }
        if self.status == "failed":
            state = {
                "status": "failed",
                "failure_code": "mesh_request_failed",
                "provider_detail": "secret=rsk_test_value",
            }
        self.repository.create(run_id, brief_data)
        self.repository.save_state(run_id, state, status=self.status)
        self.repository.append_event(
            run_id,
            "brief_received",
            "user",
            "User submitted a content brief.",
            brief_data,
        )
        if self.status == "failed":
            self.repository.append_event(
                run_id,
                "run_failed",
                "workflow",
                "Workflow failed; inspect the sanitized failure code.",
                {"failure_code": "mesh_request_failed"},
            )
        else:
            self.repository.append_event(
                run_id,
                "run_needs_review",
                "workflow",
                "Draft is ready for human review.",
                {"editor_score": 8.0},
            )
        return self.repository.get(run_id)


def make_client(tmp_path, status="needs_review"):
    from app.config import Settings
    from app.db import Database
    from app.repositories.runs import RunRepository

    database = Database(f"sqlite:///{tmp_path / 'api.db'}")
    database.create_schema()
    repository = RunRepository(database.session_factory)
    service = FakeCreationService(repository, status=status)
    app = load_create_app()(
        Settings(app_env="test", database_url=f"sqlite:///{tmp_path / 'api.db'}", _env_file=None),
        creation_service=service,
        run_repository=repository,
        database=database,
    )
    return TestClient(app), database


def brief_payload():
    return {
        "topic": "AI safety",
        "platform": "linkedin",
        "audience": "engineers",
        "goal": "educate",
    }


def test_create_run_returns_reviewable_draft_and_persisted_events(tmp_path):
    client, database = make_client(tmp_path)
    try:
        create_response = client.post("/api/v1/runs", json=brief_payload())

        assert create_response.status_code == 200
        created = create_response.json()
        assert created["status"] == "needs_review"
        assert created["draft"] == {"text": "Feedback makes safety testable."}
        assert created["editor_score"] == 8.0
        run_id = created["run_id"]

        run_response = client.get(f"/api/v1/runs/{run_id}")
        event_response = client.get(f"/api/v1/runs/{run_id}/events")

        assert run_response.status_code == 200
        assert run_response.json()["revision_count"] == 1
        assert [event["event_type"] for event in event_response.json()] == [
            "brief_received",
            "run_needs_review",
        ]
    finally:
        client.close()
        database.close()


def test_unknown_run_and_event_requests_return_404(tmp_path):
    client, database = make_client(tmp_path)
    try:
        assert client.get("/api/v1/runs/missing").status_code == 404
        assert client.get("/api/v1/runs/missing/events").status_code == 404
    finally:
        client.close()
        database.close()


def test_failed_run_response_does_not_expose_provider_details(tmp_path):
    client, database = make_client(tmp_path, status="failed")
    try:
        response = client.post("/api/v1/runs", json=brief_payload())

        assert response.status_code == 502
        assert response.json()["status"] == "failed"
        assert response.json()["failure_code"] == "mesh_request_failed"
        assert "rsk_test_value" not in response.text
        assert "provider_detail" not in response.text
    finally:
        client.close()
        database.close()


def test_api_runs_real_workflow_with_fake_mesh_and_temporary_sqlite(tmp_path):
    from app.agents.editor import EditorAgent
    from app.agents.runtime import AgentRuntime
    from app.agents.safety import SafetyAgent
    from app.agents.strategist import StrategistAgent
    from app.agents.writer import WriterAgent
    from app.config import PlatformAgentConfig, Settings
    from app.db import Database
    from app.prompts.loader import PromptLoader
    from app.repositories.runs import RunRepository
    from app.services.creation import CreationService
    from app.workflows.content_graph import build_content_graph

    class FakeMeshClient:
        def __init__(self):
            self.responses = iter(
                [
                    '{"allowed":true,"reason":"safe","risk_categories":[]}',
                    '{"angle":"practical","audience":"engineers","hook":"A safer loop","rationale":"Specific and useful."}',
                    '{"text":"AI safety matters."}',
                    '{"score":5,"approved":false,"critique":"Hook is generic.","revision_instructions":["Use a concrete outcome."]}',
                    '{"text":"Feedback makes AI safety testable."}',
                    '{"score":8,"approved":true,"critique":"The revision is specific.","revision_instructions":[]}',
                ]
            )
            self.requests = []
            self.retry_budgets = []

        def complete_structured(self, request, response_model, max_retries):
            self.requests.append(request)
            self.retry_budgets.append(max_retries)
            return response_model.model_validate_json(next(self.responses))

    project_root = Path(__file__).resolve().parents[1]
    config_dir = project_root / "platform_configs"
    configs = {
        agent_id: PlatformAgentConfig.model_validate_json(
            (config_dir / f"linkedin.{agent_id}.json").read_text(encoding="utf-8")
        )
        for agent_id in ("safety", "strategist", "writer", "editor")
    }
    mesh_client = FakeMeshClient()
    runtime = AgentRuntime(mesh_client, PromptLoader(project_root / "agent_prompts"))
    agents = {
        "safety": SafetyAgent(runtime, configs["safety"]),
        "strategist": StrategistAgent(runtime, configs["strategist"]),
        "writer": WriterAgent(runtime, configs["writer"]),
        "editor": EditorAgent(runtime, configs["editor"]),
    }
    database_url = f"sqlite:///{tmp_path / 'end_to_end.db'}"
    database = Database(database_url)
    database.create_schema()
    repository = RunRepository(database.session_factory)
    graph = build_content_graph(agents, repository, configs["editor"])
    service = CreationService(graph, repository)
    app = load_create_app()(
        Settings(app_env="test", database_url=database_url, _env_file=None),
        creation_service=service,
        run_repository=repository,
        database=database,
    )

    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/runs", json=brief_payload())

        assert response.status_code == 200
        assert response.json()["status"] == "needs_review"
        assert response.json()["draft"]["text"] == "Feedback makes AI safety testable."
        assert response.json()["editor_passed"] is True
        assert len(mesh_client.requests) == 6
        assert mesh_client.retry_budgets == [1] * 6
    finally:
        database.close()
