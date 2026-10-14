from uuid import uuid4

from app.agents.contracts import ContentBrief
from app.agents.runtime import StructuredOutputError
from app.llm.mesh_client import LLMProviderRequestError
from app.repositories.runs import RunRepository
from app.workflows.state import ContentState


class CreationService:
    def __init__(self, graph, repository: RunRepository):
        self._graph = graph
        self._repository = repository

    def create_run(self, brief: ContentBrief):
        run_id = str(uuid4())
        brief_data = brief.model_dump()
        initial_state: ContentState = {
            "run_id": run_id,
            "brief": brief_data,
            "draft_history": [],
            "revision_count": 0,
            "editor_passed": False,
            "status": "running",
        }
        self._repository.create(run_id, brief_data)
        self._repository.append_event(
            run_id,
            "brief_received",
            "user",
            "User submitted a content brief.",
            brief_data,
        )
        self._repository.save_state(run_id, dict(initial_state), status="running")

        try:
            self._graph.invoke(initial_state)
        except Exception as error:  # noqa: BLE001
            self._persist_failure(run_id, error)

        return self._repository.get(run_id)

    def _persist_failure(self, run_id: str, error: Exception) -> None:
        if isinstance(error, LLMProviderRequestError):
            failure_code = (
                "groq_request_failed" if error.provider == "groq" else "mesh_request_failed"
            )
        elif isinstance(error, StructuredOutputError):
            failure_code = "invalid_structured_output"
        else:
            failure_code = "workflow_failed"

        run = self._repository.get(run_id)
        state = dict(run.workflow_state) if run is not None else {"run_id": run_id}
        state.update({"status": "failed", "failure_code": failure_code})
        self._repository.save_state(run_id, state, status="failed")
        self._repository.append_event(
            run_id,
            "run_failed",
            "workflow",
            "Workflow failed; inspect the sanitized failure code.",
            {"failure_code": failure_code},
        )
