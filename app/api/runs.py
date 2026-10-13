from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.agents.contracts import ContentBrief
from app.models.run import ContentRun, DecisionEvent

router = APIRouter(prefix="/api/v1", tags=["runs"])


class RunResponse(BaseModel):
    run_id: str
    brief: dict[str, Any]
    status: str
    strategy: dict[str, Any] | None = None
    draft: dict[str, Any] | None = None
    draft_history: list[dict[str, Any]]
    editor_score: float | None = None
    editor_passed: bool | None = None
    revision_count: int
    failure_code: str | None = None


class EventResponse(BaseModel):
    id: str
    run_id: str
    sequence: int
    event_type: str
    agent_id: str
    reason: str
    payload: dict[str, Any]


def serialize_run(run: ContentRun) -> RunResponse:
    state = run.workflow_state
    critique = state.get("latest_critique") or {}
    return RunResponse(
        run_id=run.id,
        brief=run.brief,
        status=run.status,
        strategy=state.get("strategy"),
        draft=state.get("draft"),
        draft_history=state.get("draft_history", []),
        editor_score=critique.get("score"),
        editor_passed=state.get("editor_passed"),
        revision_count=state.get("revision_count", 0),
        failure_code=state.get("failure_code"),
    )


@router.post("/runs", response_model=RunResponse)
def create_run(brief: ContentBrief, request: Request):
    run = request.app.state.creation_service.create_run(brief)
    if run.status == "failed":
        return JSONResponse(
            status_code=502,
            content={
                "run_id": run.id,
                "status": "failed",
                "failure_code": run.workflow_state.get("failure_code", "workflow_failed"),
                "message": "Content generation failed. Inspect the run event log for details.",
            },
        )
    return serialize_run(run)


@router.get("/runs/{run_id}", response_model=RunResponse)
def get_run(run_id: str, request: Request):
    run = request.app.state.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return serialize_run(run)


@router.get("/runs/{run_id}/events", response_model=list[EventResponse])
def get_run_events(run_id: str, request: Request):
    if request.app.state.run_repository.get(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    events: list[DecisionEvent] = request.app.state.run_repository.list_events(run_id)
    return [EventResponse.model_validate(event, from_attributes=True) for event in events]
