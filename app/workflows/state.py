from typing import Any, TypedDict


class ContentState(TypedDict, total=False):
    run_id: str
    brief: dict[str, Any]
    safety_verdict: dict[str, Any]
    strategy: dict[str, Any]
    draft: dict[str, Any]
    draft_history: list[dict[str, Any]]
    latest_critique: dict[str, Any]
    revision_count: int
    revision_requested: bool
    editor_passed: bool
    status: str
    failure_code: str
