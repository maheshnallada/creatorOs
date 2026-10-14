from datetime import UTC, datetime
from pathlib import Path

import pytest


def load_workflow_types():
    try:
        from app.agents.contracts import (
            ContentBrief,
            DraftContent,
            EditorCritique,
            SafetyVerdict,
            StrategyDecision,
        )
        from app.services.creation import CreationService
        from app.workflows.content_graph import build_content_graph
    except ModuleNotFoundError as error:
        pytest.fail(f"content workflow implementation is missing: {error.name}")
    return (
        ContentBrief,
        CreationService,
        DraftContent,
        EditorCritique,
        SafetyVerdict,
        StrategyDecision,
        build_content_graph,
    )


class FakeSafetyAgent:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def check(self, text, context):
        self.calls.append((text, context))
        return self.result


class FakeStrategistAgent:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self.memories = []

    def plan(self, brief, memory=None):
        self.calls.append(brief)
        self.memories.append(memory)
        return self.result


class FakeWriterAgent:
    def __init__(self, outputs=None, failure=None):
        self.outputs = list(outputs or [])
        self.failure = failure
        self.calls = []
        self.memories = []

    def write(self, brief, strategy, critique=None, memory=None):
        self.calls.append(critique)
        self.memories.append(memory)
        if self.failure is not None:
            if callable(self.failure):
                return self.failure()
            raise self.failure
        return self.outputs.pop(0)


class FakeEditorAgent:
    def __init__(self, critiques):
        self.critiques = list(critiques)
        self.calls = []

    def review(self, brief, strategy, draft):
        self.calls.append(draft)
        return self.critiques.pop(0)


def make_service(
    tmp_path: Path,
    *,
    safety_allowed=True,
    writer_outputs=None,
    writer_failure=None,
    critiques=None,
    max_editor_revisions=2,
    quality_threshold=7.0,
):
    from app.config import PlatformAgentConfig
    from app.db import Database
    from app.repositories.brand_memory import BrandMemoryRepository
    from app.repositories.performance import PerformanceRepository
    from app.repositories.runs import RunRepository

    (
        _,
        service_type,
        draft_type,
        critique_type,
        safety_type,
        strategy_type,
        build_graph,
    ) = load_workflow_types()
    database = Database(f"sqlite:///{tmp_path / 'workflow.db'}")
    database.create_schema()
    repository = RunRepository(database.session_factory)
    brand_memory_repository = BrandMemoryRepository(database.session_factory)
    performance_repository = PerformanceRepository(database.session_factory)
    safety_agent = FakeSafetyAgent(
        safety_type(
            allowed=safety_allowed,
            reason="brief is safe" if safety_allowed else "unsafe instruction",
            risk_categories=[] if safety_allowed else ["prompt_injection"],
        )
    )
    strategist_agent = FakeStrategistAgent(
        strategy_type(
            angle="practical lessons",
            audience="engineers",
            hook="Safety needs feedback",
            rationale="It makes the post concrete.",
        )
    )
    writer_agent = FakeWriterAgent(
        outputs=[draft_type(text=text) for text in (writer_outputs or [])],
        failure=writer_failure,
    )
    editor_agent = FakeEditorAgent(
        [
            critique_type(
                score=score,
                approved=approved,
                critique=critique,
                revision_instructions=instructions,
            )
            for score, approved, critique, instructions in (critiques or [])
        ]
    )
    agents = {
        "safety": safety_agent,
        "strategist": strategist_agent,
        "writer": writer_agent,
        "editor": editor_agent,
    }
    workflow_config = PlatformAgentConfig(
        platform="linkedin",
        agent_id="editor",
        model="openai/omni-moderation-latest",
        system_prompt_id="linkedin.editor.system",
        instruction_prompt_id="linkedin.editor.instructions",
        max_editor_revisions=max_editor_revisions,
        editor_quality_threshold=quality_threshold,
    )
    graph = build_graph(
        agents,
        repository,
        workflow_config,
        brand_memory_repository=brand_memory_repository,
        performance_repository=performance_repository,
    )
    service = service_type(graph, repository)
    return database, repository, service, graph, agents


def make_brief(brief_type):
    return brief_type(
        topic="AI safety",
        platform="linkedin",
        audience="engineers",
        goal="educate",
    )


def test_workflow_uses_named_nodes_and_editor_revision_changes_draft(tmp_path: Path):
    brief_type, _, _, _, _, _, _ = load_workflow_types()
    database, repository, service, graph, agents = make_service(
        tmp_path,
        writer_outputs=["AI safety is important.", "Feedback makes safety testable."],
        critiques=[
            (5.0, False, "The hook is generic.", ["Use a concrete outcome in the opening."]),
            (8.0, True, "The hook is now specific.", []),
        ],
    )
    try:
        result = service.create_run(make_brief(brief_type))

        assert {"safety", "strategist", "writer", "editor", "finish"}.issubset(
            set(graph.get_graph().nodes)
        )
        assert result.status == "needs_review"
        assert result.workflow_state["editor_passed"] is True
        assert result.workflow_state["draft"]["text"] == "Feedback makes safety testable."
        assert [draft["text"] for draft in result.workflow_state["draft_history"]] == [
            "AI safety is important.",
            "Feedback makes safety testable.",
        ]
        assert agents["writer"].calls[1].revision_instructions == [
            "Use a concrete outcome in the opening."
        ]
        assert [event.event_type for event in repository.list_events(result.id)] == [
            "brief_received",
            "safety_checked",
            "brand_memory_loaded",
            "strategy_selected",
            "draft_created",
            "editor_critiqued",
            "draft_revised",
            "editor_critiqued",
            "run_needs_review",
        ]
    finally:
        database.close()


def test_unsafe_brief_stops_before_strategist_or_writer(tmp_path: Path):
    brief_type, _, _, _, _, _, _ = load_workflow_types()
    database, repository, service, _, agents = make_service(tmp_path, safety_allowed=False)
    try:
        result = service.create_run(make_brief(brief_type))

        assert result.status == "blocked"
        assert agents["strategist"].calls == []
        assert agents["writer"].calls == []
        assert [event.event_type for event in repository.list_events(result.id)] == [
            "brief_received",
            "safety_checked",
            "run_blocked",
        ]
    finally:
        database.close()


def test_editor_threshold_controls_revision_route(tmp_path: Path):
    brief_type, _, _, _, _, _, _ = load_workflow_types()
    database, _, service, _, agents = make_service(
        tmp_path,
        writer_outputs=["Initial draft", "Second draft"],
        critiques=[
            (4.0, True, "Score is below threshold.", ["Improve specificity."]),
            (6.5, True, "Still below threshold.", ["Improve the opening."]),
        ],
        max_editor_revisions=1,
        quality_threshold=7.0,
    )
    try:
        result = service.create_run(make_brief(brief_type))

        assert len(agents["writer"].calls) == 2
        assert result.status == "needs_review"
        assert result.workflow_state["editor_passed"] is False
        assert result.workflow_state["revision_count"] == 1
    finally:
        database.close()


def test_unchanged_draft_stops_at_revision_cap(tmp_path: Path):
    brief_type, _, _, _, _, _, _ = load_workflow_types()
    database, _, service, _, _ = make_service(
        tmp_path,
        writer_outputs=["Same draft", "Same draft"],
        critiques=[
            (4.0, False, "Needs a clearer hook.", ["Rewrite the hook."]),
            (4.0, False, "The hook is still unclear.", ["Try a different hook."]),
        ],
        max_editor_revisions=1,
    )
    try:
        result = service.create_run(make_brief(brief_type))

        assert result.status == "needs_review"
        assert result.workflow_state["editor_passed"] is False
        assert result.workflow_state["draft"]["text"] == "Same draft"
        assert result.workflow_state["revision_count"] == 1
    finally:
        database.close()


def test_mesh_failure_records_sanitized_failed_run(tmp_path: Path):
    from app.llm.mesh_client import MeshRequestError

    brief_type, _, _, _, _, _, _ = load_workflow_types()
    database, repository, service, _, _ = make_service(
        tmp_path,
        writer_failure=MeshRequestError("Mesh API request failed with HTTP 401"),
    )
    try:
        result = service.create_run(make_brief(brief_type))

        assert result.status == "failed"
        assert result.workflow_state["failure_code"] == "mesh_request_failed"
        event_types = [event.event_type for event in repository.list_events(result.id)]
        assert event_types[-1] == "run_failed"
        assert "401" not in repository.list_events(result.id)[-1].reason
    finally:
        database.close()


def test_transient_mesh_retry_exhaustion_records_failed_run(tmp_path: Path):
    from app.llm.mesh_client import MeshRequestError
    from app.llm.retry import RetryPolicy

    brief_type, _, _, _, _, _, _ = load_workflow_types()
    calls = 0

    def transient_request():
        nonlocal calls
        calls += 1
        raise MeshRequestError("Mesh API request failed with HTTP 503", retryable=True)

    def exhausted_request():
        return RetryPolicy(max_attempts=2, sleep=lambda _: None).run(transient_request)

    database, repository, service, _, _ = make_service(
        tmp_path,
        writer_failure=exhausted_request,
    )
    try:
        result = service.create_run(make_brief(brief_type))

        assert calls == 2
        assert result.status == "failed"
        assert result.workflow_state["failure_code"] == "mesh_request_failed"
        assert repository.list_events(result.id)[-1].event_type == "run_failed"
    finally:
        database.close()


def test_workflow_loads_and_audits_active_profile_and_playbook_versions(tmp_path: Path):
    from app.agents.contracts import ContentBrief, PlaybookRule
    from app.models.performance import PerformanceMetrics
    from app.repositories.brand_memory import BrandMemoryRepository
    from app.repositories.performance import PerformanceRepository

    database, repository, service, _, agents = make_service(
        tmp_path,
        writer_outputs=["A brand-aware post."],
        critiques=[(8.0, True, "Matches the profile.", [])],
    )
    brand_repository = BrandMemoryRepository(database.session_factory)
    performance_repository = PerformanceRepository(database.session_factory)
    try:
        samples = brand_repository.add_samples(
            "brand-1",
            [
                {"body": "Concise technical post one."},
                {"body": "Concise technical post two."},
                {"body": "Concise technical post three."},
            ],
        )
        profile = brand_repository.save_profile(
            "brand-1",
            {
                "persona": "Concise technical builder",
                "tone_attributes": ["concise"],
                "preferred_terms": [],
                "avoid_terms": [],
                "structure_guidelines": ["Short paragraphs"],
            },
            [sample.id for sample in samples],
        )
        repository.create("history-run", {"brand_id": "brand-1", "topic": "Previous"})
        feedback = performance_repository.record_feedback(
            "history-run",
            source="simulated",
            metrics=PerformanceMetrics(impressions=100, likes=10),
            published_at=datetime.now(UTC),
        )
        playbook = performance_repository.save_playbook_version(
            "brand-1",
            [PlaybookRule(rule="Lead with a concrete result", platform="linkedin")],
            [feedback.id],
            "This rule reflects the current successful sample.",
        )
        brief = ContentBrief(
            topic="AI safety",
            platform="linkedin",
            audience="engineers",
            goal="educate",
            brand_id="brand-1",
        )

        result = service.create_run(brief)

        context = result.workflow_state["brand_memory_context"]
        assert context["profile_version"] == profile.version
        assert context["playbook_version"] == playbook.version
        assert agents["strategist"].memories[0].profile_version == profile.version
        assert agents["writer"].memories[0].playbook_version == playbook.version
        memory_event = next(
            event
            for event in repository.list_events(result.id)
            if event.event_type == "brand_memory_loaded"
        )
        assert memory_event.payload == {
            "profile_version": profile.version,
            "playbook_version": playbook.version,
        }
    finally:
        database.close()
