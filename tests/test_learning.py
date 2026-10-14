from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError


def load_learning_persistence():
    try:
        from app.agents.contracts import PlaybookRule
        from app.db import Database
        from app.models.performance import PerformanceMetrics
        from app.repositories.performance import PerformanceRepository
        from app.repositories.runs import RunRepository
    except ModuleNotFoundError as error:
        pytest.fail(f"performance memory persistence is missing: {error.name}")
    return Database, PerformanceMetrics, PerformanceRepository, PlaybookRule, RunRepository


def test_feedback_is_run_linked_and_playbook_versions_are_auditable(tmp_path: Path):
    database_type, metrics_type, performance_repo_type, playbook_rule_type, run_repo_type = (
        load_learning_persistence()
    )
    database = database_type(f"sqlite:///{tmp_path / 'learning.db'}")
    database.create_schema()
    run_repository = run_repo_type(database.session_factory)
    repository = performance_repo_type(database.session_factory)
    try:
        run_repository.create("run-1", {"topic": "AI safety", "brand_id": "brand-1"})
        metrics = metrics_type(
            impressions=1000,
            likes=100,
            comments=20,
            shares=10,
            clicks=40,
            saves=30,
        )
        feedback = repository.record_feedback(
            "run-1",
            source="simulated",
            metrics=metrics,
            published_at=datetime(2026, 10, 13, tzinfo=UTC),
        )
        playbook_v1 = repository.save_playbook_version(
            "brand-1",
            [playbook_rule_type(rule="Use concrete examples", platform="linkedin")],
            [feedback.id],
            "Specific examples accompanied higher simulated engagement.",
        )
        playbook_v2 = repository.save_playbook_version(
            "brand-1",
            [playbook_rule_type(rule="Lead with a measurable outcome", platform="linkedin")],
            [feedback.id],
            "Outcome-first hooks had the strongest observed response.",
        )

        assert feedback.run_id == "run-1"
        assert feedback.brand_id == "brand-1"
        assert feedback.source == "simulated"
        assert feedback.engagement_rate == pytest.approx(0.2)
        assert [item.id for item in repository.list_feedback("brand-1")] == [feedback.id]
        assert playbook_v1.version == 1
        assert playbook_v2.version == 2
        assert playbook_v2.is_active is True
        assert playbook_v2.evidence_feedback_ids == [feedback.id]
        assert [version.is_active for version in repository.list_playbook_versions("brand-1")] == [
            False,
            True,
        ]
        assert repository.get_active_playbook("brand-1").id == playbook_v2.id
    finally:
        database.close()


def test_engagement_rate_is_none_when_impressions_are_zero():
    _, metrics_type, _, _, _ = load_learning_persistence()

    metrics = metrics_type(impressions=0, likes=2)

    assert metrics.engagement_rate is None


@pytest.mark.parametrize(
    "values",
    [
        {"impressions": -1},
        {"impressions": 10, "likes": -1},
    ],
)
def test_performance_metrics_reject_negative_counts(values):
    _, metrics_type, _, _, _ = load_learning_persistence()

    with pytest.raises(ValidationError):
        metrics_type(**values)


def test_feedback_for_unknown_run_is_rejected_without_orphan_record(tmp_path: Path):
    database_type, metrics_type, performance_repo_type, _, _ = load_learning_persistence()
    database = database_type(f"sqlite:///{tmp_path / 'orphan.db'}")
    database.create_schema()
    repository = performance_repo_type(database.session_factory)
    try:
        with pytest.raises(LookupError):
            repository.record_feedback(
                "unknown-run",
                source="user_entered",
                metrics=metrics_type(impressions=100, likes=10),
                published_at=datetime(2026, 10, 13, tzinfo=UTC),
            )

        assert repository.list_feedback("brand-1") == []
    finally:
        database.close()


class FakeReflectionAgent:
    def __init__(self, insight=None, failure=None):
        self.insight = insight
        self.failure = failure
        self.calls = []

    def analyze(self, feedback, prior_playbook):
        self.calls.append((feedback, prior_playbook))
        if self.failure is not None:
            raise self.failure
        return self.insight


def test_learning_service_records_feedback_and_versions_playbook(tmp_path: Path):
    try:
        from app.services.learning import FeedbackInput, LearningInsight, LearningService
    except ModuleNotFoundError as error:
        pytest.fail(f"learning service implementation is missing: {error.name}")
    from app.agents.contracts import PlaybookRule

    database_type, metrics_type, performance_repo_type, _, run_repo_type = (
        load_learning_persistence()
    )
    database = database_type(f"sqlite:///{tmp_path / 'learning-service.db'}")
    database.create_schema()
    run_repository = run_repo_type(database.session_factory)
    performance_repository = performance_repo_type(database.session_factory)
    run_repository.create("run-1", {"topic": "AI safety", "brand_id": "brand-1"})
    reflection_agent = FakeReflectionAgent(
        LearningInsight(
            rationale="Concrete outcome-first hooks led to higher simulated engagement.",
            rules=[PlaybookRule(rule="Lead with a measurable outcome", platform="linkedin")],
        )
    )
    service = LearningService(run_repository, performance_repository, reflection_agent)
    try:
        result = service.record_feedback(
            "run-1",
            FeedbackInput(
                source="simulated",
                metrics=metrics_type(impressions=1000, likes=80, comments=10, shares=5),
                published_at=datetime(2026, 10, 13, tzinfo=UTC),
            ),
        )

        events = run_repository.list_events("run-1")
        assert result.source == "simulated"
        assert result.engagement_rate == pytest.approx(0.095)
        assert result.playbook_version == 1
        assert result.event_id == events[-1].id
        assert [event.event_type for event in events] == [
            "performance_recorded",
            "learning_proposed",
            "playbook_updated",
        ]
        assert events[0].payload["source"] == "simulated"
        assert performance_repository.get_active_playbook("brand-1").rules[0]["rule"] == (
            "Lead with a measurable outcome"
        )
    finally:
        database.close()


def test_failed_reflection_preserves_active_playbook_and_logs_failure(tmp_path: Path):
    try:
        from app.services.learning import FeedbackInput, LearningService, LearningServiceError
    except ModuleNotFoundError as error:
        pytest.fail(f"learning service implementation is missing: {error.name}")

    database_type, metrics_type, performance_repo_type, playbook_rule_type, run_repo_type = (
        load_learning_persistence()
    )
    database = database_type(f"sqlite:///{tmp_path / 'learning-failure.db'}")
    database.create_schema()
    run_repository = run_repo_type(database.session_factory)
    performance_repository = performance_repo_type(database.session_factory)
    run_repository.create("run-1", {"topic": "AI safety", "brand_id": "brand-1"})
    prior_feedback = performance_repository.record_feedback(
        "run-1",
        source="user_entered",
        metrics=metrics_type(impressions=100, likes=10),
        published_at=datetime(2026, 10, 12, tzinfo=UTC),
    )
    previous = performance_repository.save_playbook_version(
        "brand-1",
        [playbook_rule_type(rule="Use concrete examples")],
        [prior_feedback.id],
        "Prior active guidance.",
    )
    service = LearningService(
        run_repository,
        performance_repository,
        FakeReflectionAgent(failure=RuntimeError("private provider detail")),
    )
    try:
        with pytest.raises(LearningServiceError):
            service.record_feedback(
                "run-1",
                FeedbackInput(
                    source="simulated",
                    metrics=metrics_type(impressions=200, likes=20),
                    published_at=datetime(2026, 10, 13, tzinfo=UTC),
                ),
            )

        assert performance_repository.get_active_playbook("brand-1").id == previous.id
        assert [event.event_type for event in run_repository.list_events("run-1")] == [
            "performance_recorded",
            "learning_failed",
        ]
        assert "private provider detail" not in run_repository.list_events("run-1")[-1].reason
    finally:
        database.close()


def test_zero_impressions_records_feedback_and_skips_reflection(tmp_path: Path):
    from app.services.learning import FeedbackInput, LearningService

    database_type, metrics_type, performance_repo_type, _, run_repo_type = (
        load_learning_persistence()
    )
    database = database_type(f"sqlite:///{tmp_path / 'zero-impressions.db'}")
    database.create_schema()
    run_repository = run_repo_type(database.session_factory)
    performance_repository = performance_repo_type(database.session_factory)
    run_repository.create("run-1", {"topic": "AI safety", "brand_id": "brand-1"})
    reflection_agent = FakeReflectionAgent()
    service = LearningService(run_repository, performance_repository, reflection_agent)
    try:
        result = service.record_feedback(
            "run-1",
            FeedbackInput(
                source="simulated",
                metrics=metrics_type(impressions=0, likes=0),
                published_at=datetime(2026, 10, 13, tzinfo=UTC),
            ),
        )

        assert result.engagement_rate is None
        assert result.playbook_version is None
        assert reflection_agent.calls == []
        assert [event.event_type for event in run_repository.list_events("run-1")] == [
            "performance_recorded",
            "learning_skipped",
        ]
    finally:
        database.close()


def test_missing_mesh_key_during_reflection_is_sanitized_and_logged(tmp_path: Path):
    from app.llm.mesh_client import MissingMeshApiKey
    from app.services.learning import FeedbackInput, LearningService, LearningServiceError

    database_type, metrics_type, performance_repo_type, _, run_repo_type = (
        load_learning_persistence()
    )
    database = database_type(f"sqlite:///{tmp_path / 'missing-mesh-key.db'}")
    database.create_schema()
    run_repository = run_repo_type(database.session_factory)
    performance_repository = performance_repo_type(database.session_factory)
    run_repository.create("run-1", {"topic": "AI safety", "brand_id": "brand-1"})
    reflection_agent = FakeReflectionAgent(
        failure=MissingMeshApiKey("MESH_API_KEY must be configured")
    )
    service = LearningService(run_repository, performance_repository, reflection_agent)
    try:
        with pytest.raises(LearningServiceError, match="learning failed"):
            service.record_feedback(
                "run-1",
                FeedbackInput(
                    source="simulated",
                    metrics=metrics_type(impressions=100, likes=10),
                    published_at=datetime(2026, 10, 13, tzinfo=UTC),
                ),
            )

        assert len(performance_repository.list_feedback("brand-1")) == 1
        assert [event.event_type for event in run_repository.list_events("run-1")] == [
            "performance_recorded",
            "learning_failed",
        ]
        assert "MESH_API_KEY" not in run_repository.list_events("run-1")[-1].reason
    finally:
        database.close()


def test_feedback_playbook_update_is_loaded_by_next_generation(tmp_path: Path):
    from app.agents.contracts import (
        ContentBrief,
        DraftContent,
        EditorCritique,
        PlaybookRule,
        SafetyVerdict,
        StrategyDecision,
    )
    from app.config import PlatformAgentConfig
    from app.db import Database
    from app.models.performance import PerformanceMetrics
    from app.repositories.brand_memory import BrandMemoryRepository
    from app.repositories.performance import PerformanceRepository
    from app.repositories.runs import RunRepository
    from app.services.creation import CreationService
    from app.services.learning import FeedbackInput, LearningInsight, LearningService
    from app.workflows.content_graph import build_content_graph

    database = Database(f"sqlite:///{tmp_path / 'future-memory.db'}")
    database.create_schema()
    run_repository = RunRepository(database.session_factory)
    brand_repository = BrandMemoryRepository(database.session_factory)
    performance_repository = PerformanceRepository(database.session_factory)
    samples = brand_repository.add_samples(
        "brand-1",
        [
            {"body": "Short, direct lesson one."},
            {"body": "Short, direct lesson two."},
            {"body": "Short, direct lesson three."},
        ],
    )
    brand_repository.save_profile(
        "brand-1",
        {
            "persona": "Direct engineering teacher",
            "tone_attributes": ["concise"],
            "preferred_terms": [],
            "avoid_terms": [],
            "structure_guidelines": ["Lead with the outcome"],
        },
        [sample.id for sample in samples],
    )
    run_repository.create("metric-run", {"brand_id": "brand-1", "topic": "AI safety"})
    learning_service = LearningService(
        run_repository,
        performance_repository,
        FakeReflectionAgent(
            LearningInsight(
                rationale="The observed response supports an outcome-first hook.",
                rules=[PlaybookRule(rule="Lead with a measurable outcome", platform="linkedin")],
            )
        ),
    )

    class SafeAgent:
        def check(self, text, context):
            return SafetyVerdict(allowed=True, reason="safe", risk_categories=[])

    class Strategist:
        def __init__(self):
            self.memory = None

        def plan(self, brief, memory=None):
            self.memory = memory
            return StrategyDecision(
                angle="practical",
                audience="engineers",
                hook="A useful outcome",
                rationale="Specific advice is useful.",
            )

    class Writer:
        def __init__(self):
            self.memory = None

        def write(self, brief, strategy, critique=None, memory=None):
            self.memory = memory
            return DraftContent(text="A post based on current brand guidance.")

    class Editor:
        def review(self, brief, strategy, draft):
            return EditorCritique(
                score=8,
                approved=True,
                critique="Clear.",
                revision_instructions=[],
            )

    agents = {
        "safety": SafeAgent(),
        "strategist": Strategist(),
        "writer": Writer(),
        "editor": Editor(),
    }
    config = PlatformAgentConfig(
        platform="linkedin",
        agent_id="editor",
        model="openai/gpt-oss-120b",
        system_prompt_id="linkedin.editor.system",
        instruction_prompt_id="linkedin.editor.instructions",
    )
    graph = build_content_graph(
        agents,
        run_repository,
        config,
        brand_memory_repository=brand_repository,
        performance_repository=performance_repository,
    )
    creation_service = CreationService(graph, run_repository)
    try:
        feedback_result = learning_service.record_feedback(
            "metric-run",
            FeedbackInput(
                source="user_entered",
                metrics=PerformanceMetrics(impressions=1000, likes=100, comments=20),
                published_at=datetime(2026, 10, 13, tzinfo=UTC),
            ),
        )
        next_run = creation_service.create_run(
            ContentBrief(
                topic="AI safety",
                platform="linkedin",
                audience="engineers",
                goal="educate",
                brand_id="brand-1",
            )
        )

        assert feedback_result.playbook_version == 1
        assert agents["strategist"].memory.playbook_version == 1
        assert agents["writer"].memory.playbook_rules[0].rule == "Lead with a measurable outcome"
        assert next_run.workflow_state["brand_memory_context"]["playbook_version"] == 1
    finally:
        database.close()


def test_reflection_agent_returns_typed_mesh_insight(tmp_path: Path):
    try:
        from app.agents.reflection import LearningInsight, ReflectionAgent
    except ModuleNotFoundError as error:
        pytest.fail(f"reflection agent implementation is missing: {error.name}")
    from app.config import PlatformAgentConfig
    from app.prompts.loader import PromptLoader

    prompt_dir = tmp_path / "linkedin" / "reflection"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "system.md").write_text("Reflection role.", encoding="utf-8")
    (prompt_dir / "instructions.md").write_text(
        "Feedback: {feedback_json}\nPrevious playbook: {prior_playbook_json}",
        encoding="utf-8",
    )
    config = PlatformAgentConfig(
        platform="linkedin",
        agent_id="reflection",
        model="openai/gpt-oss-120b",
        system_prompt_id="linkedin.reflection.system",
        instruction_prompt_id="linkedin.reflection.instructions",
    )

    class FakeStructuredMesh:
        def __init__(self):
            self.requests = []

        def resolve_model(self, configured_model):
            return configured_model

        def complete_structured(self, request, response_model, max_retries):
            self.requests.append(request)
            return response_model.model_validate_json(
                '{"rationale":"Outcome-first hooks have evidence.",'
                '"rules":[{"rule":"Lead with an outcome","platform":"linkedin",'
                '"category":"hook"}]}'
            )

    mesh_client = FakeStructuredMesh()
    runtime_type = load_learning_runtime()
    runtime = runtime_type(mesh_client, PromptLoader(tmp_path))
    agent = ReflectionAgent(runtime, config)
    feedback = SimpleNamespace(
        id="feedback-1",
        brand_id="brand-1",
        source="simulated",
        metrics={"impressions": 100, "likes": 15},
        engagement_rate=0.15,
        published_at=datetime(2026, 10, 13, tzinfo=UTC),
    )

    insight = agent.analyze(feedback, prior_playbook=None)

    assert isinstance(insight, LearningInsight)
    assert insight.rules[0].rule == "Lead with an outcome"
    assert mesh_client.requests[0].messages[0].content == "Reflection role."
    assert "simulated" in mesh_client.requests[0].messages[1].content


def load_learning_runtime():
    from app.agents.runtime import AgentRuntime

    return AgentRuntime
