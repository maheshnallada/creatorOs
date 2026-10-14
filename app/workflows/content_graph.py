from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.contracts import (
    BrandMemoryContext,
    BrandVoiceProfile,
    ContentBrief,
    DraftContent,
    EditorCritique,
    PlaybookRule,
    StrategyDecision,
)
from app.config import PlatformAgentConfig
from app.repositories.brand_memory import BrandMemoryRepository
from app.repositories.performance import PerformanceRepository
from app.repositories.runs import RunRepository
from app.workflows.state import ContentState


def build_content_graph(
    agents: dict[str, Any],
    repository: RunRepository,
    config: PlatformAgentConfig,
    *,
    brand_memory_repository: BrandMemoryRepository | None = None,
    performance_repository: PerformanceRepository | None = None,
) -> CompiledStateGraph:
    def persist(
        state: ContentState,
        event_type: str,
        agent_id: str,
        reason: str,
        payload: dict[str, Any],
    ) -> ContentState:
        repository.append_event(state["run_id"], event_type, agent_id, reason, payload)
        repository.save_state(state["run_id"], dict(state), status=state.get("status", "running"))
        return state

    def safety_node(state: ContentState) -> ContentState:
        brief = ContentBrief.model_validate(state["brief"])
        verdict = agents["safety"].check(brief.model_dump_json(), context="initial brief")
        updated = {**state, "safety_verdict": verdict.model_dump(), "status": "running"}
        return persist(
            updated,
            "safety_checked",
            "safety",
            verdict.reason,
            {"allowed": verdict.allowed, "risk_categories": verdict.risk_categories},
        )

    def load_memory_node(state: ContentState) -> ContentState:
        brief = ContentBrief.model_validate(state["brief"])
        profile_version = None
        playbook_version = None
        profile = None
        playbook_rules = []
        if brief.brand_id is not None:
            if brand_memory_repository is None or performance_repository is None:
                raise RuntimeError("brand memory repositories are not configured")
            active_profile = brand_memory_repository.get_active_profile(brief.brand_id)
            active_playbook = performance_repository.get_active_playbook(brief.brand_id)
            if active_profile is not None:
                profile = BrandVoiceProfile.model_validate(active_profile.profile)
                profile_version = active_profile.version
            if active_playbook is not None:
                playbook_rules = [
                    PlaybookRule.model_validate(rule) for rule in active_playbook.rules
                ]
                playbook_version = active_playbook.version

        memory_context = BrandMemoryContext(
            profile=profile,
            profile_version=profile_version,
            playbook_rules=playbook_rules,
            playbook_version=playbook_version,
        )
        updated = {
            **state,
            "brand_memory_context": memory_context.model_dump(mode="json"),
            "status": "running",
        }
        return persist(
            updated,
            "brand_memory_loaded",
            "memory",
            "Loaded active brand voice and playbook versions for this run.",
            {
                "profile_version": profile_version,
                "playbook_version": playbook_version,
            },
        )

    def strategist_node(state: ContentState) -> ContentState:
        brief = ContentBrief.model_validate(state["brief"])
        memory = BrandMemoryContext.model_validate(state.get("brand_memory_context", {}))
        strategy = agents["strategist"].plan(brief, memory=memory)
        updated = {**state, "strategy": strategy.model_dump(), "status": "running"}
        return persist(
            updated,
            "strategy_selected",
            "strategist",
            strategy.rationale,
            strategy.model_dump(),
        )

    def writer_node(state: ContentState) -> ContentState:
        brief = ContentBrief.model_validate(state["brief"])
        strategy = StrategyDecision.model_validate(state["strategy"])
        memory = BrandMemoryContext.model_validate(state.get("brand_memory_context", {}))
        critique_data = state.get("latest_critique")
        critique = EditorCritique.model_validate(critique_data) if critique_data else None
        draft = agents["writer"].write(brief, strategy, critique, memory=memory)
        draft_data = draft.model_dump()
        history = [*state.get("draft_history", []), draft_data]
        is_revision = state.get("revision_count", 0) > 0
        updated = {
            **state,
            "draft": draft_data,
            "draft_history": history,
            "status": "running",
        }
        event_type = "draft_revised" if is_revision else "draft_created"
        reason = (
            "Writer applied the editor's revision instructions."
            if is_revision
            else "Writer created the initial draft from the selected strategy."
        )
        return persist(updated, event_type, "writer", reason, draft_data)

    def editor_node(state: ContentState) -> ContentState:
        brief = ContentBrief.model_validate(state["brief"])
        strategy = StrategyDecision.model_validate(state["strategy"])
        draft = DraftContent.model_validate(state["draft"])
        critique = agents["editor"].review(brief, strategy, draft)
        passed = critique.approved and critique.score >= config.editor_quality_threshold
        revision_count = state.get("revision_count", 0)
        revision_requested = not passed and revision_count < config.max_editor_revisions
        updated = {
            **state,
            "latest_critique": critique.model_dump(),
            "editor_passed": passed,
            "revision_requested": revision_requested,
            "revision_count": revision_count + 1 if revision_requested else revision_count,
            "status": "running",
        }
        return persist(
            updated,
            "editor_critiqued",
            "editor",
            critique.critique,
            {
                "score": critique.score,
                "approved": critique.approved,
                "passed_threshold": passed,
                "revision_instructions": critique.revision_instructions,
            },
        )

    def finish_node(state: ContentState) -> ContentState:
        verdict = state["safety_verdict"]
        if not verdict["allowed"]:
            updated = {**state, "status": "blocked"}
            return persist(
                updated,
                "run_blocked",
                "safety",
                verdict["reason"],
                {"risk_categories": verdict["risk_categories"]},
            )

        updated = {**state, "status": "needs_review"}
        critique = state.get("latest_critique", {})
        return persist(
            updated,
            "run_needs_review",
            "workflow",
            "Draft is ready for human review; this phase does not publish content.",
            {
                "editor_passed": state.get("editor_passed", False),
                "editor_score": critique.get("score"),
                "revision_count": state.get("revision_count", 0),
            },
        )

    def route_after_safety(state: ContentState) -> str:
        return "load_memory" if state["safety_verdict"]["allowed"] else "finish"

    def route_after_editor(state: ContentState) -> str:
        return "writer" if state.get("revision_requested", False) else "finish"

    graph = StateGraph(ContentState)
    graph.add_node("safety", safety_node)
    graph.add_node("load_memory", load_memory_node)
    graph.add_node("strategist", strategist_node)
    graph.add_node("writer", writer_node)
    graph.add_node("editor", editor_node)
    graph.add_node("finish", finish_node)
    graph.add_edge(START, "safety")
    graph.add_conditional_edges(
        "safety", route_after_safety, {"load_memory": "load_memory", "finish": "finish"}
    )
    graph.add_edge("load_memory", "strategist")
    graph.add_edge("strategist", "writer")
    graph.add_edge("writer", "editor")
    graph.add_conditional_edges(
        "editor", route_after_editor, {"writer": "writer", "finish": "finish"}
    )
    graph.add_edge("finish", END)
    return graph.compile()
