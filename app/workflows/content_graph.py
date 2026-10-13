from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.contracts import ContentBrief, DraftContent, EditorCritique, StrategyDecision
from app.config import PlatformAgentConfig
from app.repositories.runs import RunRepository
from app.workflows.state import ContentState


def build_content_graph(
    agents: dict[str, Any],
    repository: RunRepository,
    config: PlatformAgentConfig,
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

    def strategist_node(state: ContentState) -> ContentState:
        brief = ContentBrief.model_validate(state["brief"])
        strategy = agents["strategist"].plan(brief)
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
        critique_data = state.get("latest_critique")
        critique = EditorCritique.model_validate(critique_data) if critique_data else None
        draft = agents["writer"].write(brief, strategy, critique)
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
        return "strategist" if state["safety_verdict"]["allowed"] else "finish"

    def route_after_editor(state: ContentState) -> str:
        return "writer" if state.get("revision_requested", False) else "finish"

    graph = StateGraph(ContentState)
    graph.add_node("safety", safety_node)
    graph.add_node("strategist", strategist_node)
    graph.add_node("writer", writer_node)
    graph.add_node("editor", editor_node)
    graph.add_node("finish", finish_node)
    graph.add_edge(START, "safety")
    graph.add_conditional_edges(
        "safety", route_after_safety, {"strategist": "strategist", "finish": "finish"}
    )
    graph.add_edge("strategist", "writer")
    graph.add_edge("writer", "editor")
    graph.add_conditional_edges(
        "editor", route_after_editor, {"writer": "writer", "finish": "finish"}
    )
    graph.add_edge("finish", END)
    return graph.compile()
