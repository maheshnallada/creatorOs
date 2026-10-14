from pathlib import Path

from app.agents.contracts import ContentBrief, StrategyDecision


class MemoryAwareMeshClient:
    def __init__(self):
        self.requests = []

    def resolve_model(self, configured_model):
        return configured_model

    def complete_structured(self, request, response_model, max_retries):
        prompt = request.messages[-1].content
        self.requests.append(request)
        if response_model is StrategyDecision:
            if "Concise builder" in prompt:
                result = {
                    "angle": "concise lesson",
                    "audience": "engineers",
                    "hook": "Short beats vague",
                    "rationale": "The profile favors concise examples.",
                }
            else:
                result = {
                    "angle": "storytelling lesson",
                    "audience": "engineers",
                    "hook": "A story makes safety stick",
                    "rationale": "The profile favors narrative teaching.",
                }
        elif "Concise builder" in prompt:
            result = {"text": "Short, concrete lesson."}
        else:
            result = {"text": "A short story about a team learning from failure."}
        return response_model.model_validate(result)


def test_distinct_brand_profiles_change_strategy_and_writer_outputs(tmp_path: Path):
    from app.agents.contracts import BrandMemoryContext, BrandVoiceProfile, PlaybookRule
    from app.agents.runtime import AgentRuntime
    from app.agents.strategist import StrategistAgent
    from app.agents.writer import WriterAgent
    from app.config import PlatformAgentConfig
    from app.prompts.loader import PromptLoader

    prompt_root = tmp_path / "prompts"
    prompt_text = {
        "strategist": (
            "Strategist role.",
            "Brief JSON: {brief_json}\nBrand memory JSON: {brand_memory_json}",
        ),
        "writer": (
            "Writer role.",
            (
                "Brief JSON: {brief_json}\nStrategy JSON: {strategy_json}\n"
                "Critique JSON: {critique_json}\nBrand memory JSON: {brand_memory_json}"
            ),
        ),
    }
    configs = {}
    for agent_id, (system_prompt, instruction_prompt) in prompt_text.items():
        prompt_dir = prompt_root / "linkedin" / agent_id
        prompt_dir.mkdir(parents=True)
        (prompt_dir / "system.md").write_text(system_prompt, encoding="utf-8")
        (prompt_dir / "instructions.md").write_text(instruction_prompt, encoding="utf-8")
        configs[agent_id] = PlatformAgentConfig(
            platform="linkedin",
            agent_id=agent_id,
            model="openai/gpt-oss-120b",
            system_prompt_id=f"linkedin.{agent_id}.system",
            instruction_prompt_id=f"linkedin.{agent_id}.instructions",
        )

    mesh_client = MemoryAwareMeshClient()
    runtime = AgentRuntime(mesh_client, PromptLoader(prompt_root))
    strategist = StrategistAgent(runtime, configs["strategist"])
    writer = WriterAgent(runtime, configs["writer"])
    brief = ContentBrief(
        topic="AI safety",
        platform="linkedin",
        audience="engineers",
        goal="educate",
        brand_id="brand-1",
    )
    concise_memory = BrandMemoryContext(
        profile=BrandVoiceProfile(
            persona="Concise builder",
            tone_attributes=["concise"],
            preferred_terms=["concrete"],
            avoid_terms=[],
            structure_guidelines=["Use short sentences"],
        ),
        profile_version=1,
        playbook_rules=[PlaybookRule(rule="Lead with the result", platform="linkedin")],
        playbook_version=2,
    )
    narrative_memory = BrandMemoryContext(
        profile=BrandVoiceProfile(
            persona="Narrative teacher",
            tone_attributes=["storytelling"],
            preferred_terms=["lesson"],
            avoid_terms=[],
            structure_guidelines=["Use a short story arc"],
        ),
        profile_version=4,
        playbook_rules=[PlaybookRule(rule="Use a small team story", platform="linkedin")],
        playbook_version=3,
    )

    concise_strategy = strategist.plan(brief, memory=concise_memory)
    concise_draft = writer.write(brief, concise_strategy, memory=concise_memory)
    narrative_strategy = strategist.plan(brief, memory=narrative_memory)
    narrative_draft = writer.write(brief, narrative_strategy, memory=narrative_memory)

    assert concise_strategy.angle != narrative_strategy.angle
    assert concise_draft.text != narrative_draft.text
    assert "Concise builder" in mesh_client.requests[0].messages[-1].content
    assert "Lead with the result" in mesh_client.requests[0].messages[-1].content
    assert "Narrative teacher" in mesh_client.requests[2].messages[-1].content
    assert "Use a small team story" in mesh_client.requests[2].messages[-1].content
    assert "Concise builder" not in mesh_client.requests[0].messages[0].content


def test_cold_start_context_has_no_profile_or_playbook():
    from app.agents.contracts import BrandMemoryContext

    context = BrandMemoryContext()

    assert context.profile is None
    assert context.profile_version is None
    assert context.playbook_rules == []
    assert context.playbook_version is None
