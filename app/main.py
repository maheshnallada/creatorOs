from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.agents.brand_voice import BrandVoiceAgent
from app.agents.editor import EditorAgent
from app.agents.reflection import ReflectionAgent
from app.agents.runtime import AgentRuntime
from app.agents.safety import SafetyAgent
from app.agents.strategist import StrategistAgent
from app.agents.writer import WriterAgent
from app.config import PlatformAgentConfig, Settings
from app.db import Database
from app.llm.mesh_client import LLMClient
from app.prompts.loader import PromptLoader
from app.repositories.brand_memory import BrandMemoryRepository
from app.repositories.performance import PerformanceRepository
from app.repositories.runs import RunRepository
from app.services.creation import CreationService
from app.services.learning import LearningService
from app.workflows.content_graph import build_content_graph

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_agent_configs(platform: str = "linkedin"):
    configs = {}
    config_dir = PROJECT_ROOT / "platform_configs"
    for agent_id in (
        "safety",
        "strategist",
        "writer",
        "editor",
        "brand_voice",
        "reflection",
    ):
        config_path = config_dir / f"{platform}.{agent_id}.json"
        configs[agent_id] = PlatformAgentConfig.model_validate_json(
            config_path.read_text(encoding="utf-8")
        )
    return configs


def _create_default_components(settings: Settings):
    database = Database(settings.database_url)
    database.create_schema()
    repository = RunRepository(database.session_factory)
    brand_memory_repository = BrandMemoryRepository(database.session_factory)
    performance_repository = PerformanceRepository(database.session_factory)
    llm_client = LLMClient(settings)
    prompt_loader = PromptLoader(PROJECT_ROOT / "agent_prompts")
    runtime = AgentRuntime(llm_client, prompt_loader)
    configs = _load_agent_configs()
    agents = {
        "safety": SafetyAgent(runtime, configs["safety"]),
        "strategist": StrategistAgent(runtime, configs["strategist"]),
        "writer": WriterAgent(runtime, configs["writer"]),
        "editor": EditorAgent(runtime, configs["editor"]),
    }
    brand_voice_agent = BrandVoiceAgent(runtime, configs["brand_voice"])
    reflection_agent = ReflectionAgent(runtime, configs["reflection"])
    graph = build_content_graph(
        agents,
        repository,
        configs["editor"],
        brand_memory_repository=brand_memory_repository,
        performance_repository=performance_repository,
    )
    service = CreationService(graph, repository)
    learning_service = LearningService(repository, performance_repository, reflection_agent)
    return (
        database,
        repository,
        brand_memory_repository,
        performance_repository,
        brand_voice_agent,
        llm_client,
        service,
        learning_service,
    )


def create_app(
    settings: Settings | None = None,
    *,
    creation_service: CreationService | None = None,
    run_repository: RunRepository | None = None,
    database: Database | None = None,
    mesh_client: LLMClient | None = None,
    brand_memory_repository: BrandMemoryRepository | None = None,
    performance_repository: PerformanceRepository | None = None,
    brand_voice_agent: BrandVoiceAgent | None = None,
    learning_service: LearningService | None = None,
) -> FastAPI:
    settings = settings or Settings()
    owns_components = creation_service is None
    if owns_components:
        (
            database,
            run_repository,
            brand_memory_repository,
            performance_repository,
            brand_voice_agent,
            mesh_client,
            creation_service,
            learning_service,
        ) = _create_default_components(settings)
    elif run_repository is None:
        raise ValueError("run_repository must be supplied with an injected creation_service")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        if owns_components:
            if mesh_client is not None:
                mesh_client.close()
            if database is not None:
                database.close()

    app = FastAPI(title="CreatorOS", version="0.2.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.database = database
    app.state.mesh_client = mesh_client
    app.state.llm_client = mesh_client
    app.state.run_repository = run_repository
    app.state.creation_service = creation_service
    app.state.brand_memory_repository = brand_memory_repository
    app.state.performance_repository = performance_repository
    app.state.brand_voice_agent = brand_voice_agent
    app.state.learning_service = learning_service

    @app.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    from app.api.brands import router as brands_router
    from app.api.feedback import router as feedback_router
    from app.api.runs import router as runs_router

    app.include_router(runs_router)
    app.include_router(brands_router)
    app.include_router(feedback_router)
    return app


app = create_app()
