flowchart TD
    User["Creator"] --> UI["CreatorOS UI"]
    UI --> Auth["FastAPI<br/>Authentication, RBAC, tenant access"]
    Auth --> API["Studio API"]

    subgraph Config["Configuration and prompts"]
        PlatformConfig["platform_configs<br/>model, token limits, tools, timeout,<br/>retry policy, prompt IDs"]
        PromptFiles["agent_prompts/*.md<br/>system and instruction templates"]
        Secrets["Secret provider<br/>Local: ignored environment/local secret store<br/>Cloud: Vault"]
    end

    API --> Orchestrator["LangGraph orchestrator"]
    PlatformConfig --> Orchestrator
    PromptFiles --> Orchestrator
    Secrets --> LLMGateway
    Secrets --> SocialMCP

    subgraph Creation["Content creation workflow"]
        Orchestrator --> IntakeSafety["Safety checks<br/>deterministic checks + moderation"]
        IntakeSafety -->|Pass| Strategist["Strategist Agent<br/>platform, format, angle, audience"]
        IntakeSafety -->|Block| SafetyResult["Return safe rejection to user"]

        Strategist --> Writer["Writer Agent<br/>post, article, carousel slides,<br/>or video script"]
        Writer --> Editor["Editor Agent<br/>format and brand rubric"]
        Editor -->|Revise, capped attempts| Writer
        Editor -->|Pass| Visual["Visual Agent<br/>asset prompts and media plan"]
        Visual --> HumanReview["Human review<br/>approve or provide feedback"]

        HumanReview -->|Reject or request changes| FeedbackSafety["Validate review feedback"]
        FeedbackSafety -->|Safe minor edit| Writer
        FeedbackSafety -->|Safe strategy change| Strategist
        FeedbackSafety -->|Unsafe| SafetyResult
    end

    subgraph Models["LLM and agent tools"]
        LLMGateway["LLM Gateway Adapter<br/>all model calls use Mesh API"]
        MCPTools["Scoped MCP / function tools<br/>brand profile and winning patterns<br/>rubric scoring and audit logging<br/>image generation and platform operations"]
        MediaTools["Deterministic media tools<br/>image sizing, carousel/PDF rendering,<br/>video asset packaging"]
    end

    Strategist --> LLMGateway
    Writer --> LLMGateway
    Editor --> LLMGateway
    IntakeSafety --> LLMGateway
    FeedbackSafety --> LLMGateway
    Visual --> LLMGateway
    Strategist --> MCPTools
    Editor --> MCPTools
    Visual --> MCPTools
    Visual --> MediaTools

    HumanReview -->|Approve and schedule| Outbox["PostgreSQL post/outbox<br/>approved content + due time"]
    Outbox --> Broker["Redis job broker"]
    Broker --> Worker["Async worker<br/>retries, backoff, idempotency, dead-letter handling"]
    Worker --> SocialMCP["LinkedIn / Meta MCP adapters"]
    SocialMCP --> PlatformAPIs["LinkedIn and Meta APIs<br/>scoped OAuth permissions"]
    PlatformAPIs --> Published["Published post"]

    subgraph Memory["Memory and durable data"]
        MemoryAPI["Memory service / MCP tools"]
        Working["Working memory<br/>Redis, TTL per active run"]
        Episodic["Episodic memory<br/>brief, drafts, edits, outcome"]
        Semantic["Semantic memory<br/>brand profile, samples, embeddings"]
        Procedural["Procedural memory<br/>learned platform and content playbook"]
        Prospective["Prospective memory<br/>scheduled actions and follow-ups"]
        Perceptual["Perceptual memory<br/>media descriptions and asset metadata"]
        Postgres["PostgreSQL + pgvector"]
        ObjectStore["Media object storage"]
        Audit["Append-only decision trail<br/>agent decisions, tool calls, score changes"]
    end

    Orchestrator --> Working
    Strategist --> MemoryAPI
    Reflection --> MemoryAPI
    MemoryAPI --> Episodic
    MemoryAPI --> Semantic
    MemoryAPI --> Procedural
    MemoryAPI --> Prospective
    MemoryAPI --> Perceptual
    Episodic --> Postgres
    Semantic --> Postgres
    Procedural --> Postgres
    Prospective --> Postgres
    Perceptual --> Postgres
    Outbox --> Postgres
    Visual --> ObjectStore
    MCPTools --> Audit
    Audit --> Postgres

    Published --> MetricsSchedule["Configurable metric polling<br/>early, 24h, later checkpoints"]
    MetricsSchedule --> Broker
    Worker -->|fetch metrics tool| SocialMCP
    SocialMCP -->|metrics| Reflection["Reflection Agent"]
    Reflection --> LLMGateway
    Reflection -->|record episode and update playbook| MemoryAPI
    Reflection --> Audit

    subgraph Operations["Operations and evaluation"]
        StructuredLogs["Structured logs and traces<br/>Logfire"]
        LLMTraces["LLM tracing<br/>LangSmith"]
        Eval["Offline DeepEval benchmark<br/>editor before/after;<br/>cold-start vs learned memory"]
    end

    Orchestrator -.-> StructuredLogs
    Worker -.-> StructuredLogs
    LLMGateway -.-> LLMTraces
    Eval --> LLMGateway
    Eval -.-> StructuredLogs