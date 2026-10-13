# CreatorOS

CreatorOS is a multi-agent content studio being built for the Agent Arena hackathon. Its current workflow screens a brief, selects a strategy, drafts and critiques a LinkedIn text post, and revises it when the editor requests changes.

## Current implementation

Four distinct LangGraph nodes use their own prompts and Pydantic output contracts: safety, strategist, writer, and editor. Instructor runs in JSON mode through the Mesh-backed OpenAI client and applies the per-agent validation retry budget; transient transport retries remain separate. Structured-call failures and Instructor retry logs are sanitized. The editor-to-writer loop is bounded; the API returns a run in `needs_review` even if it reaches the revision limit without passing the editor threshold. It never publishes content.

Runs and append-only decision events are stored in local SQLite. The API exposes run creation, run retrieval, and ordered event retrieval. LinkedIn agent settings live under `platform_configs/`; Markdown prompts live under `agent_prompts/`.

Every model request must go through `https://api.meshapi.ai/v1` using `MESH_API_KEY`. The application does not fall back to a direct model-provider endpoint.

This phase has no human approval/rejection endpoint, user authentication, platform OAuth, brand-performance learning, media generation, or social publishing. The API is intended for local development only; do not expose it to a network until authentication and authorization are implemented.

## Setup

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

Set `MESH_API_KEY` in the local `.env` file before making a live model request. Never commit `.env`, `local-secrets.json`, or `secrets.json`. Cloud deployments should inject secrets through a secret manager/environment, not a checked-in file.

## Run

```bash
uvicorn app.main:app --reload
```

The health endpoint is available at `http://127.0.0.1:8000/health`.

## Create a run

```bash
curl -X POST http://127.0.0.1:8000/api/v1/runs \
	-H 'Content-Type: application/json' \
	-d '{"topic":"AI safety","platform":"linkedin","audience":"engineers","goal":"educate"}'
```

The response includes `run_id`, `status`, strategy, current draft, editor score, and revision count. A completed generation has status `needs_review`; review actions and scheduling are later phases. Retrieve its state and decision trail with:

```bash
curl http://127.0.0.1:8000/api/v1/runs/<run_id>
curl http://127.0.0.1:8000/api/v1/runs/<run_id>/events
```

Unknown run IDs return `404`. A failed generation returns a sanitized `502` response with a run ID and failure code; provider details are not returned.

## Test

```bash
python -m pytest
```

Tests use fake Mesh responses and temporary SQLite databases; they do not require credentials or make network requests.