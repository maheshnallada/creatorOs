# CreatorOS

CreatorOS is a multi-agent content studio being built for the Agent Arena hackathon. The planned workflow is strategist -> writer -> editor, followed by human review, scheduled delivery, and performance feedback that informs future content.

## Phase 1 status

The current foundation provides validated runtime and platform-agent configuration, Markdown prompt loading, a FastAPI health endpoint, a bounded retry policy, and an OpenAI-compatible Mesh API adapter. Agent orchestration, persistence, media generation, authentication, and social publishing are later phases and are not implemented yet.

Every model request must go through `https://api.meshapi.ai/v1` using `MESH_API_KEY`. The application does not fall back to a direct model-provider endpoint.

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

## Test

```bash
python -m pytest
```

Tests use an in-memory HTTP transport for Mesh calls and do not require credentials or make network requests.