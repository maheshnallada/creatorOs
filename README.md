# CreatorOS

CreatorOS is a multi-agent content studio being built for the Agent Arena hackathon. Its current workflow screens a brief, selects a strategy, drafts and critiques a LinkedIn text post, and revises it when the editor requests changes.

## Current implementation

Four distinct LangGraph nodes use their own prompts and Pydantic output contracts: safety, strategist, writer, and editor. Instructor runs in JSON mode through a provider-configurable OpenAI-compatible client and applies the per-agent validation retry budget; transient transport retries remain separate. Structured-call failures and Instructor retry logs are sanitized. The editor-to-writer loop is bounded; the API returns a run in `needs_review` even if it reaches the revision limit without passing the editor threshold. A separate Reflection Agent updates the versioned content playbook from labeled feedback. CreatorOS does not publish content.

Runs, brand samples/profiles, performance feedback, playbook versions, and append-only decision events are stored in local SQLite. The API exposes run creation/retrieval, ordered event retrieval, brand profile ingestion/retrieval, and feedback submission. Three or more distinct user-provided posts create a versioned voice profile; active profile and playbook versions are injected into subsequent runs. Feedback must be labeled `user_entered` or `simulated`; simulated values are never presented as platform-fetched. The DeepEval benchmark and recorded measurements remain in progress. LinkedIn agent settings live under `platform_configs/`; Markdown prompts live under `agent_prompts/`.

`LLM_PROVIDER` selects `mesh` (default) or `groq`. Mesh requests use `MESH_API_KEY`; Groq requests use `GROQ_API_KEY` and `GROQ_MODEL`. Groq is restricted to local development and tests. Production and hackathon runs must use Mesh, and provider failures never silently fall back to another provider.

This phase has no human approval/rejection endpoint, user authentication, platform OAuth, media generation, or social publishing. The API is intended for local development only; do not expose it to a network until authentication and authorization are implemented.

## Setup

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

Set the API key for the selected provider in `.env` before making a live model request. To use Groq locally, set `LLM_PROVIDER=groq`, `GROQ_API_KEY`, and a valid `GROQ_MODEL`; leave `APP_ENV` as `local` or `test`. Use Mesh for production and hackathon runs. Never commit `.env`, `local-secrets.json`, or `secrets.json`. Cloud deployments should inject secrets through a secret manager/environment, not a checked-in file.

## Run

```bash
uvicorn app.main:app --reload
```

The health endpoint is available at `http://127.0.0.1:8000/health`.

## Run with Docker Compose

Requires Docker Engine and Docker Compose. The native app and test suite continue to use SQLite; Compose runs the API with PostgreSQL 16 and stores database files in the `creatoros_pgdata` named volume.

```bash
cp .env.example .env
```

Set the selected provider's key in `.env` before making live model requests. The template defaults to Mesh. For local Groq use, set `LLM_PROVIDER=groq`, `GROQ_API_KEY`, and `GROQ_MODEL`; do not use Groq for production or hackathon runs. The example PostgreSQL credentials are for local development only. Compose does not publish the database port; it binds the unauthenticated API to `127.0.0.1` on `CREATOROS_PORT` (default `8000`). Do not expose this API to a network until authentication and authorization are implemented.

```bash
docker compose up --build -d
docker compose ps
curl http://127.0.0.1:${CREATOROS_PORT:-8000}/health
docker compose exec postgres pg_isready
docker compose logs -f app
```

Stop the containers with `docker compose down`. The named database volume remains, so data survives container restarts and normal shutdown. `docker compose down -v` deletes the database volume and should only be used for an intentional local reset.

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

## Brand voice samples

Submit three or more distinct post samples as text. CreatorOS analyzes only the supplied content; it does not scrape connected social accounts.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/brands/my-brand/samples \
	-H 'Content-Type: application/json' \
	-d '{"samples":[{"body":"First historical post text"},{"body":"Second historical post text"},{"body":"Third historical post text"}]}'
```

The response contains the extracted profile, profile version, and source sample IDs. Retrieve the active profile with:

```bash
curl http://127.0.0.1:8000/api/v1/brands/my-brand/profile
```

## Record performance feedback

Feedback can be entered by a user or explicitly marked simulated. `published_at` is required; zero impressions are recorded without calculating an engagement rate or triggering a playbook update.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/runs/<run_id>/feedback \
	-H 'Content-Type: application/json' \
	-d '{"source":"simulated","published_at":"2026-10-14T12:00:00Z","metrics":{"impressions":1000,"likes":85,"comments":12,"shares":9,"clicks":30,"saves":18}}'
```

The response includes the feedback source, derived engagement rate, audit event ID, and resulting playbook version. A subsequent run for the same brand loads that playbook.

## Evaluation

The optional DeepEval harness compares saved `writer_initial`/`editor_final` drafts and `cold_start`/`learned` drafts. Populate a case file from actual CreatorOS runs using the schema documented in `eval/cases/README.md`, then run:

```bash
python -m pip install -e '.[eval]'
python -m eval.benchmark --input eval/cases/my-run-pairs.json --output eval/results/phase3.json
```

The judge uses the selected provider; in Groq mode it defaults to `GROQ_MODEL`, and `--model` overrides that selection. DeepEval telemetry is disabled. No benchmark scores are recorded yet; valid credentials for the selected provider and actual run outputs are required. Do not treat unit-test fake scores as results.

## Test

```bash
python -m pytest
```

Tests use fake Mesh responses and temporary SQLite databases; they do not require credentials or make network requests.