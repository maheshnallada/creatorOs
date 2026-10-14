FROM ghcr.io/astral-sh/uv:0.11.28 AS uv

FROM python:3.11-slim AS builder

COPY --from=uv /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /build

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY app ./app
COPY eval ./eval
RUN uv sync --locked --no-dev --no-editable

FROM python:3.11-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd --gid 10001 creatoros \
    && useradd --uid 10001 --gid creatoros --create-home --shell /usr/sbin/nologin creatoros

COPY --from=builder --chown=creatoros:creatoros /build/.venv /app/.venv
COPY --chown=creatoros:creatoros app ./app
COPY --chown=creatoros:creatoros eval ./eval
COPY --chown=creatoros:creatoros agent_prompts ./agent_prompts
COPY --chown=creatoros:creatoros platform_configs ./platform_configs

USER creatoros:creatoros

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]