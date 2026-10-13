import pytest
from fastapi.testclient import TestClient


def load_create_app():
    try:
        from app.main import create_app
    except ModuleNotFoundError as error:
        pytest.fail(f"FastAPI app implementation is missing: {error.name}")
    return create_app


def test_health_endpoint_works_without_mesh_credentials(monkeypatch):
    monkeypatch.delenv("MESH_API_KEY", raising=False)
    create_app = load_create_app()

    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_app_uses_injected_settings():
    from app.config import Settings

    create_app = load_create_app()
    app = create_app(Settings(app_env="test", _env_file=None))

    assert app.state.settings.app_env == "test"
