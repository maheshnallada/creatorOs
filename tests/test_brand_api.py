from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def load_create_app():
    try:
        from app.main import create_app
    except ModuleNotFoundError as error:
        pytest.fail(f"brand API implementation is missing: {error.name}")
    return create_app


class FakeBrandVoiceAgent:
    def __init__(self, profile=None, failure=None):
        self.profile = profile
        self.failure = failure
        self.calls = []

    def build_profile(self, brand_id, samples):
        self.calls.append((brand_id, samples))
        if self.failure is not None:
            raise self.failure
        return self.profile


def sample_batch(prefix="sample"):
    return {
        "samples": [
            {"body": f"{prefix} one: I build useful systems.", "source_label": "post one"},
            {"body": f"{prefix} two: Small teams need clear feedback.", "source_label": "post two"},
            {"body": f"{prefix} three: Measure what changes.", "source_label": "post three"},
        ]
    }


def make_client(tmp_path: Path, agent, *, raise_server_exceptions=True):
    from app.config import Settings
    from app.db import Database
    from app.repositories.brand_memory import BrandMemoryRepository

    database_url = f"sqlite:///{tmp_path / 'brand-api.db'}"
    database = Database(database_url)
    database.create_schema()
    repository = BrandMemoryRepository(database.session_factory)

    class UnusedCreationService:
        def create_run(self, brief):
            raise AssertionError("brand API tests must not start content runs")

    app = load_create_app()(
        Settings(app_env="test", database_url=database_url, _env_file=None),
        creation_service=UnusedCreationService(),
        run_repository=repository,
        brand_memory_repository=repository,
        brand_voice_agent=agent,
        database=database,
    )
    return TestClient(app, raise_server_exceptions=raise_server_exceptions), database, repository


def test_brand_api_persists_samples_and_activates_extracted_profile(tmp_path: Path):
    from app.agents.contracts import BrandVoiceProfile

    profile = BrandVoiceProfile(
        persona="Practical systems builder",
        tone_attributes=["concise", "direct"],
        preferred_terms=["measurable"],
        avoid_terms=["game-changer"],
        structure_guidelines=["Use short paragraphs"],
    )
    agent = FakeBrandVoiceAgent(profile)
    client, database, repository = make_client(tmp_path, agent)
    try:
        response = client.post("/api/v1/brands/brand-1/samples", json=sample_batch())

        assert response.status_code == 200
        assert response.json()["version"] == 1
        assert response.json()["source_sample_count"] == 3
        assert response.json()["profile"]["persona"] == profile.persona
        assert len(agent.calls) == 1
        assert agent.calls[0][0] == "brand-1"
        assert len(agent.calls[0][1]) == 3
        assert len(repository.list_samples("brand-1")) == 3

        profile_response = client.get("/api/v1/brands/brand-1/profile")
        assert profile_response.status_code == 200
        assert profile_response.json()["source_sample_count"] == 3
        assert profile_response.json()["version"] == 1
        assert client.get("/api/v1/brands/unknown/profile").status_code == 404
    finally:
        client.close()
        database.close()


def test_brand_api_rejects_short_duplicate_and_empty_batches(tmp_path: Path):
    from app.agents.contracts import BrandVoiceProfile

    agent = FakeBrandVoiceAgent(
        BrandVoiceProfile(
            persona="P",
            tone_attributes=["direct"],
            preferred_terms=[],
            avoid_terms=[],
            structure_guidelines=["Use short paragraphs"],
        )
    )
    client, database, repository = make_client(tmp_path, agent)
    try:
        too_short = client.post(
            "/api/v1/brands/brand-1/samples",
            json={"samples": [{"body": "one"}, {"body": "two"}]},
        )
        duplicate = client.post(
            "/api/v1/brands/brand-1/samples",
            json={"samples": [{"body": "same"}, {"body": " same "}, {"body": "third"}]},
        )
        empty = client.post(
            "/api/v1/brands/brand-1/samples",
            json={"samples": [{"body": "first"}, {"body": " "}, {"body": "third"}]},
        )

        assert [too_short.status_code, duplicate.status_code, empty.status_code] == [422, 422, 422]
        assert agent.calls == []
        assert repository.list_samples("brand-1") == []
    finally:
        client.close()
        database.close()


def test_failed_extraction_keeps_previous_active_profile_unchanged(tmp_path: Path):
    agent = FakeBrandVoiceAgent(failure=RuntimeError("unexpected profile extraction error"))
    client, database, repository = make_client(tmp_path, agent, raise_server_exceptions=False)
    samples = repository.add_samples("brand-1", sample_batch("old")["samples"])
    old_profile = repository.save_profile(
        "brand-1",
        {"persona": "Original profile"},
        [sample.id for sample in samples],
    )
    try:
        response = client.post("/api/v1/brands/brand-1/samples", json=sample_batch("new"))

        assert response.status_code == 502
        assert "unexpected profile extraction error" not in response.text
        assert repository.get_active_profile("brand-1").id == old_profile.id
        assert len(repository.list_samples("brand-1")) == 3
    finally:
        client.close()
        database.close()
