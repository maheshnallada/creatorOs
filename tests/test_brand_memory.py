from pathlib import Path

import pytest
from pydantic import ValidationError


def load_brand_memory_repository():
    try:
        from app.db import Database
        from app.repositories.brand_memory import BrandMemoryRepository
    except ModuleNotFoundError as error:
        pytest.fail(f"brand memory persistence is missing: {error.name}")
    return Database, BrandMemoryRepository


def test_brand_samples_and_profiles_are_versioned_with_source_ids(tmp_path: Path):
    database_type, repository_type = load_brand_memory_repository()
    database = database_type(f"sqlite:///{tmp_path / 'brand.db'}")
    database.create_schema()
    repository = repository_type(database.session_factory)
    try:
        samples = repository.add_samples(
            "brand-1",
            [
                {"body": "I build useful systems.", "source_label": "post-a"},
                {"body": "Small teams need clear feedback.", "source_label": "post-b"},
                {"body": "Measure what changes in production.", "source_label": "post-c"},
            ],
        )
        profile_v1 = repository.save_profile(
            "brand-1",
            {"persona": "Practical systems builder", "tone_attributes": ["concise"]},
            [sample.id for sample in samples],
        )
        profile_v2 = repository.save_profile(
            "brand-1",
            {"persona": "Pragmatic engineering leader", "tone_attributes": ["direct"]},
            [sample.id for sample in samples],
        )

        assert len(repository.list_samples("brand-1")) == 3
        assert profile_v1.version == 1
        assert profile_v2.version == 2
        assert profile_v2.is_active is True
        assert profile_v2.source_sample_ids == [sample.id for sample in samples]
        versions = repository.list_profile_versions("brand-1")
        assert [version.is_active for version in versions] == [False, True]
        assert repository.get_active_profile("brand-1").id == profile_v2.id
    finally:
        database.close()


def test_profile_rejects_sample_ids_owned_by_another_brand(tmp_path: Path):
    database_type, repository_type = load_brand_memory_repository()
    database = database_type(f"sqlite:///{tmp_path / 'brand-source.db'}")
    database.create_schema()
    repository = repository_type(database.session_factory)
    try:
        samples = repository.add_samples(
            "brand-a",
            [
                {"body": "Sample one"},
                {"body": "Sample two"},
                {"body": "Sample three"},
            ],
        )

        with pytest.raises(ValueError, match="source samples"):
            repository.save_profile(
                "brand-b",
                {"persona": "Other brand"},
                [sample.id for sample in samples],
            )

        assert repository.get_active_profile("brand-b") is None
    finally:
        database.close()


def test_profile_requires_three_distinct_nonempty_samples():
    try:
        from app.agents.contracts import BrandVoiceSamplesRequest
    except ImportError as error:
        pytest.fail(f"brand voice sample contract is missing: {error.name}")

    with pytest.raises(ValidationError):
        BrandVoiceSamplesRequest(samples=[{"body": "one"}, {"body": "two"}])
    with pytest.raises(ValidationError):
        BrandVoiceSamplesRequest(samples=[{"body": "same"}, {"body": " same "}, {"body": "other"}])
    with pytest.raises(ValidationError):
        BrandVoiceSamplesRequest(samples=[{"body": "first"}, {"body": "   "}, {"body": "third"}])
