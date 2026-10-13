from pathlib import Path

import pytest


def load_repository_types():
    try:
        from app.db import Database
        from app.repositories.runs import RunNotFoundError, RunRepository
    except ModuleNotFoundError as error:
        pytest.fail(f"run persistence implementation is missing: {error.name}")
    return Database, RunNotFoundError, RunRepository


def create_repository(database_url: str):
    database_type, _, repository_type = load_repository_types()
    database = database_type(database_url)
    database.create_schema()
    return database, repository_type(database.session_factory)


def test_repository_creates_reads_and_updates_run_state(tmp_path: Path):
    database, repository = create_repository(f"sqlite:///{tmp_path / 'runs.db'}")
    try:
        created = repository.create("run-1", {"topic": "AI safety"})

        assert created.id == "run-1"
        assert created.brief == {"topic": "AI safety"}
        assert created.status == "created"
        assert repository.get("missing") is None

        updated = repository.save_state("run-1", {"draft": "A revised post"}, status="needs_review")

        assert updated.status == "needs_review"
        assert repository.get("run-1").workflow_state == {"draft": "A revised post"}
    finally:
        database.close()


def test_repository_appends_ordered_run_linked_events(tmp_path: Path):
    database, repository = create_repository(f"sqlite:///{tmp_path / 'events.db'}")
    try:
        repository.create("run-1", {"topic": "AI safety"})
        first = repository.append_event(
            "run-1",
            "strategy_selected",
            "strategist",
            "Selected a concrete angle",
            {"angle": "safety"},
        )
        second = repository.append_event(
            "run-1", "editor_critiqued", "editor", "Asked for a sharper hook", {"score": 6.0}
        )

        events = repository.list_events("run-1")

        assert [event.sequence for event in events] == [1, 2]
        assert [event.id for event in events] == [first.id, second.id]
        assert all(event.run_id == "run-1" for event in events)
        assert events[0].payload == {"angle": "safety"}
    finally:
        database.close()


def test_repository_rejects_state_or_event_updates_for_missing_run(tmp_path: Path):
    _, not_found_error, _ = load_repository_types()
    database, repository = create_repository(f"sqlite:///{tmp_path / 'missing.db'}")
    try:
        with pytest.raises(not_found_error):
            repository.save_state("missing", {}, status="failed")
        with pytest.raises(not_found_error):
            repository.append_event("missing", "event", "agent", "reason", {})
    finally:
        database.close()


def test_repository_instances_are_isolated_by_database(tmp_path: Path):
    first_database, first_repository = create_repository(f"sqlite:///{tmp_path / 'first.db'}")
    second_database, second_repository = create_repository(f"sqlite:///{tmp_path / 'second.db'}")
    try:
        first_repository.create("same-id", {"topic": "first"})
        second_repository.create("same-id", {"topic": "second"})

        assert first_repository.get("same-id").brief == {"topic": "first"}
        assert second_repository.get("same-id").brief == {"topic": "second"}
    finally:
        first_database.close()
        second_database.close()
