import pytest


def load_retry_policy():
    try:
        from app.llm.retry import RetryPolicy
    except ModuleNotFoundError as error:
        pytest.fail(f"retry policy implementation is missing: {error.name}")
    return RetryPolicy


class TemporaryFailure(Exception):
    retryable = True


class PermanentFailure(Exception):
    retryable = False


def test_retry_policy_retries_transient_failure_then_returns_result():
    retry_policy_type = load_retry_policy()
    attempts = 0
    delays = []

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TemporaryFailure("temporary")
        return "done"

    result = retry_policy_type(
        max_attempts=3,
        initial_delay_seconds=0.1,
        max_delay_seconds=1,
        sleep=delays.append,
    ).run(operation)

    assert result == "done"
    assert attempts == 3
    assert delays == [0.1, 0.2]


def test_default_retry_policy_starts_with_one_point_five_second_delay():
    retry_policy_type = load_retry_policy()
    attempts = 0
    delays = []

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TemporaryFailure("temporary")
        return "done"

    result = retry_policy_type(max_attempts=2, sleep=delays.append).run(operation)

    assert result == "done"
    assert delays == [1.5]


def test_retry_policy_does_not_retry_permanent_failure():
    retry_policy_type = load_retry_policy()
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        raise PermanentFailure("permanent")

    with pytest.raises(PermanentFailure):
        retry_policy_type(max_attempts=4, sleep=lambda _: None).run(operation)

    assert attempts == 1


def test_retry_policy_stops_after_configured_attempt_limit():
    retry_policy_type = load_retry_policy()
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        raise TemporaryFailure("temporary")

    with pytest.raises(TemporaryFailure):
        retry_policy_type(max_attempts=2, sleep=lambda _: None).run(operation)

    assert attempts == 2
