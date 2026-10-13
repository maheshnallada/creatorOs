import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T")


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    initial_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)

    def __post_init__(self) -> None:
        if not 1 <= self.max_attempts <= 10:
            raise ValueError("max_attempts must be between 1 and 10")
        if self.initial_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("retry delays must not be negative")

    def run(self, operation: Callable[[], T]) -> T:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return operation()
            except Exception as error:
                if not getattr(error, "retryable", False) or attempt == self.max_attempts:
                    raise
                delay = min(
                    self.initial_delay_seconds * (2 ** (attempt - 1)),
                    self.max_delay_seconds,
                )
                self.sleep(delay)

        raise RuntimeError("retry loop exited unexpectedly")
