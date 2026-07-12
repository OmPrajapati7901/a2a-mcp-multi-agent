"""Retry and circuit-breaker primitives for cross-agent calls.

An A2A delegation is a network call to another autonomous system — it fails
in all the ways RPCs fail (timeouts, transient upstream errors, crashes), so
it gets the same treatment: bounded exponential-backoff retries in front of a
circuit breaker that stops hammering an agent that is clearly down.
"""
import asyncio
import logging
import random
import time

logger = logging.getLogger("resilience")


class CircuitOpenError(RuntimeError):
    """The breaker is open: recent calls failed; not attempting another."""


class CircuitBreaker:
    """Open after `failure_threshold` consecutive failures; allow a probe
    call again after `cooldown_s` (half-open); close on success."""

    def __init__(self, name: str, failure_threshold: int = 3, cooldown_s: float = 30.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if time.monotonic() - self._opened_at >= self.cooldown_s:
            return "half-open"
        return "open"

    def check(self) -> None:
        if self.state == "open":
            raise CircuitOpenError(
                f"circuit {self.name!r} open after "
                f"{self._consecutive_failures} consecutive failures"
            )

    def record_success(self) -> None:
        if self._opened_at is not None:
            logger.info("circuit %r closed again", self.name)
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if (self._consecutive_failures >= self.failure_threshold
                and self._opened_at is None):
            self._opened_at = time.monotonic()
            logger.warning(
                "circuit %r OPEN (%d consecutive failures, cooldown %.0fs)",
                self.name, self._consecutive_failures, self.cooldown_s,
            )


async def with_retries(
    fn,
    *,
    name: str,
    attempts: int = 3,
    base_delay_s: float = 0.5,
    retry_on: tuple[type[Exception], ...] = (Exception,),
    breaker: CircuitBreaker | None = None,
):
    """Run `await fn()` with exponential backoff + jitter and an optional
    circuit breaker. Raises the last error once attempts are exhausted."""
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        if breaker is not None:
            breaker.check()
        try:
            result = await fn()
        except retry_on as exc:
            last_exc = exc
            if breaker is not None:
                breaker.record_failure()
            if attempt == attempts:
                break
            delay = base_delay_s * (2 ** (attempt - 1)) * (1 + random.random() * 0.2)
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                name, attempt, attempts, exc, delay,
            )
            await asyncio.sleep(delay)
        else:
            if breaker is not None:
                breaker.record_success()
            if attempt > 1:
                logger.info("%s succeeded on attempt %d/%d", name, attempt, attempts)
            return result
    raise last_exc
