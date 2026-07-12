"""Chaos tests: inject failures at the A2A boundary and verify recovery.

The writer fails its first N tasks (A2A_CHAOS_FAIL_N); the research agent's
retry layer must absorb them. The circuit breaker is unit-tested directly.
"""
import asyncio

import pytest

from common.resilience import CircuitBreaker, CircuitOpenError, with_retries
from tests.test_e2e_offline import run_demo


def test_pipeline_survives_injected_writer_failures():
    proc = run_demo({"A2A_CHAOS_FAIL_N": "2", "WRITER_AGENT_PORT": "9119"})
    assert proc.returncode == 0, f"demo failed:\n{proc.stderr[-3000:]}"
    assert "FINAL REPORT" in proc.stdout
    trace = proc.stderr
    assert trace.count("CHAOS: injected failure") == 2
    assert "retrying in" in trace
    assert "succeeded on attempt 3/3" in trace


def test_pipeline_fails_cleanly_when_chaos_exceeds_retries():
    proc = run_demo({"A2A_CHAOS_FAIL_N": "5", "WRITER_AGENT_PORT": "9120"})
    assert proc.returncode != 0
    assert "FINAL REPORT" not in proc.stdout


def test_circuit_breaker_opens_and_recovers():
    breaker = CircuitBreaker("test", failure_threshold=2, cooldown_s=0.2)

    async def boom():
        raise ValueError("down")

    async def scenario():
        with pytest.raises(ValueError):
            await with_retries(boom, name="t", attempts=2,
                               base_delay_s=0.01, breaker=breaker)
        assert breaker.state == "open"
        # While open, calls are refused without touching the target.
        with pytest.raises(CircuitOpenError):
            await with_retries(boom, name="t", attempts=2,
                               base_delay_s=0.01, breaker=breaker)
        # After cooldown it half-opens; a success closes it.
        await asyncio.sleep(0.25)
        assert breaker.state == "half-open"

        async def ok():
            return 42

        assert await with_retries(ok, name="t", breaker=breaker) == 42
        assert breaker.state == "closed"

    asyncio.run(scenario())
