# Failure modes and observed behavior

Every entry below is exercised by an automated test or was observed in a real
run. "What you see" quotes the actual log trace.

| # | Failure | Injection / occurrence | System behavior | What you see | Test |
|---|---------|------------------------|-----------------|--------------|------|
| 1 | Writer task fails transiently | `A2A_CHAOS_FAIL_N=2` (also observed live: 1/6 real-mode eval runs hit an upstream NIM error) | Retry with exponential backoff + jitter, up to 3 attempts | `CHAOS: injected failure…` → `a2a.delegate(write_report) failed (attempt 1/3) … retrying in 0.6s` → `succeeded on attempt 3/3` | `test_pipeline_survives_injected_writer_failures` |
| 2 | Writer keeps failing | `A2A_CHAOS_FAIL_N=5` | Retries exhausted; pipeline aborts with non-zero exit; breaker records failures | `RuntimeError: Writer Agent task … failed` after attempt 3/3 | `test_pipeline_fails_cleanly_when_chaos_exceeds_retries` |
| 3 | Writer clearly down (repeated failures) | 3 consecutive failures | Circuit opens: further calls refused for the 30s cooldown instead of hammering the agent; half-opens after cooldown, closes on first success | `circuit 'writer-agent' OPEN (3 consecutive failures, cooldown 30s)` → `CircuitOpenError` | `test_circuit_breaker_opens_and_recovers` |
| 4 | Writer not running at startup | Stop the writer process | `run_demo` polls the Agent Card for 10s, then aborts before any A2A call | `RuntimeError: Writer Agent did not come up at http://127.0.0.1:9001` | exercised by `start_writer` timeout path |
| 5 | Slow model exceeds HTTP timeout | Observed with GLM-5.2 (reasoning model) under default a2a-sdk timeouts | Delegation client uses a 600s httpx timeout; a genuine hang is retried (mode 1) then aborted (mode 2) | `A2AClientTimeoutError: Client Request timed out` (pre-fix); silent success now | fixed in `research_agent/a2a_client.py::DELEGATION_TIMEOUT` |
| 6 | Token budget exceeded | `A2A_BUDGET_TOKENS=<small>` | Pipeline aborts *before* the A2A delegation spends more money | `BudgetExceededError: run used N tokens ≥ budget M — refusing to delegate` | `test_budget_kill_switch` |
| 7 | Malformed/missing citations from the LLM | Weak model output | Report still returned; `citation_coverage` metric drops; CI eval gate fails offline if coverage < 100% | `A2A artifact received: … 0 citations` + eval table | `test_eval_gate_offline` |
| 8 | MCP search returns nothing | Empty Tavily response | Findings synthesize from zero results; writer still produces a caveated report (graceful degradation, no crash) | `MCP web_search returned 0 results` | covered by offline determinism |
