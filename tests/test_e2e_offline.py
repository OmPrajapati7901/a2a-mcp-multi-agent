"""End-to-end test of the full pipeline in deterministic offline mode.

Runs run_demo.py exactly as a user would (subprocess), with A2A_DEMO_OFFLINE=1
so no API keys or network access are needed. Exercises the real MCP stdio
server, the real A2A server + client, and both agents' code paths.
"""
import os
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).parent.parent
TOPIC = "agent observability tools"


def run_demo(extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["A2A_DEMO_OFFLINE"] = "1"
    # Avoid colliding with a dev writer instance on the default port.
    env["WRITER_AGENT_PORT"] = "9111"
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "run_demo.py", TOPIC],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )


def test_e2e_offline_produces_report():
    proc = run_demo()
    assert proc.returncode == 0, f"demo failed:\n{proc.stderr[-3000:]}"
    assert "FINAL REPORT" in proc.stdout
    assert "metrics:" in proc.stdout
    # Offline mode must be honored end to end (no accidental key usage).
    assert "llm=offline" in proc.stderr
    assert "search=mock" in proc.stderr


def test_e2e_offline_a2a_handoff_visible_in_trace():
    proc = run_demo()
    assert proc.returncode == 0, f"demo failed:\n{proc.stderr[-3000:]}"
    trace = proc.stderr
    # The A2A lifecycle, in order: discovery → handoff → task → completion.
    discovery = trace.index("A2A DISCOVERY")
    handoff = trace.index("A2A HANDOFF: delegating")
    completed = trace.index("TASK_STATE_COMPLETED")
    assert discovery < handoff < completed
    # MCP tool boundary was exercised.
    assert "MCP session up" in trace
    assert "web_search returned" in trace
