"""Phase 3 features, all offline-deterministic: registry semantic routing,
critic reflection loop, multi-turn A2A negotiation, HITL gate, dashboard."""
import json
import os
import pathlib
import subprocess
import sys
import time

import httpx

from tests.test_e2e_offline import REPO_ROOT, run_demo


def test_registry_routes_and_critic_reflects():
    proc = run_demo({
        "A2A_REGISTRY": "1", "A2A_CRITIC": "1",
        "A2A_CRITIC_FORCE_REVISE": "1",
        "WRITER_AGENT_PORT": "9151", "CRITIC_AGENT_PORT": "9152",
        "REGISTRY_PORT": "9153",
        "A2A_REGISTRY_URL": "http://127.0.0.1:9153",
    })
    assert proc.returncode == 0, f"demo failed:\n{proc.stderr[-3000:]}"
    trace = proc.stderr
    # Both agents self-registered and were routed to semantically.
    assert trace.count("REGISTRY ROUTE") >= 2
    assert "→ 'Writer Agent'" in trace and "→ 'Critic Agent'" in trace
    # The forced revise triggered exactly one bounded revision round.
    assert "critic verdict: revise" in trace
    assert "REVISION round 1 with critic feedback" in trace
    assert "critic verdict: approve" in trace
    assert "FINAL REPORT" in proc.stdout


def test_multi_turn_negotiation_resumes_same_task():
    proc = run_demo({"A2A_MIN_FINDINGS": "5", "WRITER_AGENT_PORT": "9154"})
    assert proc.returncode == 0, f"demo failed:\n{proc.stderr[-3000:]}"
    trace = proc.stderr
    assert "TASK_STATE_INPUT_REQUIRED" in trace
    assert "task resumed with new input" in trace
    # The paused task id and the resumed task id must match.
    import re
    paused = re.search(r"pausing task (\S+)", trace).group(1)
    resumed = re.search(r"task resumed with new input: task_id=(\S+) ", trace).group(1)
    assert paused == resumed
    assert "FINAL REPORT" in proc.stdout


def test_hitl_approve_and_audit(tmp_path):
    env = {"A2A_HITL": "1", "A2A_AUDIT_DIR": str(tmp_path),
           "WRITER_AGENT_PORT": "9155"}
    proc = run_demo(env, stdin_text="approve\n")
    assert proc.returncode == 0, f"demo failed:\n{proc.stderr[-3000:]}"
    entries = [json.loads(l) for l in
               (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert entries[-1]["decision"] == "approve"
    assert entries[-1]["source"] == "interactive"


def test_hitl_reject_blocks_delegation(tmp_path):
    env = {"A2A_HITL": "1", "A2A_AUDIT_DIR": str(tmp_path),
           "WRITER_AGENT_PORT": "9156"}
    proc = run_demo(env, stdin_text="reject\n")
    assert proc.returncode != 0
    assert "delegation rejected by human reviewer" in proc.stderr
    assert "A2A HANDOFF" not in proc.stderr
    entries = [json.loads(l) for l in
               (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert entries[-1]["decision"] == "reject"


def test_dashboard_collects_events():
    env = dict(os.environ)
    env.update({"A2A_DEMO_OFFLINE": "1", "DASHBOARD_PORT": "9157"})
    proc = subprocess.Popen(
        [sys.executable, "-m", "dashboard.server"], cwd=REPO_ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = "http://127.0.0.1:9157"
    try:
        for _ in range(50):
            try:
                if httpx.get(base + "/events", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        httpx.post(base + "/event", json={
            "ts": "12:00:00.000", "agent": "writer-agent",
            "kind": "task_completed", "detail": {"chars": 42},
        })
        events = httpx.get(base + "/events").json()
        assert events and events[-1]["kind"] == "task_completed"
        page = httpx.get(base + "/").text
        assert "live events" in page
    finally:
        proc.terminate()
        proc.wait(timeout=10)
