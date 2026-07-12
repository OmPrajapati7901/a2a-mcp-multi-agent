"""A2A API-key auth and semantic-cache tests (offline, deterministic)."""
import json
import os
import pathlib
import subprocess
import sys
import time

import httpx
import pytest

from tests.test_e2e_offline import REPO_ROOT, TOPIC, run_demo

AUTH_PORT = 9125


@pytest.fixture
def secured_writer():
    env = dict(os.environ)
    env.update({
        "A2A_DEMO_OFFLINE": "1",
        "WRITER_AGENT_PORT": str(AUTH_PORT),
        "A2A_API_KEY": "test-secret",
    })
    proc = subprocess.Popen(
        [sys.executable, "-m", "writer_agent.a2a_server"],
        cwd=REPO_ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{AUTH_PORT}"
    for _ in range(50):
        try:
            if httpx.get(base + "/.well-known/agent-card.json", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    else:
        proc.terminate()
        raise RuntimeError("secured writer did not start")
    yield base
    proc.terminate()
    proc.wait(timeout=10)


def test_card_is_public_and_declares_scheme(secured_writer):
    card = httpx.get(secured_writer + "/.well-known/agent-card.json").json()
    assert "api-key" in card.get("securitySchemes", {})
    scheme = card["securitySchemes"]["api-key"]["apiKeySecurityScheme"]
    assert scheme["name"] == "X-API-Key"


def test_rpc_rejected_without_key(secured_writer):
    resp = httpx.post(secured_writer + "/", json={"jsonrpc": "2.0", "id": 1,
                                                  "method": "message/send"})
    assert resp.status_code == 401


def test_pipeline_succeeds_with_key():
    proc = run_demo({"A2A_API_KEY": "test-secret",
                     "WRITER_AGENT_PORT": "9126"})
    assert proc.returncode == 0, f"demo failed:\n{proc.stderr[-2000:]}"
    assert "FINAL REPORT" in proc.stdout


def test_semantic_cache_hit_skips_pipeline(tmp_path):
    env = {"A2A_CACHE": "1", "A2A_CACHE_DIR": str(tmp_path),
           "WRITER_AGENT_PORT": "9127"}
    first = run_demo(env)
    assert first.returncode == 0
    assert "cache MISS" in first.stderr and "cache STORE" in first.stderr

    second = run_demo(env)
    assert second.returncode == 0
    assert "cache HIT" in second.stderr
    assert "[semantic cache hit]" in second.stdout
    assert "cost:    $0" in second.stdout
    # No pipeline ran: no MCP call, no A2A handoff.
    assert "A2A HANDOFF" not in second.stderr
    assert "MCP session up" not in second.stderr
