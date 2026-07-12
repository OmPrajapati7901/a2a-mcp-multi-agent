"""Research & Report demo — topic in, report out.

Boots the Writer Agent (A2A server) if it isn't already running, then drives
the Research Agent pipeline: MCP web search → synthesize findings → delegate
writing to the Writer Agent over A2A → print the report.

Usage:
    uv run python run_demo.py "Summarize recent developments in agent observability tools"
"""
import argparse
import asyncio
import logging
import subprocess
import sys
import time

import httpx

import os

from common import (
    CRITIC_AGENT_URL,
    WRITER_AGENT_URL,
    have_anthropic,
    have_nvidia,
    have_tavily,
    setup_logging,
)
from common.tracing import setup_tracing, tracing_enabled

logger = logging.getLogger("demo")

CARD_PATH = "/.well-known/agent-card.json"


def _is_up(health_url: str) -> bool:
    try:
        return httpx.get(health_url, timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def start_service(label: str, module: str, health_url: str) -> subprocess.Popen | None:
    """Spawn a service subprocess unless it's already running; wait healthy."""
    if _is_up(health_url):
        logger.info("%s already running (%s)", label, health_url)
        return None
    logger.info("spawning %s (%s)", label, health_url)
    proc = subprocess.Popen(
        [sys.executable, "-m", module],
        stdout=sys.stderr, stderr=sys.stderr,
    )
    for _ in range(50):
        if _is_up(health_url):
            logger.info("%s is up", label)
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"{label} exited during startup")
        time.sleep(0.2)
    proc.terminate()
    raise RuntimeError(f"{label} did not come up at {health_url}")


def start_writer() -> subprocess.Popen | None:
    """Kept for the eval harness: boots just the writer."""
    return start_service("Writer Agent", "writer_agent.a2a_server",
                         WRITER_AGENT_URL + CARD_PATH)


def start_stack() -> list[subprocess.Popen]:
    """Boot the demo topology: registry (opt-in), writer, critic (opt-in).
    The registry URL must be in the env before agents spawn so they
    self-register."""
    procs: list[subprocess.Popen] = []
    if os.environ.get("A2A_DASHBOARD"):
        dash_port = os.environ.get("DASHBOARD_PORT", "9200")
        os.environ.setdefault("A2A_DASHBOARD_URL",
                              f"http://127.0.0.1:{dash_port}")
        procs.append(start_service(
            "Dashboard", "dashboard.server",
            os.environ["A2A_DASHBOARD_URL"] + "/events"))
    if os.environ.get("A2A_REGISTRY"):
        registry_port = os.environ.get("REGISTRY_PORT", "9100")
        os.environ.setdefault("A2A_REGISTRY_URL",
                              f"http://127.0.0.1:{registry_port}")
        procs.append(start_service(
            "Agent Registry", "registry.server",
            os.environ["A2A_REGISTRY_URL"] + "/agents"))
    procs.append(start_writer())
    if os.environ.get("A2A_CRITIC"):
        procs.append(start_service(
            "Critic Agent", "critic_agent.a2a_server",
            CRITIC_AGENT_URL + CARD_PATH))
    return [p for p in procs if p is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description="A2A + MCP research & report demo")
    parser.add_argument("topic", help="research topic, e.g. 'agent observability tools'")
    parser.add_argument("--max-results", type=int, default=5,
                        help="max web search results (default 5)")
    args = parser.parse_args()

    setup_logging()
    setup_tracing("research-agent")
    if tracing_enabled():
        logger.info("tracing: enabled (research-agent)")
    llm_mode = ("anthropic" if have_anthropic()
                else "nvidia-nim" if have_nvidia() else "offline")
    logger.info("mode: llm=%s search=%s",
                llm_mode, "tavily" if have_tavily() else "mock")

    from common.cache import SemanticCache, cache_enabled

    cache = SemanticCache() if cache_enabled() else None
    t0 = time.perf_counter()
    cached = cache.lookup(args.topic) if cache else None
    if cached is not None:
        # Cache hit: whole pipeline skipped — no writer, no tokens, no A2A.
        _print_result(cached["report"], cached.get("structured_report") or {},
                      cached, t0, cache_hit=True)
        return

    procs = start_stack()
    try:
        from research_agent.agent import run_research

        state = asyncio.run(run_research(args.topic, args.max_results))
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            proc.wait(timeout=10)

    if cache is not None:
        cache.store(args.topic, {
            "report": state["report"],
            "structured_report": state.get("structured_report"),
            "raw_results": state["raw_results"],
            "timings": state["timings"],
        })

    from common.costs import Ledger

    _print_result(state["report"], state.get("structured_report") or {},
                  state, t0, ledger=Ledger.from_dict(state["ledger"]))


def _print_result(report: str, structured: dict, state: dict, t0: float,
                  ledger=None, cache_hit: bool = False) -> None:
    total_s = round(time.perf_counter() - t0, 2)
    timings = state.get("timings") or {}
    citations = structured.get("citations", [])

    print("\n" + "=" * 72)
    print("FINAL REPORT" + ("  [semantic cache hit]" if cache_hit else ""))
    print("=" * 72)
    print(report)
    if citations:
        print("\nSources cited:")
        for c in citations:
            print(f"  [S{int(c['sid'])}] {c['title']} — {c['url']}")
    print("=" * 72)
    print(
        f"metrics: total={total_s}s | search={timings.get('search_s')}s | "
        f"synthesis={timings.get('synthesis_s')}s | "
        f"a2a_delegation={timings.get('delegation_s')}s | "
        f"search_results={len(state.get('raw_results') or [])} | "
        f"citations={len(citations)}/{int(structured.get('sources_available', 0))}"
        + (f" | cache=hit(sim={state.get('cache_similarity')})" if cache_hit else "")
    )
    if cache_hit:
        print("cost:    $0 — served from semantic cache, no LLM or A2A calls")
    else:
        print(f"cost:    {ledger.summary()}")


if __name__ == "__main__":
    main()
