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

from common import (
    WRITER_AGENT_URL,
    have_anthropic,
    have_nvidia,
    have_tavily,
    setup_logging,
)
from common.tracing import setup_tracing, tracing_enabled

logger = logging.getLogger("demo")

CARD_PATH = "/.well-known/agent-card.json"


def writer_is_up() -> bool:
    try:
        return httpx.get(WRITER_AGENT_URL + CARD_PATH, timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def start_writer() -> subprocess.Popen | None:
    if writer_is_up():
        logger.info("Writer Agent already running at %s", WRITER_AGENT_URL)
        return None
    logger.info("spawning Writer Agent A2A server (%s)", WRITER_AGENT_URL)
    proc = subprocess.Popen(
        [sys.executable, "-m", "writer_agent.a2a_server"],
        stdout=sys.stderr, stderr=sys.stderr,
    )
    for _ in range(50):
        if writer_is_up():
            logger.info("Writer Agent is up; Agent Card served at %s%s",
                        WRITER_AGENT_URL, CARD_PATH)
            return proc
        if proc.poll() is not None:
            raise RuntimeError("Writer Agent server exited during startup")
        time.sleep(0.2)
    proc.terminate()
    raise RuntimeError(f"Writer Agent did not come up at {WRITER_AGENT_URL}")


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

    writer_proc = start_writer()
    try:
        from research_agent.agent import run_research

        state = asyncio.run(run_research(args.topic, args.max_results))
    finally:
        if writer_proc is not None:
            writer_proc.terminate()
            writer_proc.wait(timeout=10)

    if cache is not None:
        cache.store(args.topic, {
            "report": state["report"],
            "structured_report": state.get("structured_report"),
            "raw_results": state["raw_results"],
            "timings": state["timings"],
        })

    _print_result(state["report"], state.get("structured_report") or {},
                  state, t0, ledger=state["ledger"])


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
