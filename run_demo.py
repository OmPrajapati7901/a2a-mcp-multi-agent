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
    llm_mode = ("anthropic" if have_anthropic()
                else "nvidia-nim" if have_nvidia() else "offline")
    logger.info("mode: llm=%s search=%s",
                llm_mode, "tavily" if have_tavily() else "mock")

    writer_proc = start_writer()
    t0 = time.perf_counter()
    try:
        from research_agent.agent import run_research

        state = asyncio.run(run_research(args.topic, args.max_results))
    finally:
        if writer_proc is not None:
            writer_proc.terminate()
            writer_proc.wait(timeout=10)

    total_s = round(time.perf_counter() - t0, 2)
    timings = state["timings"]

    print("\n" + "=" * 72)
    print("FINAL REPORT")
    print("=" * 72)
    print(state["report"])
    print("=" * 72)
    print(
        f"metrics: total={total_s}s | search={timings.get('search_s')}s | "
        f"synthesis={timings.get('synthesis_s')}s | "
        f"a2a_delegation={timings.get('delegation_s')}s | "
        f"search_results={len(state['raw_results'])}"
    )


if __name__ == "__main__":
    main()
