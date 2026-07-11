"""MCP server exposing one tool: web_search.

Backed by the Tavily Search API when TAVILY_API_KEY is set; otherwise returns
deterministic mock results so the demo runs fully offline.

Runs over stdio — the Research Agent spawns this process and talks MCP to it.
All logging goes to stderr (stdout is the MCP transport).
"""
import json
import logging
import os

import httpx
from mcp.server.fastmcp import FastMCP

from common import setup_logging

logger = logging.getLogger("mcp.web-search")

mcp = FastMCP("web-search")

TAVILY_ENDPOINT = "https://api.tavily.com/search"

MOCK_RESULTS = [
    {
        "title": "Agent observability: why tracing multi-agent systems is hard",
        "url": "https://example.com/agent-observability-tracing",
        "content": (
            "Multi-agent systems introduce new observability challenges: task "
            "handoffs between agents, tool-call chains, and non-deterministic "
            "LLM outputs. OpenTelemetry-based tracing with span-per-agent-step "
            "is emerging as the common pattern."
        ),
    },
    {
        "title": "Survey of LLM evaluation and monitoring platforms",
        "url": "https://example.com/llm-monitoring-survey",
        "content": (
            "Platforms such as Arize Phoenix, Langfuse, and LangSmith now offer "
            "session-level traces, token/cost accounting, and eval pipelines. "
            "Open standards (OpenInference, OTel GenAI semantic conventions) are "
            "converging on shared span attributes for LLM calls."
        ),
    },
    {
        "title": "A2A and MCP: interoperability protocols for agents",
        "url": "https://example.com/a2a-mcp-interop",
        "content": (
            "The Agent2Agent (A2A) protocol standardizes discovery via Agent "
            "Cards and task delegation between agents built on different "
            "frameworks, while MCP standardizes how a single agent reaches its "
            "tools. Together they separate the agent-to-agent boundary from the "
            "agent-to-tool boundary."
        ),
    },
]


async def _tavily_search(query: str, max_results: int) -> list[dict]:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            TAVILY_ENDPOINT,
            headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"},
            json={"query": query, "max_results": max_results},
        )
        resp.raise_for_status()
        return [
            {"title": r["title"], "url": r["url"], "content": r["content"]}
            for r in resp.json().get("results", [])
        ]


@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return a JSON list of {title, url, content} results."""
    if os.environ.get("TAVILY_API_KEY"):
        logger.info("web_search(tavily) query=%r max_results=%d", query, max_results)
        results = await _tavily_search(query, max_results)
    else:
        logger.info("web_search(mock) query=%r max_results=%d", query, max_results)
        results = MOCK_RESULTS[:max_results]
    return json.dumps(results)


if __name__ == "__main__":
    setup_logging()
    logger.info(
        "starting MCP web-search server over stdio (backend=%s)",
        "tavily" if os.environ.get("TAVILY_API_KEY") else "mock",
    )
    mcp.run(transport="stdio")
