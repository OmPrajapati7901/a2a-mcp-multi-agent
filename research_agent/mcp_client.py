"""MCP client wiring for the Research Agent.

Spawns the web-search MCP server (stdio transport) and calls its web_search
tool. This is the agent→tool boundary; the agent→agent boundary is A2A
(see a2a_client.py).
"""
import json
import logging
import os
import sys

from opentelemetry.trace import SpanKind

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from common.tracing import inject_context, tracer

logger = logging.getLogger("research.mcp")


async def mcp_web_search(query: str, max_results: int = 5) -> list[dict]:
    """Run one web_search call against the MCP server, returning result dicts."""
    span_cm = tracer().start_as_current_span("mcp.web_search", kind=SpanKind.CLIENT)
    with span_cm as span:
        span.set_attribute("mcp.tool", "web_search")
        span.set_attribute("mcp.query", query)
        # Trace context crosses the process boundary via the spawn environment.
        env = dict(os.environ)
        env.update(
            {k.upper(): v for k, v in inject_context({}).items()}
        )
        return await _search(query, max_results, env, span)


async def _search(query: str, max_results: int, env: dict, span) -> list[dict]:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.search_server"],
        env=env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            logger.info(
                "MCP session up: server exposes %d tool(s): %s",
                len(tools.tools), [t.name for t in tools.tools],
            )
            result = await session.call_tool(
                "web_search", {"query": query, "max_results": max_results}
            )
            payload = result.content[0].text
            results = json.loads(payload)
            logger.info("MCP web_search returned %d results", len(results))
            span.set_attribute("mcp.result_count", len(results))
            return results
