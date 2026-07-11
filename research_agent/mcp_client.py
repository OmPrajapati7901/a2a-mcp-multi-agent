"""MCP client wiring for the Research Agent.

Spawns the web-search MCP server (stdio transport) and calls its web_search
tool. This is the agent→tool boundary; the agent→agent boundary is A2A
(see a2a_client.py).
"""
import json
import logging
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger("research.mcp")


async def mcp_web_search(query: str, max_results: int = 5) -> list[dict]:
    """Run one web_search call against the MCP server, returning result dicts."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.search_server"],
        env=dict(os.environ),
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
            return results
