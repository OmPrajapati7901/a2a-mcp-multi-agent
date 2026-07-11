"""Research Agent — LangGraph.

Graph: search (MCP tool) → synthesize (Claude or offline) → delegate (A2A to
Writer Agent). State carries the topic through to the final report, plus
per-phase timings for the demo metrics.
"""
import logging
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from common import (
    CLAUDE_MODEL,
    NVIDIA_MODEL,
    WRITER_AGENT_URL,
    have_anthropic,
    have_nvidia,
)
from research_agent.a2a_client import delegate_report, discover_writer
from research_agent.mcp_client import mcp_web_search

logger = logging.getLogger("research.agent")

SYNTHESIS_PROMPT = (
    "You are a research analyst. Distill the following web search results "
    "about {topic!r} into 3-6 concise bullet-point findings (facts, trends, "
    "named tools/standards). Output only the bullets, one per line, each "
    "starting with '- '.\n\nSearch results:\n{results}"
)


class ResearchState(TypedDict, total=False):
    topic: str
    max_results: int
    raw_results: list[dict]
    findings: str
    report: str
    timings: dict[str, float]


def _mark(state: ResearchState, phase: str, start: float) -> dict[str, float]:
    timings = dict(state.get("timings", {}))
    timings[phase] = round(time.perf_counter() - start, 2)
    return timings


async def search_node(state: ResearchState) -> ResearchState:
    t0 = time.perf_counter()
    logger.info("node=search: querying MCP web_search for %r", state["topic"])
    results = await mcp_web_search(state["topic"], state.get("max_results", 5))
    return {"raw_results": results, "timings": _mark(state, "search_s", t0)}


async def synthesize_node(state: ResearchState) -> ResearchState:
    t0 = time.perf_counter()
    results_text = "\n\n".join(
        f"[{r['title']}]({r['url']})\n{r['content']}" for r in state["raw_results"]
    )
    llm = None
    if have_anthropic():
        from langchain_anthropic import ChatAnthropic

        logger.info("node=synthesize: distilling findings with %s", CLAUDE_MODEL)
        llm = ChatAnthropic(model=CLAUDE_MODEL, max_tokens=1024)
    elif have_nvidia():
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        logger.info("node=synthesize: distilling findings with %s (NVIDIA NIM)",
                    NVIDIA_MODEL)
        llm = ChatNVIDIA(model=NVIDIA_MODEL, max_tokens=4096)
    if llm is not None:
        msg = await llm.ainvoke(
            SYNTHESIS_PROMPT.format(topic=state["topic"], results=results_text)
        )
        findings = str(msg.text)
    else:
        logger.info("node=synthesize: offline mode — formatting results as findings")
        findings = "\n".join(
            f"- {r['content']} (source: {r['title']})" for r in state["raw_results"]
        )
    logger.info("node=synthesize: %d findings lines", findings.count("- "))
    return {"findings": findings, "timings": _mark(state, "synthesis_s", t0)}


async def delegate_node(state: ResearchState) -> ResearchState:
    t0 = time.perf_counter()
    logger.info("node=delegate: discovering Writer Agent at %s", WRITER_AGENT_URL)
    card = await discover_writer(WRITER_AGENT_URL)
    report = await delegate_report(card, state["topic"], state["findings"])
    return {"report": report, "timings": _mark(state, "delegation_s", t0)}


def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("search", search_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("delegate", delegate_node)
    g.add_edge(START, "search")
    g.add_edge("search", "synthesize")
    g.add_edge("synthesize", "delegate")
    g.add_edge("delegate", END)
    return g.compile()


async def run_research(topic: str, max_results: int = 5) -> ResearchState:
    graph = build_graph()
    logger.info("research pipeline start: topic=%r", topic)
    final: ResearchState = await graph.ainvoke(
        {"topic": topic, "max_results": max_results}
    )
    logger.info("research pipeline done: timings=%s", final["timings"])
    return final
