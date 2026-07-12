"""Research Agent — LangGraph.

Graph: search (MCP tool) → synthesize (Claude or offline) → delegate (A2A to
Writer Agent). State carries the topic through to the final report, plus
per-phase timings for the demo metrics.
"""
import logging
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

import os

from common import (
    CLAUDE_MODEL,
    CRITIC_AGENT_URL,
    NVIDIA_MODEL,
    WRITER_AGENT_URL,
    have_anthropic,
    have_nvidia,
)
from common.costs import Ledger, estimate_tokens
from common.tracing import tracer
from common.report import format_sources
from registry.client import find_agent
from research_agent.a2a_client import (
    InputRequiredError,
    continue_task,
    delegate_report,
    delegate_review,
    discover_agent,
)
from research_agent.mcp_client import mcp_web_search

MAX_REVISION_ROUNDS = 1


def critic_enabled() -> bool:
    return bool(os.environ.get("A2A_CRITIC"))

logger = logging.getLogger("research.agent")

SYNTHESIS_PROMPT = (
    "You are a research analyst. Distill the following web search results "
    "about {topic!r} into 3-6 concise bullet-point findings (facts, trends, "
    "named tools/standards). Each search result is numbered [S1], [S2], … — "
    "every bullet MUST end with the [S#] marker(s) of the result(s) it came "
    "from. Output only the bullets, one per line, each starting with '- '."
    "\n\nSearch results:\n{results}"
)


class ResearchState(TypedDict, total=False):
    topic: str
    max_results: int
    raw_results: list[dict]
    findings: str
    report: str
    structured_report: dict | None
    timings: dict[str, float]
    ledger: Ledger
    critic_verdict: str
    critic_feedback: str
    revision_rounds: int


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
        f"[S{i}] {r['title']} ({r['url']})\n{r['content']}"
        for i, r in enumerate(state["raw_results"], 1)
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
    ledger = state["ledger"]
    if llm is not None:
        prompt = SYNTHESIS_PROMPT.format(topic=state["topic"], results=results_text)
        msg = await llm.ainvoke(prompt)
        findings = str(msg.text)
        usage = getattr(msg, "usage_metadata", None) or {}
        ledger.add(
            "research-agent",
            usage.get("input_tokens") or estimate_tokens(prompt),
            usage.get("output_tokens") or estimate_tokens(findings),
            estimated=not usage,
        )
    else:
        logger.info("node=synthesize: offline mode — formatting results as findings")
        findings = "\n".join(
            f"- {r['content']} [S{i}]"
            for i, r in enumerate(state["raw_results"], 1)
        )
        ledger.add(
            "research-agent",
            estimate_tokens(results_text), estimate_tokens(findings),
            estimated=True,
        )
    logger.info("node=synthesize: %d findings lines", findings.count("- "))
    return {"findings": findings, "timings": _mark(state, "synthesis_s", t0)}


async def delegate_node(state: ResearchState) -> ResearchState:
    t0 = time.perf_counter()
    # Kill switch: stop before spending more, not after.
    state["ledger"].check_budget("delegate to the Writer Agent")
    # Semantic routing when a registry is configured; static URL otherwise.
    writer_url = await find_agent(
        "turn research findings into a polished written report",
        WRITER_AGENT_URL,
    )
    logger.info("node=delegate: discovering Writer Agent at %s", writer_url)
    card = await discover_agent(writer_url)
    feedback = state.get("critic_feedback") or None
    if feedback:
        logger.info("node=delegate: REVISION round %d with critic feedback",
                    state.get("revision_rounds", 0))
    findings, results = state["findings"], state["raw_results"]
    try:
        report, structured = await delegate_report(
            card, state["topic"], findings, results, feedback=feedback,
        )
    except InputRequiredError as need:
        # The writer paused the task asking for more input — negotiate:
        # search again, merge anything new, and resume the SAME task.
        logger.info("node=delegate: NEGOTIATION — writer asked: %s "
                    "(task_id=%r ctx=%r)",
                    need.request, need.task_id, need.context_id)
        extra = await mcp_web_search(
            f"{state['topic']} additional developments",
            state.get("max_results", 5) + 3,
        )
        known = {r["url"] for r in results}
        fresh = [r for r in extra if r["url"] not in known]
        results = results + fresh
        if fresh:
            findings += "\n" + "\n".join(
                f"- {r['content']} [S{i}]"
                for i, r in enumerate(fresh, len(known) + 1)
            )
            note = f"Added {len(fresh)} more finding(s) from a follow-up search."
        else:
            note = ("A follow-up search surfaced no new sources; these "
                    "findings are complete — please proceed.")
        task_text = (
            f"Topic: {state['topic']}\n\nFindings:\n{findings}\n\n"
            f"Sources:\n{format_sources(results)}\n\n{note}"
        )
        report, structured = await continue_task(
            card, need.task_id, need.context_id, task_text,
        )
    usage = (structured or {}).get("usage") or {}
    state["ledger"].add(
        "writer-agent",
        usage.get("input_tokens", 0), usage.get("output_tokens", 0),
        estimated=bool(usage.get("estimated")),
    )
    return {
        "report": report,
        "structured_report": structured,
        "raw_results": results,
        "timings": _mark(state, "delegation_s", t0),
    }


async def review_node(state: ResearchState) -> ResearchState:
    """Reflection: a Critic Agent on a third framework (OpenAI Agents SDK)
    reviews the report over A2A; a revise verdict loops back to the writer
    with feedback, bounded by MAX_REVISION_ROUNDS."""
    t0 = time.perf_counter()
    critic_url = await find_agent(
        "review and critique a research report for quality and citations",
        CRITIC_AGENT_URL,
    )
    logger.info("node=review: discovering Critic Agent at %s", critic_url)
    card = await discover_agent(critic_url)
    review = await delegate_review(card, state["report"], state["findings"])
    usage = review.get("usage") or {}
    state["ledger"].add(
        "critic-agent",
        usage.get("input_tokens", 0), usage.get("output_tokens", 0),
        estimated=bool(usage.get("estimated")),
    )
    logger.info("node=review: critic verdict=%s", review["verdict"])
    timings = _mark(state, f"review_{state.get('revision_rounds', 0)}_s", t0)
    return {
        "critic_verdict": review["verdict"],
        "critic_feedback": review.get("feedback", ""),
        "revision_rounds": state.get("revision_rounds", 0) + 1,
        "timings": timings,
    }


def _after_review(state: ResearchState) -> str:
    if (state["critic_verdict"] == "revise"
            and state["revision_rounds"] <= MAX_REVISION_ROUNDS):
        logger.info("reflection: revising report (round %d/%d)",
                    state["revision_rounds"], MAX_REVISION_ROUNDS)
        return "delegate"
    if state["critic_verdict"] == "revise":
        logger.warning("reflection: revision budget exhausted — "
                       "shipping last draft")
    return END


def _after_delegate(state: ResearchState) -> str:
    return "review" if critic_enabled() else END


def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("search", search_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("delegate", delegate_node)
    g.add_node("review", review_node)
    g.add_edge(START, "search")
    g.add_edge("search", "synthesize")
    g.add_edge("synthesize", "delegate")
    g.add_conditional_edges("delegate", _after_delegate, ["review", END])
    g.add_conditional_edges("review", _after_review, ["delegate", END])
    return g.compile()


async def run_research(topic: str, max_results: int = 5) -> ResearchState:
    graph = build_graph()
    logger.info("research pipeline start: topic=%r", topic)
    with tracer().start_as_current_span("research.pipeline") as span:
        span.set_attribute("research.topic", topic)
        final: ResearchState = await graph.ainvoke(
            {"topic": topic, "max_results": max_results, "ledger": Ledger()}
        )
        span.set_attribute("research.report_chars", len(final["report"]))
    logger.info("research pipeline done: timings=%s | %s",
                final["timings"], final["ledger"].summary())
    return final
