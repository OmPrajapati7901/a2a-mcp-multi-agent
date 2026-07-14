"""Research Agent — LangGraph.

Graph: search (MCP tool) → synthesize (Claude or offline) → delegate (A2A to
Writer Agent). State carries the topic through to the final report, plus
per-phase timings for the demo metrics.
"""
import logging
import os
import time
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from common import (
    CLAUDE_MODEL,
    CRITIC_AGENT_URL,
    NVIDIA_MODEL,
    WRITER_AGENT_URL,
    have_anthropic,
    have_nvidia,
)
from common.bandit import DEFAULT_ARMS, BanditRouter
from common.costs import Ledger, estimate_tokens
from common.events import emit_event
from common.guard import screen_results
from common.tracing import tracer
from common.report import VERDICT_APPROVE, VERDICT_REVISE, format_sources
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


def hitl_enabled() -> bool:
    return bool(os.environ.get("A2A_HITL"))

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
    ledger: dict
    guard_report: dict
    bandit_state: dict  # BanditRouter snapshot for msgpack serialization
    model_tier: str  # chosen model tier for this run
    writer_usage: dict  # writer token usage from the last delegation
    critic_verdict: str
    critic_feedback: str
    revision_rounds: int
    # NOTE: state must stay msgpack-serializable (checkpointer); the ledger
    # travels as a plain dict and is rehydrated via Ledger.from_dict.


def _mark(state: ResearchState, phase: str, start: float) -> dict[str, float]:
    timings = dict(state.get("timings", {}))
    timings[phase] = round(time.perf_counter() - start, 2)
    return timings


async def search_node(state: ResearchState) -> ResearchState:
    t0 = time.perf_counter()
    logger.info("node=search: querying MCP web_search for %r", state["topic"])
    results = await mcp_web_search(state["topic"], state.get("max_results", 5))
    # Trust boundary: web content is attacker-controllable. Screen it for
    # indirect prompt injection before it can reach the synthesis LLM.
    results, guard_report = screen_results(results)
    if guard_report["enabled"] and guard_report["flagged"]:
        logger.warning("node=search: guard neutralized injection in %d/%d "
                       "result(s), categories=%s",
                       guard_report["flagged"], len(results),
                       guard_report["categories"])
        emit_event("research-agent", "injection_blocked",
                   flagged=guard_report["flagged"],
                   categories=guard_report["categories"])
    emit_event("research-agent", "mcp_search_done", results=len(results))
    return {"raw_results": results, "guard_report": guard_report,
            "timings": _mark(state, "search_s", t0)}


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
    ledger = Ledger.from_dict(state["ledger"])
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
        # Bullets must stay single-line: the offline writer extracts them with
        # a `^- ` regex, so newlines inside result content would split a
        # bullet and detach its [S#] marker.
        findings = "\n".join(
            f"- {' '.join(r['content'].split())} [S{i}]"
            for i, r in enumerate(state["raw_results"], 1)
        )
        ledger.add(
            "research-agent",
            estimate_tokens(results_text), estimate_tokens(findings),
            estimated=True,
        )
    logger.info("node=synthesize: %d findings lines", findings.count("- "))
    return {"findings": findings, "ledger": ledger.to_dict(),
            "timings": _mark(state, "synthesis_s", t0)}


async def delegate_node(state: ResearchState) -> ResearchState:
    t0 = time.perf_counter()
    if hitl_enabled() and not state.get("critic_feedback"):
        # Human approval gate before autonomy crosses the A2A boundary.
        # interrupt() checkpoints the graph; run_research resumes it with
        # the human's decision.
        from common.audit import record_decision

        decision, source = interrupt({
            "gate": "delegate_to_writer",
            "topic": state["topic"],
            "findings": state["findings"],
        })
        record_decision(state["topic"], decision, state["findings"], source)
        if decision != "approve":
            raise RuntimeError(
                f"HITL: delegation rejected by human reviewer ({source})"
            )
    ledger = Ledger.from_dict(state["ledger"])
    # Kill switch: stop before spending more, not after.
    ledger.check_budget("delegate to the Writer Agent")
    # Semantic routing when a registry is configured; static URL otherwise.
    writer_url = await find_agent(
        "turn research findings into a polished written report",
        WRITER_AGENT_URL,
    )
    logger.info("node=delegate: discovering Writer Agent at %s", writer_url)
    card = await discover_agent(writer_url)
    emit_event("research-agent", "a2a_handoff",
               to=card.name, revision=state.get("revision_rounds", 0))
    feedback = state.get("critic_feedback") or None
    # Bandit: choose a model tier for the writer.
    chosen_tier = BanditRouter().choose()
    emit_event("research-agent", "bandit_choose", tier=chosen_tier)
    if feedback:
        logger.info("node=delegate: REVISION round %d with critic feedback",
                    state.get("revision_rounds", 0))
    findings, results = state["findings"], state["raw_results"]
    try:
        report, structured = await delegate_report(
            card, state["topic"], findings, results, feedback=feedback,
            model_tier=chosen_tier,
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
                f"- {' '.join(r['content'].split())} [S{i}]"
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
    ledger.add(
        "writer-agent",
        usage.get("input_tokens", 0), usage.get("output_tokens", 0),
        estimated=bool(usage.get("estimated")),
    )
    return {
        "report": report,
        "structured_report": structured,
        "raw_results": results,
        "ledger": ledger.to_dict(),
        "model_tier": chosen_tier,
        "writer_usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        },
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
    ledger = Ledger.from_dict(state["ledger"])
    usage = review.get("usage") or {}
    ledger.add(
        "critic-agent",
        usage.get("input_tokens", 0), usage.get("output_tokens", 0),
        estimated=bool(usage.get("estimated")),
    )
    logger.info("node=review: critic verdict=%s", review["verdict"])
    emit_event("critic-agent", "verdict",
               verdict=review["verdict"], feedback=review.get("feedback", ""))

    # Bandit: record reward based on critic verdict.
    # approve → 1.0, revise → 0.3 (partial credit — report was usable).
    tier = state.get("model_tier") or DEFAULT_ARMS[0]["name"]
    reward = 1.0 if review["verdict"] == VERDICT_APPROVE else 0.3
    bandit = BanditRouter()
    # Cost of this pull from the arm's $/MTok spec and the writer's actual
    # token usage for the delegation under review.
    wusage = state.get("writer_usage") or {}
    cost = bandit.estimate_cost(
        tier, wusage.get("input_tokens", 0), wusage.get("output_tokens", 0),
    )
    bandit.record(tier, reward, cost)
    emit_event("research-agent", "bandit_record",
               tier=tier, reward=reward, cost=cost, verdict=review["verdict"])
    timings = _mark(state, f"review_{state.get('revision_rounds', 0)}_s", t0)
    return {
        "critic_verdict": review["verdict"],
        "critic_feedback": review.get("feedback", ""),
        "revision_rounds": state.get("revision_rounds", 0) + 1,
        "ledger": ledger.to_dict(),
        "timings": timings,
    }


def _after_review(state: ResearchState) -> str:
    if (state["critic_verdict"] == VERDICT_REVISE
            and state["revision_rounds"] <= MAX_REVISION_ROUNDS):
        logger.info("reflection: revising report (round %d/%d)",
                    state["revision_rounds"], MAX_REVISION_ROUNDS)
        return "delegate"
    if state["critic_verdict"] == VERDICT_REVISE:
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
    # Checkpointer enables interrupt()/resume for the HITL gate.
    return g.compile(checkpointer=InMemorySaver())


def _prompt_human(payload: dict) -> tuple[str, str]:
    """Ask the operator to approve/reject on stdin; auto-approve (audited)
    when no interactive stdin is available."""
    import sys

    print(
        f"\n--- HUMAN APPROVAL REQUIRED ---\n"
        f"gate:  {payload.get('gate')}\n"
        f"topic: {payload.get('topic')}\n"
        f"findings:\n{payload.get('findings')}\n"
        f"approve delegation? [approve/reject] > ",
        file=sys.stderr, end="", flush=True,
    )
    line = sys.stdin.readline()
    if not line:
        logger.warning("HITL: no interactive stdin — auto-approving (audited)")
        return "approve", "auto-noninteractive"
    decision = line.strip().lower()
    decision = "approve" if decision in ("approve", "a", "yes", "y") else "reject"
    return decision, "interactive"


async def run_research(topic: str, max_results: int = 5) -> ResearchState:
    graph = build_graph()
    logger.info("research pipeline start: topic=%r", topic)
    config = {"configurable": {"thread_id": f"run-{time.time_ns()}"}}
    with tracer().start_as_current_span("research.pipeline") as span:
        span.set_attribute("research.topic", topic)
        final: ResearchState = await graph.ainvoke(
            {"topic": topic, "max_results": max_results,
             "ledger": Ledger().to_dict()},
            config,
        )
        while final.get("__interrupt__"):
            decision = _prompt_human(final["__interrupt__"][0].value)
            logger.info("HITL decision: %s (%s)", *decision)
            final = await graph.ainvoke(Command(resume=decision), config)
        span.set_attribute("research.report_chars", len(final["report"]))
    logger.info("research pipeline done: timings=%s | %s",
                final["timings"], Ledger.from_dict(final["ledger"]).summary())
    return final
