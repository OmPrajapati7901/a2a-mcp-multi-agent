"""A2A client for the Research Agent.

Discovers the Writer Agent via its published Agent Card and delegates the
"write report" task over the A2A protocol. This is the agent→agent handoff —
the seam where LangGraph ends and Pydantic AI begins.
"""
import logging
import os

import httpx

import a2a.types as ty
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_artifact_text, get_data_parts, new_text_message
from opentelemetry.trace import SpanKind

from a2a.client.errors import A2AClientError

from common.report import format_sources
from common.resilience import CircuitBreaker, with_retries
from common.tracing import inject_context, tracer

logger = logging.getLogger("research.a2a")

# One breaker per remote agent: repeated delegation failures stop further
# attempts instead of hammering a downed agent.
WRITER_BREAKER = CircuitBreaker("writer-agent", failure_threshold=3, cooldown_s=30)
CRITIC_BREAKER = CircuitBreaker("critic-agent", failure_threshold=3, cooldown_s=30)

# Report writing with a reasoning model can take minutes; don't let the
# A2A stream time out under it.
DELEGATION_TIMEOUT = httpx.Timeout(600, connect=10)


async def _inject_trace_headers(request: httpx.Request) -> None:
    """W3C traceparent on the HTTP layer too, so the A2A SDK's own server
    spans join the same trace as ours."""
    inject_context(request.headers)


async def discover_agent(base_url: str) -> ty.AgentCard:
    """Fetch and log an agent's card from /.well-known/agent-card.json."""
    async with httpx.AsyncClient(timeout=10) as http:
        card = await A2ACardResolver(http, base_url).get_agent_card()
    logger.info(
        "A2A DISCOVERY: found agent %r v%s at %s — skills: %s",
        card.name, card.version, base_url,
        [s.id for s in card.skills],
    )
    return card


# Backwards-compatible alias.
discover_writer = discover_agent


async def delegate_report(
    card: ty.AgentCard, topic: str, findings: str, sources: list[dict],
    feedback: str | None = None,
) -> tuple[str, dict | None]:
    """Send the writing task to the Writer Agent over A2A, with bounded
    retries behind a circuit breaker. Returns the report text plus the
    structured report (citations) from the artifact's data part."""
    task_text = (
        f"Topic: {topic}\n\nFindings:\n{findings}\n\n"
        f"Sources:\n{format_sources(sources)}"
    )
    if feedback:
        task_text += f"\n\nReviewer feedback to address:\n{feedback}"
    return await with_retries(
        lambda: _run_task(card, task_text, skill="write_report"),
        name="a2a.delegate(write_report)",
        attempts=3,
        retry_on=(A2AClientError, httpx.HTTPError, RuntimeError),
        breaker=WRITER_BREAKER,
    )


async def delegate_review(
    card: ty.AgentCard, report: str, findings: str
) -> dict:
    """Delegate report review to the Critic Agent over A2A; returns the
    structured review ({verdict, feedback, usage}) from its data part."""
    task_text = f"Findings:\n{findings}\n\nReport:\n{report}"
    _, review = await with_retries(
        lambda: _run_task(card, task_text, skill="review_report"),
        name="a2a.delegate(review_report)",
        attempts=3,
        retry_on=(A2AClientError, httpx.HTTPError, RuntimeError),
        breaker=CRITIC_BREAKER,
    )
    if not review or "verdict" not in review:
        raise RuntimeError("Critic Agent returned no structured review")
    return review


async def _run_task(
    card: ty.AgentCard, task_text: str, skill: str
) -> tuple[str, dict | None]:
    headers = {}
    if os.environ.get("A2A_API_KEY"):
        # Card declares an api-key scheme (X-API-Key header); present it.
        headers["X-API-Key"] = os.environ["A2A_API_KEY"]
    http = httpx.AsyncClient(
        timeout=DELEGATION_TIMEOUT,
        headers=headers,
        event_hooks={"request": [_inject_trace_headers]},
    )
    client = await create_client(card, ClientConfig(httpx_client=http))
    logger.info(
        "A2A HANDOFF: delegating %r to %r (%d chars of task text)",
        skill, card.name, len(task_text),
    )
    report_text = ""
    chunks_received = 0
    structured: dict | None = None
    task_id = None
    span_cm = tracer().start_as_current_span("a2a.delegate", kind=SpanKind.CLIENT)
    try:
        with span_cm as span:
            span.set_attribute("a2a.remote_agent", card.name)
            span.set_attribute("a2a.skill", skill)
            message = new_text_message(task_text, role=ty.Role.ROLE_USER)
            # Trace context crosses the A2A boundary in the message metadata,
            # so the writer's spans join this trace (W3C traceparent).
            message.metadata.update(inject_context({}))
            req = ty.SendMessageRequest(message=message)
            async for resp in client.send_message(req):
                kind = resp.WhichOneof("payload")
                if kind == "task":
                    task_id = resp.task.id
                    span.set_attribute("a2a.task_id", task_id)
                    logger.info("A2A task created: id=%s", task_id)
                elif kind == "status_update":
                    state = ty.TaskState.Name(resp.status_update.status.state)
                    logger.info("A2A task status: %s", state)
                    if resp.status_update.status.state == ty.TaskState.TASK_STATE_FAILED:
                        raise RuntimeError(
                            f"{card.name} task {task_id} failed"
                        )
                elif kind == "artifact_update":
                    ev = resp.artifact_update
                    text = get_artifact_text(ev.artifact)
                    report_text = report_text + text if ev.append else text
                    chunks_received += 1
                    data_parts = get_data_parts(ev.artifact.parts)
                    if data_parts:
                        structured = data_parts[0]
                    if ev.last_chunk or not ev.append:
                        logger.info(
                            "A2A artifact %s: %r (%d chars in %d chunk(s), "
                            "%d citations)",
                            "complete" if ev.last_chunk else "received",
                            ev.artifact.name, len(report_text), chunks_received,
                            len((structured or {}).get("citations", [])),
                        )
    finally:
        await client.close()
        await http.aclose()

    if not report_text:
        raise RuntimeError(f"{card.name} returned no artifact")
    logger.info("A2A HANDOFF complete: %r result received (%d chars)",
                skill, len(report_text))
    return report_text, structured
