"""A2A client for the Research Agent.

Discovers the Writer Agent via its published Agent Card and delegates the
"write report" task over the A2A protocol. This is the agent→agent handoff —
the seam where LangGraph ends and Pydantic AI begins.
"""
import logging

import httpx

import a2a.types as ty
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_artifact_text, new_text_message

logger = logging.getLogger("research.a2a")

# Report writing with a reasoning model can take minutes; don't let the
# A2A stream time out under it.
DELEGATION_TIMEOUT = httpx.Timeout(600, connect=10)


async def discover_writer(base_url: str) -> ty.AgentCard:
    """Fetch and log the Writer Agent's card from /.well-known/agent-card.json."""
    async with httpx.AsyncClient(timeout=10) as http:
        card = await A2ACardResolver(http, base_url).get_agent_card()
    logger.info(
        "A2A DISCOVERY: found agent %r v%s at %s — skills: %s",
        card.name, card.version, base_url,
        [s.id for s in card.skills],
    )
    return card


async def delegate_report(card: ty.AgentCard, topic: str, findings: str) -> str:
    """Send the writing task to the Writer Agent over A2A and collect the report."""
    task_text = f"Topic: {topic}\n\nFindings:\n{findings}"
    http = httpx.AsyncClient(timeout=DELEGATION_TIMEOUT)
    client = await create_client(card, ClientConfig(httpx_client=http))
    logger.info(
        "A2A HANDOFF: delegating 'write_report' to %r (%d chars of findings)",
        card.name, len(task_text),
    )
    report_chunks: list[str] = []
    task_id = None
    try:
        req = ty.SendMessageRequest(
            message=new_text_message(task_text, role=ty.Role.ROLE_USER)
        )
        async for resp in client.send_message(req):
            kind = resp.WhichOneof("payload")
            if kind == "task":
                task_id = resp.task.id
                logger.info("A2A task created: id=%s", task_id)
            elif kind == "status_update":
                state = ty.TaskState.Name(resp.status_update.status.state)
                logger.info("A2A task status: %s", state)
                if resp.status_update.status.state == ty.TaskState.TASK_STATE_FAILED:
                    raise RuntimeError(f"Writer Agent task {task_id} failed")
            elif kind == "artifact_update":
                artifact = resp.artifact_update.artifact
                report_chunks.append(get_artifact_text(artifact))
                logger.info(
                    "A2A artifact received: %r (%d chars)",
                    artifact.name, len(report_chunks[-1]),
                )
    finally:
        await client.close()
        await http.aclose()

    if not report_chunks:
        raise RuntimeError("Writer Agent returned no report artifact")
    report = "\n".join(report_chunks)
    logger.info("A2A HANDOFF complete: report received (%d chars)", len(report))
    return report
