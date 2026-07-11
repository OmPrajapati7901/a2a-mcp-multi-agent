"""A2A server for the Writer Agent.

Publishes an Agent Card at /.well-known/agent-card.json and accepts task
delegation over A2A JSON-RPC. Run: `uv run python -m writer_agent.a2a_server`.
"""
import json
import logging
import pathlib

import uvicorn
from fastapi import FastAPI
from google.protobuf.json_format import MessageToDict

import a2a.types as ty
from a2a.helpers import new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.utils import TransportProtocol

from common import (
    WRITER_AGENT_BIND,
    WRITER_AGENT_HOST,
    WRITER_AGENT_PORT,
    setup_logging,
)
from writer_agent.agent import write_report

logger = logging.getLogger("writer.a2a")


def build_agent_card() -> ty.AgentCard:
    return ty.AgentCard(
        name="Writer Agent",
        description=(
            "Turns raw research findings into a polished ~300-word report. "
            "Built with Pydantic AI, exposed over the A2A protocol."
        ),
        version="1.0.0",
        supported_interfaces=[
            ty.AgentInterface(
                url=f"http://{WRITER_AGENT_HOST}:{WRITER_AGENT_PORT}/",
                protocol_binding=TransportProtocol.JSONRPC,
            )
        ],
        capabilities=ty.AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            ty.AgentSkill(
                id="write_report",
                name="Write research report",
                description=(
                    "Given a topic and bullet-point findings, produce a "
                    "polished ~300-word report."
                ),
                tags=["writing", "summarization", "report"],
                examples=[
                    "Topic: agent observability\n\nFindings:\n- tracing is "
                    "converging on OpenTelemetry\n- ...",
                ],
            )
        ],
    )


class WriterExecutor(AgentExecutor):
    """Bridges A2A task execution to the Pydantic AI writer agent."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.current_task:
            await event_queue.enqueue_event(new_task_from_user_message(context.message))
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.start_work()

        task_text = context.get_user_input()
        logger.info(
            "A2A task received: task_id=%s context_id=%s input=%d chars",
            context.task_id, context.context_id, len(task_text),
        )
        try:
            report = await write_report(task_text)
        except Exception:
            logger.exception("writer failed for task %s", context.task_id)
            await updater.failed()
            return

        await updater.add_artifact([ty.Part(text=report)], name="report")
        await updater.complete()
        logger.info(
            "A2A task completed: task_id=%s report=%d chars",
            context.task_id, len(report),
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()


def build_app() -> FastAPI:
    card = build_agent_card()

    card_path = pathlib.Path(__file__).parent / "agent_card.json"
    card_path.write_text(json.dumps(MessageToDict(card), indent=2) + "\n")

    handler = DefaultRequestHandler(
        agent_executor=WriterExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    app = FastAPI(title="Writer Agent (A2A)")
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(handler, "/"),
    )
    return app


if __name__ == "__main__":
    setup_logging()
    logger.info(
        "starting Writer Agent A2A server on http://%s:%d "
        "(card at /.well-known/agent-card.json)",
        WRITER_AGENT_HOST, WRITER_AGENT_PORT,
    )
    uvicorn.run(build_app(), host=WRITER_AGENT_BIND, port=WRITER_AGENT_PORT,
                log_level="warning")
