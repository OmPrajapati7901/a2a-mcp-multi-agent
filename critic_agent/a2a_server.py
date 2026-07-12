"""A2A server for the Critic Agent (OpenAI Agents SDK inside).

Publishes its Agent Card, self-registers with the agent registry when one is
configured, and reviews reports delegated to it over A2A.
Run: `uv run python -m critic_agent.a2a_server` (port 9002).
"""
import logging

import uvicorn
from fastapi import FastAPI
from opentelemetry.trace import SpanKind

import a2a.types as ty
from a2a.helpers import new_data_part, new_task_from_user_message
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
    CRITIC_AGENT_BIND,
    CRITIC_AGENT_HOST,
    CRITIC_AGENT_PORT,
    CRITIC_AGENT_URL,
    setup_logging,
)
from common.tracing import extract_context, setup_tracing, tracer
from critic_agent.agent import review_report
from registry.client import self_register

logger = logging.getLogger("critic.a2a")


def build_agent_card() -> ty.AgentCard:
    return ty.AgentCard(
        name="Critic Agent",
        description=(
            "Reviews research reports for faithfulness, citations, and "
            "editorial quality; approves or requests revision with concrete "
            "feedback. Built with the OpenAI Agents SDK, exposed over A2A."
        ),
        version="1.0.0",
        supported_interfaces=[
            ty.AgentInterface(
                url=f"http://{CRITIC_AGENT_HOST}:{CRITIC_AGENT_PORT}/",
                protocol_binding=TransportProtocol.JSONRPC,
            )
        ],
        capabilities=ty.AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            ty.AgentSkill(
                id="review_report",
                name="Review research report",
                description=(
                    "Critique and review a research report against its "
                    "findings: verdict approve or revise with feedback on "
                    "quality, citations, faithfulness."
                ),
                tags=["review", "critique", "editing", "quality"],
            )
        ],
    )


class CriticExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.current_task:
            await event_queue.enqueue_event(new_task_from_user_message(context.message))
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.start_work()

        meta = context.message.metadata
        parent = extract_context({k: meta[k] for k in meta.fields})
        with tracer().start_as_current_span(
            "critic.execute_task", context=parent, kind=SpanKind.SERVER
        ) as span:
            span.set_attribute("a2a.task_id", context.task_id)
            task_text = context.get_user_input()
            logger.info("A2A review task received: task_id=%s input=%d chars",
                        context.task_id, len(task_text))
            try:
                review = await review_report(task_text)
            except Exception:
                logger.exception("critic failed for task %s", context.task_id)
                await updater.failed()
                return
            span.set_attribute("critic.verdict", review["verdict"])
            await updater.add_artifact(
                [ty.Part(text=review["verdict"]), new_data_part(review)],
                name="review",
            )
            await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()


def build_app() -> FastAPI:
    card = build_agent_card()
    handler = DefaultRequestHandler(
        agent_executor=CriticExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    app = FastAPI(title="Critic Agent (A2A)")
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(handler, "/"),
    )

    @app.on_event("startup")
    async def register() -> None:
        import asyncio

        asyncio.get_running_loop().create_task(self_register(CRITIC_AGENT_URL))

    return app


if __name__ == "__main__":
    setup_logging()
    setup_tracing("critic-agent")
    logger.info("starting Critic Agent A2A server on http://%s:%d",
                CRITIC_AGENT_HOST, CRITIC_AGENT_PORT)
    uvicorn.run(build_app(), host=CRITIC_AGENT_BIND, port=CRITIC_AGENT_PORT,
                log_level="warning")
