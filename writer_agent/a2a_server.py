"""A2A server for the Writer Agent.

Publishes an Agent Card at /.well-known/agent-card.json and accepts task
delegation over A2A JSON-RPC. Run: `uv run python -m writer_agent.a2a_server`.
"""
import json
import logging
import os
import pathlib

import uvicorn
from fastapi import FastAPI
from google.protobuf.json_format import MessageToDict
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
    WRITER_AGENT_BIND,
    WRITER_AGENT_HOST,
    WRITER_AGENT_PORT,
    setup_logging,
)
from common import WRITER_AGENT_URL
from common.report import parse_report, parse_sources
from common.tracing import extract_context, setup_tracing, tracer
from registry.client import self_register
from writer_agent.agent import write_report

logger = logging.getLogger("writer.a2a")


def build_agent_card() -> ty.AgentCard:
    card = _base_card()
    if os.environ.get("A2A_API_KEY"):
        # Declare the scheme on the card so callers know how to authenticate.
        card.security_schemes["api-key"].api_key_security_scheme.CopyFrom(
            ty.APIKeySecurityScheme(
                name="X-API-Key", location="header",
                description="Static API key issued to trusted caller agents.",
            )
        )
    return card


def _base_card() -> ty.AgentCard:
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

    def __init__(self) -> None:
        # Chaos injection for resilience tests: fail the first N tasks.
        self._chaos_failures_left = int(os.environ.get("A2A_CHAOS_FAIL_N", "0"))

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.current_task:
            await event_queue.enqueue_event(new_task_from_user_message(context.message))
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.start_work()

        # The caller's trace context arrives in the A2A message metadata, so
        # this span (and the LLM spans under it) join the caller's trace.
        meta = context.message.metadata
        parent = extract_context({k: meta[k] for k in meta.fields})
        with tracer().start_as_current_span(
            "writer.execute_task", context=parent, kind=SpanKind.SERVER
        ) as span:
            span.set_attribute("a2a.task_id", context.task_id)
            if self._chaos_failures_left > 0:
                self._chaos_failures_left -= 1
                logger.warning(
                    "CHAOS: injected failure for task %s (%d more to come)",
                    context.task_id, self._chaos_failures_left,
                )
                await updater.failed()
                return
            task_text = context.get_user_input()
            logger.info(
                "A2A task received: task_id=%s context_id=%s input=%d chars",
                context.task_id, context.context_id, len(task_text),
            )
            artifact_id = f"report-{context.task_id}"
            chunks_sent = 0

            async def emit(chunk: str) -> None:
                nonlocal chunks_sent
                await updater.add_artifact(
                    [ty.Part(text=chunk)],
                    artifact_id=artifact_id, name="report",
                    append=chunks_sent > 0,
                )
                chunks_sent += 1

            try:
                report, usage = await write_report(task_text, emit=emit)
            except Exception:
                logger.exception("writer failed for task %s", context.task_id)
                await updater.failed()
                return

            structured = parse_report(report, parse_sources(task_text))
            span.set_attribute("report.citations", len(structured.citations))
            # Final chunk: the structured report (citations + usage) as a
            # data part appended to the same artifact.
            await updater.add_artifact(
                [new_data_part(structured.model_dump() | {"usage": usage})],
                artifact_id=artifact_id, name="report",
                append=True, last_chunk=True,
            )
            logger.info("A2A artifact streamed in %d chunk(s)", chunks_sent + 1)
            await updater.complete()
            logger.info(
                "A2A task completed: task_id=%s report=%d chars, %d/%d sources cited",
                context.task_id, len(report),
                len(structured.citations), structured.sources_available,
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

    api_key = os.environ.get("A2A_API_KEY")
    if api_key:
        from fastapi.responses import JSONResponse

        @app.middleware("http")
        async def require_api_key(request, call_next):
            # The Agent Card stays public — that's how callers discover the
            # required scheme; every A2A RPC needs the key.
            if request.url.path.startswith("/.well-known"):
                return await call_next(request)
            if request.headers.get("x-api-key") != api_key:
                logger.warning("A2A AUTH: rejected request to %s (bad/missing key)",
                               request.url.path)
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await call_next(request)

    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(handler, "/"),
    )
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except ImportError:
        pass

    @app.on_event("startup")
    async def register() -> None:
        import asyncio

        asyncio.get_running_loop().create_task(self_register(WRITER_AGENT_URL))

    return app


if __name__ == "__main__":
    setup_logging()
    setup_tracing("writer-agent")
    logger.info(
        "starting Writer Agent A2A server on http://%s:%d "
        "(card at /.well-known/agent-card.json)",
        WRITER_AGENT_HOST, WRITER_AGENT_PORT,
    )
    uvicorn.run(build_app(), host=WRITER_AGENT_BIND, port=WRITER_AGENT_PORT,
                log_level="warning")
