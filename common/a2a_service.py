"""Shared FastAPI scaffolding for A2A agent servers.

Every agent server does the same four things: publish its Agent Card, serve
A2A JSON-RPC, enforce the static API key when one is configured, and
self-register with the agent registry on startup. This module owns that
scaffolding — and the auth boundary, so it is identical on every agent —
leaving each agent as just a card plus an executor.
"""
import logging
import os

import uvicorn
from fastapi import FastAPI

import a2a.types as ty
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore

from common import setup_logging
from common.tracing import setup_tracing
from registry.client import self_register

logger = logging.getLogger("a2a.service")


def declare_api_key_scheme(card: ty.AgentCard) -> ty.AgentCard:
    """When auth is enabled, declare the API-key scheme on the card so
    callers know how to authenticate."""
    if os.environ.get("A2A_API_KEY"):
        card.security_schemes["api-key"].api_key_security_scheme.CopyFrom(
            ty.APIKeySecurityScheme(
                name="X-API-Key", location="header",
                description="Static API key issued to trusted caller agents.",
            )
        )
    return card


def build_a2a_app(
    card: ty.AgentCard, executor: AgentExecutor, *, self_register_url: str
) -> FastAPI:
    """Standard A2A agent app: card + JSON-RPC routes, API-key middleware
    when A2A_API_KEY is set, tracing instrumentation, and registry
    self-registration on startup."""
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    app = FastAPI(title=f"{card.name} (A2A)")

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
                logger.warning(
                    "A2A AUTH: rejected request to %s (bad/missing key)",
                    request.url.path,
                )
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

        asyncio.get_running_loop().create_task(self_register(self_register_url))

    return app


def serve(build_app, service_name: str, host: str, bind: str, port: int) -> None:
    """Entry-point runner shared by the agent servers' __main__ blocks.
    Takes the app factory (not the app) so tracing is installed before the
    app is built and instrumented."""
    setup_logging()
    setup_tracing(service_name)
    logger.info(
        "starting %s A2A server on http://%s:%d "
        "(card at /.well-known/agent-card.json)",
        service_name, host, port,
    )
    uvicorn.run(build_app(), host=bind, port=port, log_level="warning")
