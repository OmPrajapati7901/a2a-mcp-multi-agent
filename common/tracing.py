"""Distributed tracing across every seam of the pipeline.

One trace spans three services and two protocols:

    research-agent ──(traceparent in spawn env)──▶ mcp-web-search
    research-agent ──(traceparent in A2A message metadata)──▶ writer-agent

Export targets, in order of precedence:
  - PHOENIX_COLLECTOR_ENDPOINT / OTEL_EXPORTER_OTLP_ENDPOINT → OTLP HTTP
    (e.g. Arize Phoenix at http://localhost:6006)
  - A2A_DEMO_TRACE_CONSOLE=1 → spans printed to stderr (used by tests)
  - neither → tracing is a no-op
"""
import atexit
import os
import sys

from opentelemetry import propagate, trace

_TRACER_NAME = "a2a-mcp-demo"


def tracing_enabled() -> bool:
    return bool(
        os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        or os.environ.get("A2A_DEMO_TRACE_CONSOLE")
    )


def setup_tracing(service_name: str) -> None:
    """Install a real TracerProvider for this process, or leave the no-op one."""
    if not tracing_enabled():
        return

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name})
    )
    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT") or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        exporter = OTLPSpanExporter(
            endpoint=endpoint.rstrip("/") + "/v1/traces"
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
    else:
        provider.add_span_processor(
            SimpleSpanProcessor(ConsoleSpanExporter(out=sys.stderr))
        )
    trace.set_tracer_provider(provider)
    atexit.register(provider.shutdown)

    # LLM-call spans (OpenInference conventions, renders richly in Phoenix).
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor

        LangChainInstrumentor().instrument(tracer_provider=provider)
    except ImportError:
        pass
    try:
        from pydantic_ai import Agent

        Agent.instrument_all()
    except ImportError:
        pass


def tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME)


def inject_context(carrier: dict) -> dict:
    """Serialize the current span context (W3C traceparent) into a dict —
    carried in A2A message metadata or a subprocess environment."""
    propagate.inject(carrier)
    return carrier


def extract_context(carrier: dict):
    """Recover a remote parent context on the receiving side of a boundary."""
    return propagate.extract({k.lower(): v for k, v in carrier.items() if v})
