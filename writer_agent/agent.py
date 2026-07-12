"""Writer Agent — Pydantic AI.

Turns research findings into a polished ~300-word report. Uses Claude when
ANTHROPIC_API_KEY is set; otherwise a deterministic FunctionModel stands in so
the full A2A pipeline is demoable offline. Either way the same Pydantic AI
Agent object and prompt are exercised — only the model is swapped.
"""
import logging
import os
import re

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from common import (
    CLAUDE_MODEL,
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
    have_anthropic,
    have_nvidia,
)

logger = logging.getLogger("writer.agent")

SYSTEM_PROMPT = (
    "You are a professional research writer. You receive a topic, raw "
    "research findings, and a numbered source list ([S1], [S2], …). Write a "
    "polished report of roughly 300 words: a title, a short introduction, "
    "2-3 body paragraphs synthesizing the findings, and a one-sentence "
    "takeaway. Every factual claim MUST carry an inline [S#] citation "
    "matching the source it came from. Plain text, no markdown headings "
    "other than the title line."
)


def _offline_report(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Deterministic stand-in model: composes a structured report from the
    findings bullets in the prompt, without calling any LLM."""
    prompt = ""
    for part in messages[-1].parts:
        if hasattr(part, "content") and isinstance(part.content, str):
            prompt += part.content
    topic_match = re.search(r"Topic:\s*(.+)", prompt)
    topic = topic_match.group(1).strip() if topic_match else "the requested topic"
    bullets = re.findall(r"^- (.+)$", prompt, flags=re.MULTILINE)

    body = " ".join(bullets) if bullets else prompt[:600]
    report = (
        f"{topic.title()}: A Brief Report\n\n"
        f"This report summarizes current findings on {topic}. "
        f"The research surfaced {len(bullets) or 'several'} key threads, "
        "reviewed below.\n\n"
        f"{body}\n\n"
        "Taken together, these findings suggest the space is consolidating "
        "around shared open standards, and teams adopting them early will "
        "integrate more easily as the ecosystem matures.\n\n"
        "[offline mode: deterministic report — set an LLM API key for a "
        "real model-written report]"
    )
    return ModelResponse(parts=[TextPart(report)])


def build_writer_agent() -> Agent:
    if have_anthropic():
        model = f"anthropic:{CLAUDE_MODEL}"
        logger.info("writer model: %s", model)
    elif have_nvidia():
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        model = OpenAIChatModel(
            NVIDIA_MODEL,
            provider=OpenAIProvider(
                base_url=NVIDIA_BASE_URL,
                api_key=os.environ["NVIDIA_API_KEY"],
            ),
        )
        logger.info("writer model: %s via NVIDIA NIM", NVIDIA_MODEL)
    else:
        model = FunctionModel(_offline_report)
        logger.info("writer model: offline FunctionModel (no LLM API key)")
    return Agent(model, system_prompt=SYSTEM_PROMPT)


_agent: Agent | None = None

# Flush streamed text to the A2A artifact roughly per paragraph.
STREAM_CHUNK_CHARS = 300


def _get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = build_writer_agent()
    return _agent


def _usage_dict(usage) -> dict:
    return {
        "input_tokens": usage.input_tokens or 0,
        "output_tokens": usage.output_tokens or 0,
        "estimated": not have_anthropic() and not have_nvidia(),
    }


async def write_report(task_text: str, emit=None) -> tuple[str, dict]:
    """Entry point used by the A2A executor: findings text in, report out,
    plus this agent's token usage so the caller can attribute cost.

    When `emit` (async fn(chunk: str)) is given and a real model is in use,
    the report is streamed to it in chunks as the model generates."""
    agent = _get_agent()
    if emit is not None and (have_anthropic() or have_nvidia()):
        async with agent.run_stream(task_text) as result:
            buffer = ""
            async for delta in result.stream_text(delta=True):
                buffer += delta
                if len(buffer) >= STREAM_CHUNK_CHARS:
                    await emit(buffer)
                    buffer = ""
            if buffer:
                await emit(buffer)
            return await result.get_output(), _usage_dict(result.usage)
    result = await agent.run(task_text)
    if emit is not None:
        await emit(result.output)
    return result.output, _usage_dict(result.usage)
