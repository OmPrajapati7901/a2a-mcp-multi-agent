"""Critic Agent — OpenAI Agents SDK (third framework in the mesh).

Reviews a report against the findings it was written from and returns a
verdict: approve, or revise with actionable feedback. Real mode drives the
OpenAI Agents SDK against NVIDIA NIM's OpenAI-compatible endpoint; offline
mode applies deterministic quality checks so the reflection loop is testable
with zero credentials.
"""
import json
import logging
import os
import re

from common import NVIDIA_BASE_URL, NVIDIA_MODEL, have_nvidia

logger = logging.getLogger("critic.agent")

INSTRUCTIONS = (
    "You are an exacting editorial reviewer. You receive a research report "
    "and the findings it was written from. Approve only if the report is "
    "faithful to the findings, cites sources inline as [S#], has a clear "
    "title, and is roughly 300 words. Reply with ONLY a JSON object: "
    '{"verdict": "approve" | "revise", "feedback": "<one concrete, '
    'actionable sentence if revising, else empty>"}'
)

_openai_agent = None


def _force_revise() -> bool:
    """Test/demo hook: force the first review into a revise verdict."""
    n = int(os.environ.get("A2A_CRITIC_FORCE_REVISE", "0"))
    if n > 0:
        os.environ["A2A_CRITIC_FORCE_REVISE"] = str(n - 1)
        return True
    return False


def _offline_review(task_text: str) -> dict:
    report = task_text.split("Report:", 1)[-1]
    problems = []
    if not re.search(r"\[S\d+\]", report):
        problems.append("add inline [S#] citations for each claim")
    if len(report.split()) < 80:
        problems.append("expand the report toward ~300 words")
    lines = [ln for ln in report.strip().splitlines() if ln.strip()]
    if not lines or len(lines[0].split()) > 20:
        problems.append("start with a concise title line")
    if problems:
        return {"verdict": "revise", "feedback": "; ".join(problems)}
    return {"verdict": "approve", "feedback": ""}


async def review_report(task_text: str) -> dict:
    """Returns {"verdict", "feedback", "usage"}."""
    if _force_revise():
        logger.info("critic verdict: revise (forced via A2A_CRITIC_FORCE_REVISE)")
        return {
            "verdict": "revise",
            "feedback": "tighten the introduction and cite every claim inline",
            "usage": {"input_tokens": 0, "output_tokens": 0, "estimated": True},
        }

    if have_nvidia():
        global _openai_agent
        from agents import (
            Agent as OpenAIAgent,
            OpenAIChatCompletionsModel,
            Runner,
            set_tracing_disabled,
        )
        from openai import AsyncOpenAI

        if _openai_agent is None:
            set_tracing_disabled(True)  # no OpenAI-platform trace upload
            _openai_agent = OpenAIAgent(
                name="Critic",
                instructions=INSTRUCTIONS,
                model=OpenAIChatCompletionsModel(
                    model=NVIDIA_MODEL,
                    openai_client=AsyncOpenAI(
                        base_url=NVIDIA_BASE_URL,
                        api_key=os.environ["NVIDIA_API_KEY"],
                    ),
                ),
            )
            logger.info("critic model: %s via NVIDIA NIM (OpenAI Agents SDK)",
                        NVIDIA_MODEL)
        result = await Runner.run(_openai_agent, task_text)
        raw = str(result.final_output)
        match = re.search(r"\{[^{}]*\}", raw)
        try:
            parsed = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            parsed = {}
        verdict = parsed.get("verdict", "approve")
        usage = result.context_wrapper.usage
        review = {
            "verdict": verdict if verdict in ("approve", "revise") else "approve",
            "feedback": str(parsed.get("feedback", ""))[:500],
            "usage": {
                "input_tokens": usage.input_tokens or 0,
                "output_tokens": usage.output_tokens or 0,
                "estimated": False,
            },
        }
    else:
        review = _offline_review(task_text) | {
            "usage": {"input_tokens": 0, "output_tokens": 0, "estimated": True}
        }
    logger.info("critic verdict: %s%s", review["verdict"],
                f" — {review['feedback']}" if review["feedback"] else "")
    return review
