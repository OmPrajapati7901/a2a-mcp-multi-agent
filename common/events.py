"""Fire-and-forget event emission to the live dashboard.

Opt-in via A2A_DASHBOARD_URL; failures never affect the pipeline. Events are
small JSON blobs: who did what, when — enough for a live topology view.
"""
import asyncio
import datetime
import logging
import os

import httpx

logger = logging.getLogger("events")


def dashboard_url() -> str | None:
    return os.environ.get("A2A_DASHBOARD_URL") or None


async def _post(event: dict) -> None:
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            await client.post(dashboard_url() + "/event", json=event)
    except httpx.HTTPError:
        pass  # dashboard is observability, never a dependency


def emit_event(agent: str, kind: str, **detail) -> None:
    """Schedule an event post on the running loop; no-op without a dashboard."""
    if not dashboard_url():
        return
    event = {
        "ts": datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "agent": agent,
        "kind": kind,
        "detail": detail,
    }
    try:
        asyncio.get_running_loop().create_task(_post(event))
    except RuntimeError:
        pass  # no loop (sync context) — skip rather than block
