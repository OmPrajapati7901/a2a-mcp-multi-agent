"""Client helpers for the agent registry: self-registration and semantic
capability lookup. Everything degrades gracefully — no registry means agents
fall back to their statically configured URLs."""
import asyncio
import logging

import httpx

from common import registry_url

logger = logging.getLogger("registry.client")

CARD_PATH = "/.well-known/agent-card.json"


async def self_register(agent_url: str) -> None:
    """Called by agent servers on startup; best-effort. Runs as a background
    task: FastAPI's startup hook fires before the server accepts connections,
    so we wait for our own card endpoint to answer before telling the
    registry to fetch it."""
    url = registry_url()
    if not url:
        return
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            for _ in range(50):  # wait for our own server to be reachable
                try:
                    if (await client.get(agent_url.rstrip("/") + CARD_PATH)
                            ).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.2)
            resp = await client.post(url.rstrip("/") + "/register",
                                     json={"card_url": agent_url})
            resp.raise_for_status()
        logger.info("registered %s with registry %s", agent_url, url)
    except httpx.HTTPError as exc:
        logger.warning("registry registration failed (continuing): %s", exc)


async def find_agent(capability: str, fallback_url: str) -> str:
    """Semantic capability lookup; returns the fallback URL when the registry
    is absent, unreachable, or has no confident match."""
    url = registry_url()
    if not url:
        return fallback_url
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(url.rstrip("/") + "/find",
                                     json={"capability": capability})
            resp.raise_for_status()
            data = resp.json()
        if data.get("match") and data.get("score", 0) >= 0.3:
            logger.info(
                "REGISTRY ROUTE: %r → %r at %s (score=%s)",
                capability, data["match"], data["url"], data["score"],
            )
            return data["url"]
        logger.warning("registry had no confident match for %r (got %s) — "
                       "using fallback", capability, data)
    except httpx.HTTPError as exc:
        logger.warning("registry lookup failed (%s) — using fallback", exc)
    return fallback_url
