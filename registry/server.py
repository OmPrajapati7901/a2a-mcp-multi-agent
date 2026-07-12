"""Agent registry with semantic capability routing.

Agents self-register their A2A card URL on startup; callers ask for a
capability in plain language ("turn findings into a polished report") and get
the best-matching agent back. Matching uses NIM embeddings over each card's
skill text when available, else deterministic token overlap — so discovery
works offline too.

Run: `uv run python -m registry.server`  (port 9100)
"""
import logging
import math
import re

import httpx
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from common import have_nvidia, setup_logging

logger = logging.getLogger("registry")

import os

REGISTRY_PORT = int(os.environ.get("REGISTRY_PORT", "9100"))

app = FastAPI(title="A2A Agent Registry")

# agent name → {"card": dict, "url": str, "skill_text": str, "vec": list|None}
_agents: dict[str, dict] = {}


class RegisterRequest(BaseModel):
    card_url: str


class FindRequest(BaseModel):
    capability: str


def _skill_text(card: dict) -> str:
    parts = [card.get("name", ""), card.get("description", "")]
    for skill in card.get("skills", []):
        parts += [skill.get("name", ""), skill.get("description", "")]
        parts += skill.get("tags", [])
    return " ".join(p for p in parts if p)


def _embed(text: str) -> list[float] | None:
    if not have_nvidia():
        return None
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    return NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5").embed_query(text)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def _score(capability: str, entry: dict) -> float:
    query_vec = _embed(capability)
    if query_vec is not None and entry.get("vec"):
        dot = sum(x * y for x, y in zip(query_vec, entry["vec"]))
        na = math.sqrt(sum(x * x for x in query_vec))
        nb = math.sqrt(sum(x * x for x in entry["vec"]))
        return dot / (na * nb) if na and nb else 0.0
    # Offline: Jaccard-ish token overlap against the skill text.
    q, s = _tokens(capability), _tokens(entry["skill_text"])
    return len(q & s) / len(q) if q else 0.0


@app.post("/register")
async def register(req: RegisterRequest) -> dict:
    async with httpx.AsyncClient(timeout=5) as client:
        card = (await client.get(
            req.card_url.rstrip("/") + "/.well-known/agent-card.json"
        )).json()
    entry = {
        "card": card,
        "url": req.card_url,
        "skill_text": _skill_text(card),
    }
    entry["vec"] = _embed(entry["skill_text"])
    _agents[card["name"]] = entry
    logger.info("registered agent %r (%s) — %d agent(s) total",
                card["name"], req.card_url, len(_agents))
    return {"registered": card["name"], "agents": len(_agents)}


@app.get("/agents")
async def agents() -> dict:
    return {name: e["url"] for name, e in _agents.items()}


@app.post("/find")
async def find(req: FindRequest) -> dict:
    if not _agents:
        return {"match": None}
    scored = sorted(
        ((name, _score(req.capability, e)) for name, e in _agents.items()),
        key=lambda x: -x[1],
    )
    best_name, best_score = scored[0]
    logger.info("semantic route %r → %r (score=%.3f; candidates=%s)",
                req.capability, best_name,
                best_score, [(n, round(s, 3)) for n, s in scored])
    return {
        "match": best_name,
        "url": _agents[best_name]["url"],
        "score": round(best_score, 3),
        "candidates": {n: round(s, 3) for n, s in scored},
    }


if __name__ == "__main__":
    setup_logging()
    logger.info("starting agent registry on http://127.0.0.1:%d", REGISTRY_PORT)
    uvicorn.run(app, host="127.0.0.1", port=REGISTRY_PORT, log_level="warning")
