"""Shared NIM embedding helpers for the semantic cache and agent registry.

`embed()` returns None when no NVIDIA key is available (or offline mode is
forced) — callers fall back to their lexical matching paths, so everything
keeps working with zero credentials.
"""
import math

from common import have_nvidia

EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"

_client = None


def embed(text: str) -> list[float] | None:
    global _client
    if not have_nvidia():
        return None
    if _client is None:
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

        _client = NVIDIAEmbeddings(model=EMBED_MODEL)
    return _client.embed_query(text)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0
