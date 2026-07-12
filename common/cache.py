"""Semantic cache over the whole research pipeline.

Keyed by topic embedding (NVIDIA NIM embeddings in real mode) with cosine
similarity ≥ SIM_THRESHOLD counting as a hit; falls back to normalized
exact-match offline. A hit returns the cached report without spending a
single token or A2A call. Opt-in via A2A_CACHE=1 so evals stay unskewed.
"""
import json
import logging
import math
import os
import pathlib
import re
import sqlite3
import time

from common import have_nvidia

logger = logging.getLogger("cache")

SIM_THRESHOLD = 0.90
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"


def cache_enabled() -> bool:
    return bool(os.environ.get("A2A_CACHE"))


def _db_path() -> pathlib.Path:
    root = pathlib.Path(os.environ.get("A2A_CACHE_DIR", ".cache"))
    root.mkdir(exist_ok=True)
    return root / "semantic_cache.db"


def _normalize(topic: str) -> str:
    return re.sub(r"\s+", " ", topic.lower().strip())


def _embed(topic: str) -> list[float] | None:
    if not have_nvidia():
        return None
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    return NVIDIAEmbeddings(model=EMBED_MODEL).embed_query(topic)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class SemanticCache:
    def __init__(self) -> None:
        self._conn = sqlite3.connect(_db_path())
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS entries ("
            "  topic_norm TEXT, embedding TEXT, payload TEXT, created REAL)"
        )
        self._conn.commit()

    def lookup(self, topic: str) -> dict | None:
        t0 = time.perf_counter()
        rows = self._conn.execute(
            "SELECT topic_norm, embedding, payload FROM entries"
        ).fetchall()
        if not rows:
            logger.info("cache MISS for %r (cache empty)", topic)
            return None
        query_vec = _embed(topic)
        best: tuple[float, str] | None = None
        for topic_norm, emb_json, payload in rows:
            if query_vec is not None and emb_json:
                sim = _cosine(query_vec, json.loads(emb_json))
                if sim >= SIM_THRESHOLD and (best is None or sim > best[0]):
                    best = (sim, payload)
            elif topic_norm == _normalize(topic):
                best = (1.0, payload)
        if best is None:
            logger.info("cache MISS for %r (%d entries)", topic, len(rows))
            return None
        lookup_ms = round((time.perf_counter() - t0) * 1000)
        logger.info("cache HIT for %r (similarity=%.3f, lookup=%dms)",
                    topic, best[0], lookup_ms)
        return json.loads(best[1]) | {"cache_similarity": round(best[0], 3)}

    def store(self, topic: str, payload: dict) -> None:
        vec = _embed(topic)
        self._conn.execute(
            "INSERT INTO entries VALUES (?, ?, ?, ?)",
            (_normalize(topic), json.dumps(vec) if vec else None,
             json.dumps(payload), time.time()),
        )
        self._conn.commit()
        logger.info("cache STORE for %r", topic)

    def close(self) -> None:
        self._conn.close()
