"""Shared configuration and logging for the A2A interop demo."""
import logging
import os
import pathlib
import sys

import certifi
from dotenv import load_dotenv


def offline_forced() -> bool:
    """A2A_DEMO_OFFLINE=1 forces the zero-credential deterministic mode, even
    if keys exist in .env — used by CI and the eval gate for reproducibility."""
    return bool(os.environ.get("A2A_DEMO_OFFLINE"))


# Keys live in .env at the repo root (gitignored), so every process — demo CLI,
# writer server subprocess, MCP server subprocess — picks them up on import.
if not offline_forced():
    load_dotenv(pathlib.Path(__file__).parent.parent / ".env")

# macOS framework Python ships without system CA certs wired up; aiohttp
# (used by langchain-nvidia-ai-endpoints) needs this, httpx bundles certifi.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

# Anthropic model used by both agents when ANTHROPIC_API_KEY is set.
CLAUDE_MODEL = "claude-opus-4-8"

# NVIDIA NIM (OpenAI-compatible) model used when NVIDIA_API_KEY is set.
NVIDIA_MODEL = "z-ai/glm-5.2"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Hostname other processes use to reach the writer (in Docker: the service
# name); BIND is the interface the server listens on (in Docker: 0.0.0.0).
WRITER_AGENT_HOST = os.environ.get("WRITER_AGENT_HOST", "127.0.0.1")
WRITER_AGENT_BIND = os.environ.get("WRITER_AGENT_BIND", WRITER_AGENT_HOST)
WRITER_AGENT_PORT = int(os.environ.get("WRITER_AGENT_PORT", "9001"))
WRITER_AGENT_URL = os.environ.get(
    "WRITER_AGENT_URL", f"http://{WRITER_AGENT_HOST}:{WRITER_AGENT_PORT}"
)


def have_anthropic() -> bool:
    return not offline_forced() and bool(os.environ.get("ANTHROPIC_API_KEY"))


def have_nvidia() -> bool:
    return not offline_forced() and bool(os.environ.get("NVIDIA_API_KEY"))


def have_tavily() -> bool:
    return not offline_forced() and bool(os.environ.get("TAVILY_API_KEY"))


def setup_logging(stream=None) -> None:
    """Consistent log format across processes so the A2A handoff is easy to trace."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-18s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
        stream=stream or sys.stderr,
    )
    # Keep third-party chatter out of the demo trace.
    for noisy in ("httpx", "httpcore", "uvicorn.access", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
