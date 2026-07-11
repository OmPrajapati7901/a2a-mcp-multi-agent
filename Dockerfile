FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev

EXPOSE 9001

# Default: run the Writer Agent A2A server; compose overrides for the demo.
CMD ["uv", "run", "--no-sync", "python", "-m", "writer_agent.a2a_server"]
