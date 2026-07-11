# A2A + MCP Multi-Agent Interop Demo

Two agents on **different frameworks** — a LangGraph Research Agent and a
Pydantic AI Writer Agent — that discover each other and delegate work over the
**A2A (Agent2Agent) protocol**, with **MCP** used for tool access.

```
User
  │
  ▼
Research Agent (LangGraph)
  │   MCP (stdio) → web_search tool (Tavily, or mock offline)
  │
  │   A2A: discovers Writer Agent via /.well-known/agent-card.json
  │   A2A: delegates task → "write report from findings"
  ▼
Writer Agent (Pydantic AI, A2A server on :9001)
  │   returns polished ~300-word report as an A2A artifact
  ▼
Research Agent → final report to user
```

The two protocols draw two different boundaries on purpose: **MCP is the
agent→tool seam** (the Research Agent doesn't know or care how search is
implemented), and **A2A is the agent→agent seam** (the Research Agent knows
only the Writer's published Agent Card — name, skills, transport — not that
it's Pydantic AI under the hood). Either side can be swapped without touching
the other.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). Python and all dependencies are
resolved from `pyproject.toml` / `uv.lock`.

```bash
uv sync
uv run python run_demo.py "Summarize recent developments in agent observability tools"
```

That's it — `run_demo.py` boots the Writer Agent's A2A server, runs the
research pipeline, prints the report and per-phase latency metrics, and shuts
the server down.

### Modes

Keys are read from a gitignored `.env` at the repo root (or the environment).

| Env var | Set | Unset |
|---|---|---|
| `ANTHROPIC_API_KEY` | Both agents use Claude (`claude-opus-4-8`) | fall through ↓ |
| `NVIDIA_API_KEY` | Both agents use `z-ai/glm-5.2` via NVIDIA NIM (OpenAI-compatible endpoint) | Deterministic offline models (same agent code paths) |
| `TAVILY_API_KEY` | Real web search via Tavily | Canned mock results |

The LLM seam shows off the interop story twice over: the Research Agent
reaches the model through LangChain (`ChatAnthropic` / `ChatNVIDIA`), the
Writer through Pydantic AI (`anthropic:` provider / `OpenAIChatModel` pointed
at NIM's OpenAI-compatible base URL) — and offline mode swaps in a
deterministic `FunctionModel` so the full protocol pipeline (MCP call, A2A
discovery, delegation, artifact streaming) is demoable with zero credentials.

### Running the agents separately

```bash
# Terminal 1 — Writer Agent A2A server
uv run python -m writer_agent.a2a_server

# Inspect its Agent Card
curl http://127.0.0.1:9001/.well-known/agent-card.json | jq

# Terminal 2 — demo reuses the running server
uv run python run_demo.py "your topic"
```

## Where the A2A handoff happens

Watch the log trace (all timestamps stderr):

```
research.a2a | A2A DISCOVERY: found agent 'Writer Agent' v1.0.0 ... skills: ['write_report']
research.a2a | A2A HANDOFF: delegating 'write_report' to 'Writer Agent' ...
writer.a2a   | A2A task received: task_id=... input=1038 chars
research.a2a | A2A task status: TASK_STATE_WORKING
research.a2a | A2A artifact received: 'report' (1460 chars)
research.a2a | A2A task status: TASK_STATE_COMPLETED
```

`research_agent/a2a_client.py` resolves the card and streams the task;
`writer_agent/a2a_server.py`'s `WriterExecutor` bridges the A2A task lifecycle
(submitted → working → artifact → completed) to the Pydantic AI agent.

## Repo layout

```
├── run_demo.py               # CLI: topic in → report out (+ metrics)
├── common/                   # shared config + logging
├── mcp_server/
│   └── search_server.py      # MCP stdio server: web_search (Tavily or mock)
├── research_agent/           # LangGraph
│   ├── agent.py              # graph: search → synthesize → delegate
│   ├── mcp_client.py         # MCP stdio client (agent→tool)
│   └── a2a_client.py         # A2A discovery + delegation (agent→agent)
└── writer_agent/             # Pydantic AI
    ├── agent.py              # writer agent + offline FunctionModel fallback
    ├── a2a_server.py         # FastAPI + a2a-sdk routes, port 9001
    └── agent_card.json       # generated snapshot of the served Agent Card
```

## Stack

- Python 3.14, managed with **uv**
- **LangGraph** — Research Agent
- **Pydantic AI** — Writer Agent
- **a2a-sdk 1.x** (official A2A Python SDK; protobuf-typed, JSON-RPC transport,
  card at `/.well-known/agent-card.json`)
- **MCP Python SDK** (FastMCP server over stdio) + Tavily Search API

## Metrics captured per run

Printed after every run: end-to-end latency, per-phase latency (MCP search /
synthesis / A2A delegation), and search result count. Example offline run:
`total=1.29s | search=0.44s | synthesis=0.0s | a2a_delegation=0.05s`.

## Next steps (deliberately out of scope for v1)

- Multi-turn negotiation between agents (A2A `input_required` state)
- More than two agents / a registry of Agent Cards
- Auth on the A2A calls (Agent Card `security_schemes`)
- Tracing via Arize Phoenix or Langfuse (OpenInference spans on both seams)
- Production hosting
