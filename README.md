# A2A + MCP Multi-Agent Interop Demo

[![ci](https://github.com/OmPrajapati7901/a2a-mcp-multi-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/OmPrajapati7901/a2a-mcp-multi-agent/actions/workflows/ci.yml)

Two agents on **different frameworks** — a LangGraph Research Agent and a
Pydantic AI Writer Agent — that discover each other and delegate work over the
**A2A (Agent2Agent) protocol**, with **MCP** used for tool access. Traced,
evaluated, tested in CI, and measured.

```
User
  │
  ▼
Research Agent (LangGraph)
  │   MCP (stdio) → web_search tool (Tavily, or mock offline)
  │
  │   A2A: discovers Writer Agent via /.well-known/agent-card.json
  │   A2A: delegates task → "write report from findings" (+ numbered sources)
  ▼
Writer Agent (Pydantic AI, A2A server on :9001)
  │   returns ~300-word report with [S#] citations
  │   as an A2A artifact: text part + structured JSON data part
  ▼
Research Agent → final report + machine-readable citations to user
```

The two protocols draw two different boundaries on purpose: **MCP is the
agent→tool seam** (the Research Agent doesn't know or care how search is
implemented), and **A2A is the agent→agent seam** (the Research Agent knows
only the Writer's published Agent Card — name, skills, transport — not that
it's Pydantic AI under the hood). Either side can be swapped without touching
the other.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python run_demo.py "Summarize recent developments in agent observability tools"
```

`run_demo.py` boots the Writer Agent's A2A server, runs the research pipeline,
prints the cited report and per-phase latency metrics, and shuts the server
down. Or with Docker: `docker compose run --rm demo`.

### Modes

Keys are read from a gitignored `.env` at the repo root (or the environment).
`A2A_DEMO_OFFLINE=1` forces the deterministic zero-credential mode even when
keys exist — that's what CI runs.

| Env var | Set | Unset |
|---|---|---|
| `ANTHROPIC_API_KEY` | Both agents use Claude (`claude-opus-4-8`) | fall through ↓ |
| `NVIDIA_API_KEY` | Both agents use `z-ai/glm-5.2` via NVIDIA NIM (OpenAI-compatible endpoint) | Deterministic offline models (same agent code paths) |
| `TAVILY_API_KEY` | Real web search via Tavily | Canned mock results |

The LLM seam shows the interop story twice over: the Research Agent reaches
the model through LangChain (`ChatAnthropic` / `ChatNVIDIA`), the Writer
through Pydantic AI (`anthropic:` provider / `OpenAIChatModel` pointed at
NIM's base URL) — and offline mode swaps in a deterministic `FunctionModel`
so the full protocol pipeline runs with zero credentials.

## Measured results

From `uv run python -m evals.run_evals` (per-run JSON in `evals/results/`):

**Offline mode — 21 runs (3 topics × 7), deterministic:**

| metric | value |
|---|---|
| delegation success rate | **100%** (21/21) |
| latency p50 / p95 | 0.27s / 0.28s |
| citation coverage | 100% |

**Real mode — GLM-5.2 via NVIDIA NIM + live Tavily search, 6 runs, LLM-judged:**

| metric | value |
|---|---|
| delegation success rate | 83% (5/6 — one transient upstream API failure; retries are Phase 2) |
| latency p50 / p95 | 38.1s / 49.5s (reasoning model) |
| mean report length | 263 words (100% within 120–500 bounds) |
| citation coverage | 92% of sources cited inline |
| LLM-judge (faithfulness / completeness / writing) | 5.0 / 5.0 / 5.0 |

The eval harness always computes the deterministic metrics; pass `--judge` to
add LLM-as-judge scoring. CI gates on the offline run: 100% success and 100%
citation coverage required, or the build fails.

## Observability: one trace, three services, two protocols

With tracing enabled, a single distributed trace spans the whole pipeline —
the W3C `traceparent` is propagated across the **MCP boundary** in the spawn
environment and across the **A2A boundary** in message metadata (and HTTP
headers):

```
research-agent   research.pipeline
research-agent   ├─ search → mcp.web_search          (CLIENT)
mcp-web-search   │   └─ web_search.execute            (SERVER, other process)
research-agent   ├─ synthesize → LLM span             (OpenInference)
research-agent   └─ delegate → a2a.delegate           (CLIENT)
writer-agent         └─ writer.execute_task           (SERVER, other process)
writer-agent             └─ pydantic-ai agent run → LLM span
```

View it in [Arize Phoenix](https://phoenix.arize.com/):

```bash
uvx arize-phoenix serve                       # UI + OTLP collector on :6006
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006 uv run python run_demo.py "your topic"
```

`tests/test_tracing.py` asserts the three services' spans share one trace id
on every CI run. LLM calls are auto-instrumented on both frameworks
(OpenInference for LangChain/LangGraph, `Agent.instrument_all()` for
Pydantic AI), so token counts and prompts land in the same trace.

## Where the A2A handoff happens

```
research.a2a | A2A DISCOVERY: found agent 'Writer Agent' v1.0.0 ... skills: ['write_report']
research.a2a | A2A HANDOFF: delegating 'write_report' to 'Writer Agent' ...
writer.a2a   | A2A task received: task_id=... input=1038 chars
research.a2a | A2A task status: TASK_STATE_WORKING
research.a2a | A2A artifact received: 'report' (2227 chars, 5 citations)
research.a2a | A2A task status: TASK_STATE_COMPLETED
writer.a2a   | A2A task completed: ... 5/5 sources cited
```

`research_agent/a2a_client.py` resolves the card and streams the task;
`writer_agent/a2a_server.py`'s `WriterExecutor` bridges the A2A task lifecycle
(submitted → working → artifact → completed) to the Pydantic AI agent. The
artifact carries both the plain-text report and a schema-validated JSON data
part (`common/report.py`) with per-claim citations.

## Testing

```bash
uv run pytest tests/ -q
```

- `test_e2e_offline.py` — runs the real CLI as a subprocess; asserts the A2A
  lifecycle ordering (discovery → handoff → completed) in the trace
- `test_tracing.py` — asserts one trace id across all three services
- `test_eval_gate.py` — runs the eval harness; fails on any delegation error
  or citation regression

All run in CI (GitHub Actions) in deterministic offline mode, plus a Docker
image build + compose smoke test.

## Repo layout

```
├── run_demo.py               # CLI: topic in → cited report out (+ metrics)
├── common/                   # config, logging, tracing, report schema
├── mcp_server/
│   └── search_server.py      # MCP stdio server: web_search (Tavily or mock)
├── research_agent/           # LangGraph
│   ├── agent.py              # graph: search → synthesize → delegate
│   ├── mcp_client.py         # MCP stdio client (agent→tool seam)
│   └── a2a_client.py         # A2A discovery + delegation (agent→agent seam)
├── writer_agent/             # Pydantic AI
│   ├── agent.py              # writer agent + offline FunctionModel fallback
│   ├── a2a_server.py         # FastAPI + a2a-sdk routes, port 9001
│   └── agent_card.json       # generated snapshot of the served Agent Card
├── evals/run_evals.py        # N-run harness: reliability + quality + judge
├── tests/                    # e2e, tracing, eval-gate (all offline-capable)
└── .github/workflows/ci.yml  # tests + Docker build on every push/PR
```

## Stack

- Python 3.14, managed with **uv**
- **LangGraph** (Research Agent) · **Pydantic AI** (Writer Agent)
- **a2a-sdk 1.x** — official A2A Python SDK (protobuf-typed, JSON-RPC
  transport, card at `/.well-known/agent-card.json`)
- **MCP Python SDK** (FastMCP over stdio) + Tavily Search API
- **OpenTelemetry + OpenInference + Arize Phoenix** for tracing

## Next steps (Phase 2+)

- Retries/backoff + circuit breaker on the A2A call (the measured 1-in-6
  real-mode transient failure is exactly why), chaos tests
- Multi-turn negotiation via A2A `TASK_STATE_INPUT_REQUIRED`
- Cost ledger per agent + budget kill switch; semantic cache
- Auth on A2A calls (Agent Card `security_schemes`)
- Third agent (critic/reflection loop) on a third framework
