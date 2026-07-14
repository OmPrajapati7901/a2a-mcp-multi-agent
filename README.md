# A2A + MCP Multi-Agent Interop Demo

[![ci](https://github.com/OmPrajapati7901/a2a-mcp-multi-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/OmPrajapati7901/a2a-mcp-multi-agent/actions/workflows/ci.yml)

Three agents on **three different frameworks** — LangGraph, Pydantic AI, and
the OpenAI Agents SDK — discovering each other through a **semantic agent
registry** and delegating work over the **A2A (Agent2Agent) protocol**, with
**MCP** for tool access. Traced end-to-end, evaluated, chaos-tested,
cost-metered, and CI-gated.

```
User ──▶ run_demo.py
              │
              ▼
      Research Agent (LangGraph)
        │  MCP (stdio) → web_search (Tavily / mock)
        │  [HITL approval gate, budget kill switch]
        │
        │  registry: "who can write a report?" ──▶ Agent Registry (:9100)
        │  A2A: delegate write_report ─────────▶ Writer Agent (Pydantic AI, :9001)
        │    ↔ multi-turn negotiation               │ streams report chunks
        │      (TASK_STATE_INPUT_REQUIRED)          │ + citations data part
        │                                           ▼
        │  A2A: delegate review_report ────────▶ Critic Agent (OpenAI Agents SDK, :9002)
        │    ⟲ bounded reflection loop             │ verdict: approve / revise+feedback
        ▼                                           ▼
      report + citations + per-agent cost      live SSE dashboard (:9200)
```

Two protocols draw two different boundaries on purpose: **MCP is the
agent→tool seam**, **A2A is the agent→agent seam**. Every agent publishes an
Agent Card, self-registers with the registry, and is discovered by
*capability description* (embedding similarity in real mode, token overlap
offline) — no hardcoded topology.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python run_demo.py "Summarize recent developments in agent observability tools"
```

Everything is opt-in via env vars (all combinable, all with offline fallbacks):

| Env var | Effect |
|---|---|
| `A2A_CRITIC=1` | boot the Critic Agent; reflection loop (revise ≤1 round) |
| `A2A_REGISTRY=1` | boot the registry; semantic capability routing |
| `A2A_DASHBOARD=1` | boot the live dashboard at http://127.0.0.1:9200 |
| `A2A_HITL=1` | human approval on stdin before delegation (audited JSONL) |
| `A2A_MIN_FINDINGS=5` | writer negotiates for more input (multi-turn A2A) |
| `A2A_BUDGET_TOKENS=N` | abort before the next delegation once N tokens spent |
| `A2A_CACHE=1` | semantic cache — repeat topics cost $0 and skip all agents |
| `A2A_API_KEY=...` | API-key auth on A2A calls, declared in the Agent Card |
| `A2A_DEMO_OFFLINE=1` | force deterministic zero-credential mode (what CI runs) |
| `A2A_GUARD=off` | disable prompt-injection guard (red-team harness toggles this) |
| `A2A_BANDIT_EPSILON=0.15` | exploration rate for cost-quality bandit router |

LLM keys (gitignored `.env`): `ANTHROPIC_API_KEY` → Claude;
`NVIDIA_API_KEY` → GLM-5.2 via NIM (all three frameworks reach it three
different ways: `ChatNVIDIA`, Pydantic AI `OpenAIChatModel`, OpenAI Agents
SDK `AsyncOpenAI`); neither → deterministic offline models, same code paths.
`TAVILY_API_KEY` → real web search.

## Measured results

From `uv run python -m evals.run_evals` (raw JSON in `evals/results/`):

**Offline — 21 runs:** 100% delegation success, p50 0.27s, 100% citation
coverage.

**Real mode (GLM-5.2 + live Tavily), 6 judged runs:** 83% success (one
transient upstream failure — now absorbed by the retry layer built in
Phase 2), p50 38s, 263 mean words, 92% citation coverage, LLM-judge 5/5/5.

**Full three-agent mesh (real mode):** 60.5s end-to-end; per-agent cost
attribution `research=1866tok writer=1032tok critic=835tok | $0.0034/run`;
registry routing scores 0.55–0.60 (NIM embeddings); 5/5 sources cited.

CI gates on the offline eval: 100% success and 100% citation coverage or the
build fails.

## Production behaviors (each exercised by a test)

- **Retries + circuit breaker** on every A2A delegation (backoff+jitter ×3;
  breaker opens after 3 consecutive failures). Chaos-injected via
  `A2A_CHAOS_FAIL_N` — see [FAILURE_MODES.md](FAILURE_MODES.md) for the
  full failure-mode table with observed traces.
- **Multi-turn A2A negotiation**: the writer pauses a task with
  `TASK_STATE_INPUT_REQUIRED` when findings are too thin; the research agent
  searches again and resumes the *same task* (verified by task id).
- **Streaming**: report chunks stream over A2A artifact updates as the model
  generates; the structured citations+usage data part rides the last chunk.
- **Cost controls**: per-agent token/cost ledger (writer/critic usage flows
  back over A2A data parts); `A2A_BUDGET_TOKENS` stops *before* the next
  handoff, not after.
- **HITL**: LangGraph `interrupt()` checkpoints the graph pre-delegation;
  decisions (including non-interactive auto-approvals) land in an
  append-only audit log with a findings hash.
- **AuthN**: Agent Card publicly declares an `X-API-Key` scheme; middleware
  401s unauthenticated RPCs.
- **Injection guard**: every MCP search result is screened for 8 categories
  of indirect prompt injection (sentence-level neutralization + spotlighting)
  before reaching any LLM. Red-team validated: 100% ASR reduction, 0 false
  positives on a 20-attack corpus. See [SECURITY.md](SECURITY.md).
- **Bandit model router**: epsilon-greedy bandit chooses between model tiers
  (strong vs cheap) based on cumulative critic scores; sqlite-persistent
  across runs. Emits cost-quality frontier data for analysis.

## Observability: one trace, five services, two protocols

W3C `traceparent` crosses the **MCP boundary** in the spawn environment and
the **A2A boundary** in message metadata + HTTP headers, so one distributed
trace spans research-agent → mcp-web-search → writer-agent (→ critic-agent).
LLM calls are auto-instrumented on all frameworks (OpenInference for
LangChain/LangGraph, `Agent.instrument_all()` for Pydantic AI).

```bash
uvx arize-phoenix serve   # UI + OTLP collector on :6006
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006 uv run python run_demo.py "your topic"
```

`tests/test_tracing.py` asserts the single-trace property on every CI run.
The eval gate also caught a real bug here once: an env-leak made "offline"
subprocesses silently run real Tavily searches — documented in the commit
history.

## Testing

```bash
uv run pytest tests/ -q     # 51 tests, all offline-deterministic
```

e2e lifecycle ordering · single-trace assertion · eval regression gate ·
chaos/retry/breaker · budget kill switch · auth · semantic cache · registry
routing · reflection loop · negotiation (same-task resume) · HITL
approve/reject+audit · dashboard · injection guard (17 tests) · bandit
router (10 tests). CI also builds the Docker image and runs the compose
smoke test.

## Repo layout

```
├── run_demo.py               # CLI + service orchestration (boots the stack)
├── common/                   # config, logging, tracing, report schema,
│   ├── bandit.py             #   epsilon-greedy cost-quality model router
│   ├── cache.py              #   semantic cache (cosine ≥ 0.9, sqlite)
│   ├── guard.py              #   prompt-injection defense (8 patterns)
│   └── ...                   #   costs, resilience, audit, events
├── mcp_server/search_server.py   # MCP stdio server: web_search
├── research_agent/           # LangGraph: search → synthesize → delegate → review
├── writer_agent/             # Pydantic AI + A2A server (:9001)
├── critic_agent/             # OpenAI Agents SDK + A2A server (:9002)
├── registry/                 # agent registry + semantic routing (:9100)
├── dashboard/                # live SSE event dashboard (:9200)
├── evals/run_evals.py        # N-run harness: reliability + quality + judge
├── benchmarks/               # A2A protocol microbenchmark
├── redteam/                  # injection attack corpus + ASR harness
├── tests/                    # 51 offline-deterministic tests
├── SECURITY.md               # threat model, defense layers, red-team results
└── FAILURE_MODES.md          # failure-mode table with observed traces
```

## Stack

Python 3.14 / **uv** · **LangGraph** · **Pydantic AI** · **OpenAI Agents
SDK** · **a2a-sdk 1.x** (protobuf-typed, JSON-RPC) · **MCP SDK** (FastMCP,
stdio) · Tavily · **OpenTelemetry + OpenInference + Arize Phoenix** ·
FastAPI · Docker + GitHub Actions

## Security

See [SECURITY.md](SECURITY.md) for the threat model, defense layers, and
red-team validation results.

## Protocol overhead

The A2A JSON-RPC protocol adds ~2.76x latency over direct in-process calls
(offline FunctionModel, measured with `benchmarks/a2a_overhead.py`). This is
the cost of typed message serialization, HTTP transport, and Agent Card
discovery — worthwhile for the interop and discoverability guarantees.

## Next steps

- A2A conformance/fuzzing test kit (lifecycle-order violations, malformed
  protos, mid-stream disconnects) run against public A2A implementations
- Polyglot mesh: a writer in TypeScript (a2a-js) on the same registry
- Kubernetes deploy with per-agent HPA
