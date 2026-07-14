"""A2A protocol microbenchmark: protocol overhead vs in-process call.

Measures:
  1. In-process call to the writer agent (direct function call)
  2. A2A JSON-RPC call to the writer agent (HTTP + protobuf serialization)
  3. Comparison: latency distributions, overhead ratio, serialization cost

Usage:
    A2A_DEMO_OFFLINE=1 uv run python -m benchmarks.a2a_overhead
    A2A_DEMO_OFFLINE=1 uv run python -m benchmarks.a2a_overhead --runs 50 --json
"""
import argparse
import asyncio
import json
import multiprocessing
import os
import statistics
import sys
import time

# Ensure offline mode for deterministic benchmarking.
os.environ.setdefault("A2A_DEMO_OFFLINE", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

FINDINGS_TEXT = (
    "Topic: agent observability tools\n\n"
    "Findings:\n"
    "- OpenTelemetry is the emerging standard for LLM tracing [S1]\n"
    "- Arize Phoenix provides session-level traces and cost accounting [S2]\n"
    "- A2A protocol standardizes agent-to-agent task delegation [S3]\n\n"
    "Sources:\n"
    "[S1] https://example.com/otel\n"
    "[S2] https://example.com/phoenix\n"
    "[S3] https://example.com/a2a"
)

BENCH_PORT = 9099  # dedicated port to avoid conflicts


def _start_writer_server():
    """Boot the writer A2A server in a subprocess."""
    import uvicorn
    os.environ["A2A_DEMO_OFFLINE"] = "1"
    os.environ["WRITER_AGENT_PORT"] = str(BENCH_PORT)
    os.environ["WRITER_AGENT_HOST"] = "127.0.0.1"
    from writer_agent.a2a_server import build_app
    uvicorn.run(build_app(), host="127.0.0.1", port=BENCH_PORT,
                log_level="error")


async def bench_inprocess(runs: int) -> list[float]:
    """Measure direct function-call latency (no HTTP, no serialization)."""
    from writer_agent.agent import write_report

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        await write_report(FINDINGS_TEXT)
        latencies.append(time.perf_counter() - t0)
    return latencies


async def bench_a2a(port: int, runs: int) -> list[float]:
    """Measure A2A JSON-RPC call latency (HTTP + protobuf ser/de)."""
    import httpx

    import a2a.types as ty
    from a2a.client import ClientConfig, create_client
    from a2a.helpers import new_text_message

    # Discover the card.
    base_url = f"http://127.0.0.1:{port}"
    async with httpx.AsyncClient(timeout=10) as http:
        from a2a.client import A2ACardResolver
        card = await A2ACardResolver(http, base_url).get_agent_card()

    latencies = []
    for _ in range(runs):
        http_client = httpx.AsyncClient(timeout=60)
        client = await create_client(card, ClientConfig(httpx_client=http_client))
        message = new_text_message(FINDINGS_TEXT, role=ty.Role.ROLE_USER)

        t0 = time.perf_counter()
        report_text = ""
        async for resp in client.send_message(ty.SendMessageRequest(message=message)):
            kind = resp.WhichOneof("payload")
            if kind == "artifact_update":
                from a2a.helpers import get_artifact_text
                text = get_artifact_text(resp.artifact_update.artifact)
                report_text += text
        latencies.append(time.perf_counter() - t0)

        await client.close()
        await http_client.aclose()
    return latencies


def stats(latencies: list[float]) -> dict:
    def pct(p: int) -> float:
        # Interpolated percentile (same method as evals/run_evals.py); the
        # previous nearest-rank indexing returned the max for small n.
        if len(latencies) < 2:
            return latencies[0]
        return statistics.quantiles(latencies, n=100)[p - 1]

    return {
        "n": len(latencies),
        "mean_s": round(statistics.mean(latencies), 4),
        "median_s": round(statistics.median(latencies), 4),
        "p95_s": round(pct(95), 4),
        "p99_s": round(pct(99), 4),
        "min_s": round(min(latencies), 4),
        "max_s": round(max(latencies), 4),
        "stdev_s": round(statistics.stdev(latencies), 4) if len(latencies) > 1 else 0,
    }


async def wait_for_server(port: int, timeout: float = 15):
    """Poll until the writer server is ready."""
    import httpx
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=2) as http:
        while time.monotonic() < deadline:
            try:
                r = await http.get(f"http://127.0.0.1:{port}/.well-known/agent-card.json")
                if r.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.3)
    raise RuntimeError(f"Writer server not ready after {timeout}s")


async def main():
    parser = argparse.ArgumentParser(description="A2A protocol microbenchmark")
    parser.add_argument("--runs", type=int, default=20,
                        help="Number of benchmark iterations")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    # Boot writer server in a subprocess.
    print("Booting writer server for A2A benchmark...", file=sys.stderr)
    server = multiprocessing.Process(target=_start_writer_server, daemon=True)
    server.start()
    await wait_for_server(BENCH_PORT)
    print("Writer server ready.", file=sys.stderr)

    try:
        print(f"Running in-process benchmark ({args.runs} runs)...", file=sys.stderr)
        inprocess = await bench_inprocess(args.runs)

        print(f"Running A2A benchmark ({args.runs} runs)...", file=sys.stderr)
        a2a = await bench_a2a(BENCH_PORT, args.runs)
    finally:
        server.terminate()
        server.join(timeout=5)

    in_stats = stats(inprocess)
    a2a_stats = stats(a2a)

    overhead_ratio = a2a_stats["mean_s"] / in_stats["mean_s"] if in_stats["mean_s"] else 0
    overhead_pct = (overhead_ratio - 1) * 100 if overhead_ratio > 1 else 0

    if args.json:
        print(json.dumps({
            "in_process": in_stats,
            "a2a_jsonrpc": a2a_stats,
            "overhead_ratio": round(overhead_ratio, 2),
            "overhead_pct": round(overhead_pct, 1),
        }, indent=2))
        return

    print("\n=== A2A Protocol Microbenchmark ===\n")
    print(f"  Runs: {args.runs}")
    print(f"  Writer model: offline FunctionModel")
    print()
    print(f"  {'Metric':<12} {'In-Process':>12} {'A2A JSON-RPC':>14} {'Overhead':>10}")
    print(f"  {'-'*12} {'-'*12} {'-'*14} {'-'*10}")
    for key in ("mean_s", "median_s", "p95_s", "p99_s"):
        label = key.replace("_s", "").upper()
        ip = in_stats[key]
        a2 = a2a_stats[key]
        ov = ((a2 / ip) - 1) * 100 if ip else 0
        print(f"  {label:<12} {ip:>11.4f}s {a2:>13.4f}s {ov:>+9.1f}%")
    print()
    print(f"  Overhead ratio: {overhead_ratio:.2f}x "
          f"({overhead_pct:+.1f}% latency added by A2A protocol)")
    print(f"  Serialization: HTTP + protobuf (a2a-sdk v1.1.0)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
