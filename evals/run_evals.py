"""N-run evaluation harness for the research → report pipeline.

Measures what the demo alone can't prove: delegation reliability and output
quality over repeated runs. Deterministic metrics always run (success rate,
latency percentiles, word count, citation coverage); an LLM judge scores
faithfulness/completeness/writing when an LLM key is available.

Usage:
    uv run python -m evals.run_evals --n 20                    # offline-safe
    uv run python -m evals.run_evals --n 5 --judge             # + LLM judge
Results: printed table + evals/RESULTS.md + raw JSON in evals/results/.
"""
import argparse
import asyncio
import datetime
import json
import logging
import pathlib
import re
import statistics
import time

from common import NVIDIA_MODEL, have_anthropic, have_nvidia, setup_logging

logger = logging.getLogger("evals")

EVALS_DIR = pathlib.Path(__file__).parent
DEFAULT_TOPICS = [
    "agent observability tools",
    "on-device LLM inference",
    "vector database index tradeoffs",
]
WORD_BOUNDS = (120, 500)  # "roughly 300 words", tolerant of the leaner offline template

JUDGE_PROMPT = """You are grading a research report against the findings it \
was written from. Score 1-5 (5 best) on:
- faithfulness: no claims beyond the findings
- completeness: covers all major findings
- writing: clear, professional prose

Findings:
{findings}

Report:
{report}

Reply with ONLY a JSON object: {{"faithfulness": n, "completeness": n, "writing": n}}"""


def _word_count(report: str) -> int:
    body = report.split("[offline mode:")[0]
    return len(body.split())


async def _judge(findings: str, report: str) -> dict | None:
    if not (have_nvidia() or have_anthropic()):
        return None
    if have_anthropic():
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model="claude-opus-4-8", max_tokens=1024)
    else:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        llm = ChatNVIDIA(model=NVIDIA_MODEL, max_tokens=4096)
    msg = await llm.ainvoke(JUDGE_PROMPT.format(findings=findings, report=report))
    match = re.search(r"\{[^{}]*\}", str(msg.text))
    if not match:
        return None
    try:
        scores = json.loads(match.group(0))
        return {k: float(scores[k]) for k in ("faithfulness", "completeness", "writing")}
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


async def run_once(topic: str, judge: bool) -> dict:
    from research_agent.agent import run_research

    t0 = time.perf_counter()
    row: dict = {"topic": topic, "success": False}
    try:
        state = await run_research(topic)
        structured = state.get("structured_report") or {}
        n_cited = len(structured.get("citations", []))
        n_sources = int(structured.get("sources_available", 0)) or len(
            state["raw_results"]
        )
        row.update(
            success=True,
            report_head=state["report"][:200],
            total_s=round(time.perf_counter() - t0, 2),
            words=_word_count(state["report"]),
            citation_coverage=round(n_cited / n_sources, 2) if n_sources else 0.0,
            words_in_bounds=WORD_BOUNDS[0] <= _word_count(state["report"]) <= WORD_BOUNDS[1],
            **{k: v for k, v in state["timings"].items()},
        )
        if judge:
            row["judge"] = await _judge(state["findings"], state["report"])
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        logger.error("run failed for %r: %s", topic, row["error"])
    return row


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["success"]]
    latencies = sorted(r["total_s"] for r in ok)

    def pct(p: float) -> float | None:
        return round(statistics.quantiles(latencies, n=100)[int(p) - 1], 2) if len(latencies) >= 2 else (latencies[0] if latencies else None)

    summary = {
        "runs": len(rows),
        "success_rate": round(len(ok) / len(rows), 3) if rows else 0.0,
        "latency_p50_s": pct(50),
        "latency_p95_s": pct(95),
        "mean_words": round(statistics.mean(r["words"] for r in ok), 0) if ok else None,
        "words_in_bounds_rate": round(sum(r["words_in_bounds"] for r in ok) / len(ok), 3) if ok else None,
        "mean_citation_coverage": round(statistics.mean(r["citation_coverage"] for r in ok), 3) if ok else None,
    }
    judged = [r["judge"] for r in ok if r.get("judge")]
    if judged:
        for k in ("faithfulness", "completeness", "writing"):
            summary[f"judge_{k}"] = round(statistics.mean(j[k] for j in judged), 2)
    return summary


def write_results(rows: list[dict], summary: dict, mode: str) -> pathlib.Path:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = EVALS_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    raw_path = results_dir / f"{ts}-{mode}.json"
    raw_path.write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))

    lines = [
        f"# Eval results — {ts} ({mode} mode)",
        "",
        "| metric | value |",
        "|---|---|",
        *(f"| {k} | {v} |" for k, v in summary.items()),
        "",
        f"Raw per-run data: `evals/results/{raw_path.name}`",
        "",
    ]
    (EVALS_DIR / "RESULTS.md").write_text("\n".join(lines))
    return raw_path


async def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5, help="runs per topic")
    parser.add_argument("--topics", nargs="*", default=DEFAULT_TOPICS)
    parser.add_argument("--judge", action="store_true", help="LLM-judge scoring")
    args = parser.parse_args()

    setup_logging()
    logging.getLogger().setLevel(logging.WARNING)  # keep eval output readable

    from run_demo import start_writer

    mode = "anthropic" if have_anthropic() else "nvidia" if have_nvidia() else "offline"
    print(f"eval: {args.n} run(s) × {len(args.topics)} topic(s), mode={mode}, "
          f"judge={'on' if args.judge else 'off'}")

    writer_proc = start_writer()
    rows: list[dict] = []
    try:
        for topic in args.topics:
            for i in range(args.n):
                row = await run_once(topic, judge=args.judge)
                rows.append(row)
                status = "ok" if row["success"] else "FAIL"
                print(f"  [{len(rows):>3}] {status:<4} {topic!r} "
                      f"{row.get('total_s', '-')}s "
                      f"cite={row.get('citation_coverage', '-')}")
    finally:
        if writer_proc is not None:
            writer_proc.terminate()
            writer_proc.wait(timeout=10)

    summary = summarize(rows)
    raw_path = write_results(rows, summary, mode)
    print("\nsummary:")
    for k, v in summary.items():
        print(f"  {k:<28} {v}")
    print(f"\nwritten: evals/RESULTS.md, {raw_path}")
    return summary


if __name__ == "__main__":
    asyncio.run(main())
