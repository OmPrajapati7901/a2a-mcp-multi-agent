"""Typed report schema shared across the A2A boundary.

The Research Agent numbers its sources ([S1], [S2], …) in the task text it
delegates; the Writer must cite them inline. The Writer parses its own output
against this schema and ships the structured form as an A2A data part
alongside the plain text, so the caller gets machine-readable citations.
"""
import re

from pydantic import BaseModel

_SOURCE_LINE = re.compile(r"^\[S(\d+)\]\s+(.*?)\s+—\s+(\S+)\s*$", re.MULTILINE)
_CITE = re.compile(r"\[S(\d+)\]")


class Citation(BaseModel):
    sid: int
    title: str
    url: str


class Report(BaseModel):
    title: str
    body: str
    citations: list[Citation]
    sources_available: int


def format_sources(results: list[dict]) -> str:
    """Render search results as the numbered source block sent to the Writer."""
    return "\n".join(
        f"[S{i}] {r['title']} — {r['url']}" for i, r in enumerate(results, 1)
    )


def parse_sources(task_text: str) -> dict[int, Citation]:
    """Recover the numbered sources from a delegated task's text."""
    return {
        int(sid): Citation(sid=int(sid), title=title.strip(), url=url)
        for sid, title, url in _SOURCE_LINE.findall(task_text)
    }


def parse_report(report_text: str, sources: dict[int, Citation]) -> Report:
    """Validate writer output into the typed schema; citations are the
    intersection of [S#] markers in the text and the known sources."""
    lines = [ln.strip() for ln in report_text.strip().splitlines() if ln.strip()]
    title = lines[0] if lines else "Untitled report"
    cited_ids = sorted({int(sid) for sid in _CITE.findall(report_text)})
    citations = [sources[i] for i in cited_ids if i in sources]
    return Report(
        title=title,
        body=report_text,
        citations=citations,
        sources_available=len(sources),
    )
