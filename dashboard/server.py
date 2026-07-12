"""Live pipeline dashboard: agents POST events, browsers watch them stream in.

Run: `uv run python -m dashboard.server` (port 9200), then open
http://127.0.0.1:9200 and run the demo with A2A_DASHBOARD_URL set (or let
run_demo boot everything with A2A_DASHBOARD=1).
"""
import asyncio
import collections
import logging
import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from common import setup_logging

logger = logging.getLogger("dashboard")

DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "9200"))

app = FastAPI(title="A2A Pipeline Dashboard")

_events: collections.deque = collections.deque(maxlen=500)
_subscribers: set[asyncio.Queue] = set()

PAGE = """<!doctype html>
<html><head><title>A2A Pipeline Dashboard</title><style>
body { font: 13px/1.5 ui-monospace, monospace; background: #0d1117;
       color: #c9d1d9; margin: 2rem; }
h1 { font-size: 16px; color: #58a6ff; }
table { border-collapse: collapse; width: 100%; }
td { padding: 3px 10px; border-bottom: 1px solid #21262d; vertical-align: top; }
.agent-research-agent { color: #58a6ff; } .agent-writer-agent { color: #3fb950; }
.agent-critic-agent { color: #d29922; } .agent-demo { color: #8b949e; }
.kind { color: #f0883e; }
</style></head><body>
<h1>A2A multi-agent pipeline — live events</h1>
<table id="log"></table>
<script>
const log = document.getElementById("log");
function row(e) {
  const tr = document.createElement("tr");
  tr.innerHTML = `<td>${e.ts}</td>` +
    `<td class="agent-${e.agent}">${e.agent}</td>` +
    `<td class="kind">${e.kind}</td>` +
    `<td>${JSON.stringify(e.detail)}</td>`;
  log.prepend(tr);
}
fetch("/events").then(r => r.json()).then(es => es.forEach(row));
new EventSource("/stream").onmessage = ev => row(JSON.parse(ev.data));
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return PAGE


@app.post("/event")
async def post_event(request: Request) -> dict:
    event = await request.json()
    _events.append(event)
    logger.info("event: %(agent)s %(kind)s %(detail)s", event)
    for queue in list(_subscribers):
        queue.put_nowait(event)
    return {"ok": True, "events": len(_events)}


@app.get("/events")
async def events() -> list:
    return list(_events)


@app.get("/stream")
async def stream():
    import json

    queue: asyncio.Queue = asyncio.Queue()
    _subscribers.add(queue)

    async def gen():
        try:
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            _subscribers.discard(queue)

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    setup_logging()
    logger.info("dashboard on http://127.0.0.1:%d", DASHBOARD_PORT)
    uvicorn.run(app, host="127.0.0.1", port=DASHBOARD_PORT, log_level="warning")
