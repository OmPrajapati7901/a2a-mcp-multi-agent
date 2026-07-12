"""Verify one distributed trace spans all three services.

Runs the demo with the console span exporter and asserts that the spans
emitted by research-agent, mcp-web-search (via spawn-env propagation), and
writer-agent (via A2A message-metadata propagation) share a single trace id.
"""
import re

from tests.test_e2e_offline import run_demo

_SPAN_BLOCK = re.compile(r'"name": "(?P<name>[^"]+)"[\s\S]{0,400}?"trace_id": "(?P<tid>0x\w+)"')

CROSS_SERVICE_SPANS = {
    "research.pipeline",   # research-agent root
    "web_search.execute",  # mcp-web-search (child via spawn env)
    "a2a.delegate",        # research-agent client side of the handoff
    "writer.execute_task", # writer-agent (child via A2A message metadata)
}


def test_single_trace_across_three_services():
    proc = run_demo({"A2A_DEMO_TRACE_CONSOLE": "1", "WRITER_AGENT_PORT": "9118"})
    assert proc.returncode == 0, f"demo failed:\n{proc.stderr[-3000:]}"

    trace_ids = {
        m["name"]: m["tid"] for m in _SPAN_BLOCK.finditer(proc.stderr)
    }
    missing = CROSS_SERVICE_SPANS - trace_ids.keys()
    assert not missing, f"spans not emitted: {missing}"

    cross = {name: trace_ids[name] for name in CROSS_SERVICE_SPANS}
    assert len(set(cross.values())) == 1, f"trace ids diverge: {cross}"
