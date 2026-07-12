"""CI quality gate: the offline pipeline must hit its reliability and
citation targets over repeated runs, not just once."""
import json
import os
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "evals" / "results"


def test_eval_gate_offline():
    env = dict(os.environ)
    env["A2A_DEMO_OFFLINE"] = "1"
    env["WRITER_AGENT_PORT"] = "9112"
    proc = subprocess.run(
        [sys.executable, "-m", "evals.run_evals",
         "--n", "3", "--topics", "agent observability tools"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"evals crashed:\n{proc.stderr[-3000:]}"

    latest = max(RESULTS_DIR.glob("*-offline.json"), key=lambda p: p.name)
    summary = json.loads(latest.read_text())["summary"]

    assert summary["success_rate"] == 1.0, summary
    assert summary["mean_citation_coverage"] == 1.0, summary
    assert summary["words_in_bounds_rate"] == 1.0, summary
