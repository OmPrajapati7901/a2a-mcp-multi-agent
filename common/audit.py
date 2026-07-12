"""Append-only audit log for human-in-the-loop decisions."""
import datetime
import hashlib
import json
import logging
import os
import pathlib

logger = logging.getLogger("audit")


def record_decision(topic: str, decision: str, findings: str, source: str) -> None:
    audit_dir = pathlib.Path(os.environ.get("A2A_AUDIT_DIR", ".audit"))
    audit_dir.mkdir(exist_ok=True)
    entry = {
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "topic": topic,
        "decision": decision,
        "source": source,  # "interactive" | "auto-noninteractive"
        "findings_sha256": hashlib.sha256(findings.encode()).hexdigest(),
    }
    with (audit_dir / "decisions.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info("HITL AUDIT: %s (%s) recorded to %s",
                decision, source, audit_dir / "decisions.jsonl")
