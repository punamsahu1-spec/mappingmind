import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", "logs/audit.jsonl")


def write_audit_event(event: dict[str, Any]) -> None:
    """
    Writes one audit event per line.

    This gives MappingMind a production-style audit trail:
    who asked, what was asked, what answer was generated,
    what sources were used, what the agent decided, and when.
    """
    Path(AUDIT_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

    audit_event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event,
    }

    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(audit_event, ensure_ascii=False) + "\n")


def read_recent_audit_events(limit: int = 20) -> list[dict[str, Any]]:
    """
    Reads the most recent audit events.
    Useful later for dashboard observability.
    """
    path = Path(AUDIT_LOG_PATH)

    if not path.exists():
        return []

    lines = path.read_text(encoding="utf-8").splitlines()
    recent_lines = lines[-limit:]

    events = []
    for line in recent_lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return events