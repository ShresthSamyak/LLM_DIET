#!/usr/bin/env python3
"""
PostToolUse hook.
Appends tool usage to session log.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from context_engine.policy import PolicySession, normalize_path, _GATED_TOOLS


def main() -> None:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        sys.exit(0)

    tool_name: str = payload.get("tool_name", "")
    if tool_name not in _GATED_TOOLS:
        sys.exit(0)

    tool_input: dict = payload.get("tool_input", {})
    filepath = None
    for key in ("file_path", "path", "file"):
        val = tool_input.get(key)
        if val and isinstance(val, str):
            filepath = normalize_path(val)
            break

    if not filepath:
        sys.exit(0)

    session = PolicySession.load()

    # Record successful read/edit for audit trail
    entry = {"tool": tool_name, "file": filepath}
    if not hasattr(session, "_tool_uses"):
        session.__dict__.setdefault("_tool_uses", [])
    session.__dict__["_tool_uses"].append(entry)

    session.save()
    session.save_log()
    sys.exit(0)


if __name__ == "__main__":
    main()
