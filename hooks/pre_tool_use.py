#!/usr/bin/env python3
"""
PreToolUse hook.
Gates Read / Edit / Write against the session allowed set.
Passes Grep / Glob / Bash through unconditionally.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from context_engine.policy import (
    PolicySession,
    _GATED_TOOLS,
    basename,
    gate,
)

_FILEPATH_KEYS = ("file_path", "path", "file")


def _extract_filepath(tool_input: dict) -> str | None:
    for key in _FILEPATH_KEYS:
        val = tool_input.get(key)
        if val and isinstance(val, str):
            return val
    return None


def _last_assistant_message() -> str:
    """
    Read last Claude message from payload context if provided.
    Falls back to empty string (hook payload doesn't include full transcript).
    """
    return ""


def main() -> None:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        sys.exit(0)

    tool_name: str = payload.get("tool_name", "")

    # Discovery tools: always allow
    if tool_name not in _GATED_TOOLS:
        sys.exit(0)

    tool_input: dict = payload.get("tool_input", {})
    filepath = _extract_filepath(tool_input)
    if not filepath:
        sys.exit(0)

    session = PolicySession.load()

    # Open mode: allow everything, just log
    if session.mode == "open":
        sys.exit(0)

    # Check if an assistant message with NEED/REASON was provided in payload
    last_message = payload.get("last_assistant_message", _last_assistant_message())

    decision, reason = gate(session, filepath, last_message)

    if decision == "allow":
        sys.exit(0)

    # Block: output structured reason for Claude to see
    response = {"decision": "block", "reason": reason}
    sys.stdout.write(json.dumps(response))
    sys.exit(1)


if __name__ == "__main__":
    main()
