"""LLM-based planning layer: produces a JSON list of file-action steps."""

from __future__ import annotations

import json
import re

_PLAN_SYSTEM = """\
You are a software architect planning code changes for a Python project.

Given a task and codebase context, output a JSON array of steps to complete the task.
Each step must be an object with exactly these fields:
  "file"   — relative path from project root (e.g. "app/routes/auth.py")
  "action" — one of: "create", "modify", "append"
  "reason" — one sentence explaining what this step does

RULES:
- Prefer modifying existing files over creating new ones.
- Use "create" only when a file clearly does not exist yet.
- Use "append" when adding new top-level functions/classes to an existing file without touching existing code.
- Order steps so dependencies come first (e.g. schema before router).
- Output ONLY the JSON array — no markdown, no prose.

EXAMPLE:
[
  {"file": "app/schemas/auth.py", "action": "create", "reason": "Define LoginRequest and TokenResponse Pydantic models."},
  {"file": "app/routes/auth.py", "action": "modify", "reason": "Add POST /login endpoint using LoginRequest schema."}
]
"""


def plan(query: str, context: str) -> list[dict]:
    """
    Call Anthropic Haiku to generate a step-by-step plan.
    Returns a list of {'file', 'action', 'reason'} dicts.
    Raises RuntimeError if the anthropic package is missing or the response is malformed.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic package required: pip install anthropic>=0.40") from exc

    client = anthropic.Anthropic()

    user_prompt = f"""TASK: {query}

CODEBASE CONTEXT:
{context}

Output the JSON plan now."""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=_PLAN_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = message.content[0].text.strip()

    # Strip markdown fences if present
    raw = re.sub(r"^```[^\n]*\n", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"^```\s*$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    try:
        steps = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Planner returned invalid JSON: {exc}\n\nRaw response:\n{raw}") from exc

    if not isinstance(steps, list):
        raise RuntimeError(f"Planner returned non-list JSON: {type(steps)}")

    validated: list[dict] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        if "file" not in step or "action" not in step:
            continue
        validated.append({
            "file": step["file"],
            "action": step.get("action", "modify"),
            "reason": step.get("reason", ""),
        })

    if not validated:
        raise RuntimeError("Planner returned zero valid steps.")

    return validated
