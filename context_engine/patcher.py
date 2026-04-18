"""Unified diff generation (via LLM) and application."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_DIFF_SYSTEM = """\
You are an expert backend engineer modifying an existing Python codebase.

STRICT RULES:
1. NEVER output full files — only unified diffs.
2. Every diff MUST use git-style unified diff format:
   --- a/<relative/path/to/file.py>
   +++ b/<relative/path/to/file.py>
   @@ -<start>,<count> +<start>,<count> @@
   (context lines unchanged)
   +added line
   -removed line
3. Include 3 lines of unchanged context around every change.
4. To CREATE a new file use /dev/null as the 'a' path:
   --- a//dev/null
   +++ b/<relative/path/to/newfile.py>
   @@ -0,0 +1,<N> @@
   +<all lines of the new file>
5. NEVER invent function names — only use names from KNOWN FUNCTIONS.
6. NEVER import from modules not listed in KNOWN IMPORTS.
7. Write async code only when the existing file uses async def.
8. Output ONLY the diff — no prose, no markdown fences, no explanation.
"""


@dataclass
class Hunk:
    orig_start: int
    orig_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)


@dataclass
class FileDiff:
    orig_path: str   # "a/..." or "/dev/null"
    new_path: str    # "b/..."
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def is_new_file(self) -> bool:
        return self.orig_path in ("/dev/null", "a//dev/null")

    @property
    def relative_path(self) -> str:
        p = self.new_path
        if p.startswith("b/"):
            p = p[2:]
        return p


_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_diff(diff_text: str) -> list[FileDiff]:
    """Parse a unified diff string into FileDiff objects."""
    # Strip markdown fences if LLM wrapped output
    diff_text = re.sub(r"^```[^\n]*\n", "", diff_text, flags=re.MULTILINE)
    diff_text = re.sub(r"^```\s*$", "", diff_text, flags=re.MULTILINE)

    diffs: list[FileDiff] = []
    current: FileDiff | None = None
    current_hunk: Hunk | None = None

    for line in diff_text.splitlines():
        if line.startswith("--- "):
            if current_hunk and current:
                current.hunks.append(current_hunk)
                current_hunk = None
            orig = line[4:].strip()
            current = FileDiff(orig_path=orig, new_path="")
            diffs.append(current)
        elif line.startswith("+++ ") and current is not None:
            current.new_path = line[4:].strip()
        elif line.startswith("@@ ") and current is not None:
            if current_hunk:
                current.hunks.append(current_hunk)
            m = _HUNK_HEADER.match(line)
            if m:
                os_, oc, ns, nc = m.group(1), m.group(2), m.group(3), m.group(4)
                current_hunk = Hunk(
                    orig_start=int(os_),
                    orig_count=int(oc) if oc is not None else 1,
                    new_start=int(ns),
                    new_count=int(nc) if nc is not None else 1,
                )
        elif current_hunk is not None:
            if line.startswith(("+", "-", " ")):
                current_hunk.lines.append(line)
            elif line == "\\ No newline at end of file":
                pass  # ignore

    if current_hunk and current:
        current.hunks.append(current_hunk)

    return [d for d in diffs if d.new_path]


def _apply_hunks(original: str, hunks: list[Hunk]) -> str:
    """Apply hunks to original text, returning patched text."""
    orig_lines = original.splitlines(keepends=True)
    # Ensure lines end with newline for uniform handling
    result: list[str] = []
    offset = 0  # cumulative line offset from previous hunks

    for hunk in sorted(hunks, key=lambda h: h.orig_start):
        # Lines before this hunk (1-indexed → 0-indexed)
        hunk_start_0 = hunk.orig_start - 1 + offset
        # Copy lines before hunk
        result.extend(orig_lines[:hunk_start_0])
        orig_lines = orig_lines[hunk_start_0:]

        removed = 0
        for hline in hunk.lines:
            if hline.startswith(" "):
                # context — keep
                result.append(orig_lines[removed] if removed < len(orig_lines) else hline[1:] + "\n")
                removed += 1
            elif hline.startswith("-"):
                # remove — skip
                removed += 1
            elif hline.startswith("+"):
                # add
                content = hline[1:]
                if not content.endswith("\n"):
                    content += "\n"
                result.append(content)

        orig_lines = orig_lines[removed:]
        offset = offset  # recalculated implicitly via list slicing

    result.extend(orig_lines)
    text = "".join(result)
    # Strip trailing newline added if original didn't have one
    return text


def _extract_create_content(hunks: list[Hunk]) -> str:
    lines = []
    for hunk in hunks:
        for hline in hunk.lines:
            if hline.startswith("+"):
                lines.append(hline[1:])
    return "".join(lines)


def backup_file(file_path: Path, backup_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / f"{file_path.name}.{timestamp}.bak"
    shutil.copy2(file_path, dest)
    return dest


def apply_file_diff(diff: FileDiff, project_root: Path) -> tuple[Path, str, str]:
    """
    Apply a FileDiff to disk.
    Returns (target_path, original_content, patched_content).
    """
    target = project_root / diff.relative_path

    if diff.is_new_file:
        original = ""
        patched = _extract_create_content(diff.hunks)
    else:
        if not target.exists():
            raise FileNotFoundError(f"Target file not found: {target}")
        original = target.read_text(encoding="utf-8")
        patched = _apply_hunks(original, diff.hunks)

    return target, original, patched


def generate_diff(
    query: str,
    plan: list[dict],
    context: str,
    known_functions: list[str],
    missing_functions: list[str],
    project_root: Path,
) -> str:
    """Call Anthropic Sonnet to generate a unified diff for the given plan."""
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic package required: pip install anthropic>=0.40") from exc

    client = anthropic.Anthropic()

    plan_text = "\n".join(
        f"- {step.get('action', 'modify')} {step['file']}: {step.get('reason', '')}"
        for step in plan
    )

    user_prompt = f"""TASK: {query}

PLAN:
{plan_text}

KNOWN FUNCTIONS (use only these):
{chr(10).join(known_functions) if known_functions else "none"}

MISSING FUNCTIONS (you must implement these):
{chr(10).join(missing_functions) if missing_functions else "none"}

CODEBASE CONTEXT:
{context}

PROJECT ROOT: {project_root}

Generate the unified diff now."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=_DIFF_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return message.content[0].text
