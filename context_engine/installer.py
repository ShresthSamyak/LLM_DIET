"""One-command installer — indexes the repo, configures all detected AI tools.

Supported platforms
-------------------
Claude Code  — .claude/settings.json   (UserPromptSubmit hook)
Cursor       — .cursor/rules/context-engine.mdc
Windsurf     — .windsurf/rules/context-engine.md
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .graph_builder import build_graph
from .js_parser import JS_EXTENSIONS, parse_js_file
from .parser import parse_file

_IGNORE_DIRS = frozenset({"venv", ".venv", "__pycache__", ".git", "node_modules", ".cecl"})
_OUTPUT_DIR = ".cecl"
_OUTPUT_FILE = "graph.json"


# ---------------------------------------------------------------------------
# Index step
# ---------------------------------------------------------------------------

@dataclass
class IndexResult:
    nodes: int
    edges: int
    py_files: int
    js_files: int
    elapsed_ms: float
    skipped: bool        # True when an existing graph was reused


def _collect(root: Path) -> tuple[list[Path], list[Path]]:
    py, js = [], []
    for path in root.rglob("*"):
        if any(part in _IGNORE_DIRS for part in path.parts):
            continue
        if path.suffix == ".py":
            py.append(path)
        elif path.suffix in JS_EXTENSIONS:
            js.append(path)
    return sorted(py), sorted(js)


def _run_index(root: Path) -> IndexResult:
    t0 = time.monotonic()
    py_files, js_files = _collect(root)

    results = []
    for path in py_files:
        r = parse_file(path)
        if r:
            results.append(r)
    for path in js_files:
        r = parse_js_file(path)
        if r:
            results.append(r)

    graph = build_graph(results)

    out_dir = root / _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / _OUTPUT_FILE).write_text(
        json.dumps(graph, indent=2), encoding="utf-8"
    )

    return IndexResult(
        nodes=len(graph["nodes"]),
        edges=len(graph["edges"]),
        py_files=len(py_files),
        js_files=len(js_files),
        elapsed_ms=(time.monotonic() - t0) * 1000,
        skipped=False,
    )


def _read_existing_graph(root: Path) -> IndexResult | None:
    graph_path = root / _OUTPUT_DIR / _OUTPUT_FILE
    if not graph_path.exists():
        return None
    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        return IndexResult(
            nodes=len(graph["nodes"]),
            edges=len(graph["edges"]),
            py_files=0,
            js_files=0,
            elapsed_ms=0.0,
            skipped=True,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Platform configuration
# ---------------------------------------------------------------------------

@dataclass
class Platform:
    name: str
    detection_dir: str
    config_path: str
    write_fn: str


_PLATFORMS: list[Platform] = [
    Platform("Claude Code", ".claude",   ".claude/settings.json",              "claude"),
    Platform("Cursor",      ".cursor",   ".cursor/rules/context-engine.mdc",   "rules"),
    Platform("Windsurf",    ".windsurf", ".windsurf/rules/context-engine.md",  "rules"),
]

_HOOK_COMMAND = "python -m context_engine.hooks.user_prompt_submit"

_CLAUDE_MD_SENTINEL = "<!-- context-engine -->"

_CLAUDE_MD_CONTENT = f"""\
{_CLAUDE_MD_SENTINEL}
## context-engine is active

This project uses [llm-diet](https://github.com/ShresthSamyak/LLM_DIET) for context-aware file access.

### Low Bandwidth Mode

The `llm-diet-shadow` MCP server is registered in `.mcp.json`. All `read_file` calls are intercepted and return compressed call-graph representations for indexed files.

**Rules:**
1. When context is injected before your prompt, use it directly — do not re-read those files.
2. When you call `read_file` and the response starts with `# [compressed by llm-diet]`, treat it as the complete file — do not call `read_file` again on the same path.
3. When you call `read_file` and the response starts with `# [llm-diet] strict mode:`, the file is not indexed — ask the user to run `context-engine index` rather than trying to read the raw file another way.
4. Prefer `list_directory` over shell `ls` or `find` — the shadow server's `list_directory` returns filenames only.

Say "Low Bandwidth Mode active" at the start of your first response in each session.
"""

_RULES_CONTENT = """\
# context-engine

Use context from `additionalContext` if provided — it contains the most
relevant functions from this codebase for the current query, pre-selected
by context-engine (AST + call graph analysis). Prefer these over searching
the codebase yourself when the provided context is sufficient.
"""


def _write_claude(dest: Path) -> None:
    existing: dict = {}
    if dest.exists():
        try:
            existing = json.loads(dest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    hook_entry = {
        "matcher": "",
        "hooks": [{"type": "command", "command": _HOOK_COMMAND}],
    }
    hooks = existing.setdefault("hooks", {})
    ups: list[dict] = hooks.setdefault("UserPromptSubmit", [])
    already = any(
        any(h.get("command") == _HOOK_COMMAND for h in e.get("hooks", []))
        for e in ups
    )
    if not already:
        ups.append(hook_entry)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _write_shadow_mcp(root: Path, strict: bool = False) -> None:
    """Register the shadow MCP server in .mcp.json if the graph exists."""
    if not (root / _OUTPUT_DIR / _OUTPUT_FILE).exists():
        return

    mcp_path = root / ".mcp.json"
    existing: dict = {}
    if mcp_path.exists():
        try:
            existing = json.loads(mcp_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    servers: dict = existing.setdefault("mcpServers", {})
    entry: dict = {
        "command": "python",
        "args": ["-m", "context_engine.shadow_server"],
        "cwd": str(root),
    }
    if strict:
        entry["env"] = {"LLM_DIET_STRICT": "1"}

    if "llm-diet-shadow" not in servers:
        servers["llm-diet-shadow"] = entry

    mcp_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _write_rules(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(_RULES_CONTENT, encoding="utf-8")


def _write_claude_md(root: Path) -> None:
    """Write or append the context-engine section to CLAUDE.md."""
    dest = root / "CLAUDE.md"
    existing = dest.read_text(encoding="utf-8") if dest.exists() else ""
    if _CLAUDE_MD_SENTINEL in existing:
        return  # already present — don't duplicate
    separator = "\n" if existing and not existing.endswith("\n\n") else ""
    dest.write_text(existing + separator + _CLAUDE_MD_CONTENT, encoding="utf-8")


def _configure_platforms(root: Path) -> list[str]:
    """Write configs for every detected platform. Returns list of platform names."""
    configured: list[str] = []
    for p in _PLATFORMS:
        if not (root / p.detection_dir).exists():
            continue
        dest = root / p.config_path
        try:
            if p.write_fn == "claude":
                _write_claude(dest)
                _write_shadow_mcp(root)
            else:
                _write_rules(dest)
            configured.append(p.name)
        except OSError:
            pass
    return configured


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class InstallResult:
    index: IndexResult
    platforms: list[str]


def run_install(root: Path, force_reindex: bool = False) -> InstallResult:
    """Index (or reuse) the graph, then configure all detected platforms."""
    existing = None if force_reindex else _read_existing_graph(root)
    index_result = existing if existing else _run_index(root)
    platforms = _configure_platforms(root)
    _write_claude_md(root)
    return InstallResult(index=index_result, platforms=platforms)
