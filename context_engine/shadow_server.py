"""Shadow MCP server — intercepts read_file/list_directory calls.

Serves compressed call-graph representations for indexed files,
passes through unindexed files unchanged.
"""
from __future__ import annotations

import ast
import json
import os
import textwrap
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("llm-diet-shadow")

_GRAPH_PATH = Path(".cecl/graph.json")
_graph_cache: dict | None = None
_graph_loaded = False

_MAX_BODY_LINES = 8


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def _load_graph() -> dict | None:
    global _graph_cache, _graph_loaded
    if _graph_loaded:
        return _graph_cache
    _graph_loaded = True
    if not _GRAPH_PATH.exists():
        return None
    try:
        _graph_cache = json.loads(_GRAPH_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return _graph_cache


def _resolve(file_path: str, graph: dict) -> list[dict]:
    """Return all function/method nodes belonging to file_path."""
    target = str(Path(file_path).resolve())
    return [
        n for n in graph.get("nodes", [])
        if n.get("type") in ("function", "method")
        and str(Path(n.get("file", "")).resolve()) == target
    ]


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

def _compress_fn(code: str) -> str:
    """Return a compact representation of a function.

    Keeps: signature, first 8 body lines, any return/raise beyond that.
    Strips: docstring, pure-comment lines, remainder of body.
    Falls back to raw truncation if AST parse fails.
    """
    dedented = textwrap.dedent(code).rstrip()
    lines = dedented.splitlines()
    if not lines:
        return code

    # --- Parse for precise boundary info ---
    try:
        tree = ast.parse(dedented)
        fn = tree.body[0]
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError
    except Exception:
        # Fallback: keep signature line + up to MAX_BODY body lines
        if len(lines) <= _MAX_BODY_LINES + 1:
            return dedented
        head = lines[:_MAX_BODY_LINES + 1]
        tail_count = len(lines) - len(head)
        return "\n".join(head) + f"\n    # ... ({tail_count} more lines)"

    # Signature: everything up to (but not including) first body statement
    body_stmts = fn.body
    body_start = body_stmts[0].lineno - 1  # 0-indexed line of first stmt

    sig_lines = lines[:body_start]
    raw_body = lines[body_start:]

    # Strip leading docstring
    first_stmt = body_stmts[0]
    if (
        isinstance(first_stmt, ast.Expr)
        and isinstance(first_stmt.value, ast.Constant)
        and isinstance(first_stmt.value.value, str)
        and len(body_stmts) > 1
    ):
        next_start = body_stmts[1].lineno - 1 - body_start
        raw_body = raw_body[next_start:]
        body_stmts = body_stmts[1:]

    # Strip lines that are only comments
    body = [l for l in raw_body if l.strip() and not l.lstrip().startswith("#")]

    if len(body) <= _MAX_BODY_LINES:
        return "\n".join(sig_lines + body)

    kept = body[:_MAX_BODY_LINES]
    remaining = body[_MAX_BODY_LINES:]

    # Harvest return/raise from the truncated tail (AST-level)
    tail_extras: list[str] = []
    body_line_offset = body_start + (len(raw_body) - len(body))  # adjust for stripped docstring
    for stmt in body_stmts[1:]:
        if not isinstance(stmt, (ast.Return, ast.Raise)):
            continue
        rel = stmt.lineno - 1 - body_line_offset
        if rel >= _MAX_BODY_LINES and 0 <= rel < len(body):
            tail_extras.append(body[rel])
        if len(tail_extras) >= 2:
            break

    result = sig_lines + kept
    result.append(f"    # ... ({len(remaining)} more lines)")
    result.extend(tail_extras)
    return "\n".join(result)


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
def read_file(file_path: str) -> str:
    """Return compressed call-graph context for indexed files, raw content otherwise."""
    graph = _load_graph()

    if graph is not None:
        nodes = _resolve(file_path, graph)
        if nodes:
            try:
                original_size = len(Path(file_path).read_text(encoding="utf-8"))
            except OSError:
                original_size = 0

            fn_blocks: list[str] = []
            for node in sorted(nodes, key=lambda n: n.get("line", 0)):
                code = node.get("code", "").rstrip()
                if code:
                    fn_blocks.append(_compress_fn(code))

            body = "\n\n".join(fn_blocks)
            compressed_size = len(body)

            saved = original_size - compressed_size
            saved_pct = int(100 * saved / original_size) if original_size else 0

            header = "\n".join([
                "# [compressed by llm-diet]",
                f"# file: {file_path}",
                f"# functions: {len(nodes)}",
                f"# tokens saved: ~{saved // 4} (original ~{original_size // 4}, compressed ~{compressed_size // 4})",
                "",
            ])
            return header + body

    # Passthrough — file not in graph or graph not built
    try:
        return Path(file_path).read_text(encoding="utf-8")
    except OSError as exc:
        return f"ERROR: {exc}"


@mcp.tool()
def list_directory(path: str) -> str:
    """List filenames in a directory (names only, no content)."""
    try:
        entries = sorted(os.listdir(path))
    except OSError as exc:
        return f"ERROR: {exc}"
    return "\n".join(entries)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
