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
_LARGE_FILE_CHARS = 50_000
_LARGE_FILE_HEAD_LINES = 200

_BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf",
    ".zip", ".whl", ".exe", ".pyc", ".pkl", ".db", ".sqlite",
})

_STRICT_MODE: bool = os.environ.get("LLM_DIET_STRICT", "").lower() in ("1", "true")


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

    try:
        tree = ast.parse(dedented)
        fn = tree.body[0]
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError
    except Exception:
        if len(lines) <= _MAX_BODY_LINES + 1:
            return dedented
        head = lines[:_MAX_BODY_LINES + 1]
        tail_count = len(lines) - len(head)
        return "\n".join(head) + f"\n    # ... ({tail_count} more lines)"

    body_stmts = fn.body
    body_start = body_stmts[0].lineno - 1

    sig_lines = lines[:body_start]
    raw_body = lines[body_start:]

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

    body = [l for l in raw_body if l.strip() and not l.lstrip().startswith("#")]

    if len(body) <= _MAX_BODY_LINES:
        return "\n".join(sig_lines + body)

    kept = body[:_MAX_BODY_LINES]
    remaining = body[_MAX_BODY_LINES:]

    tail_extras: list[str] = []
    body_line_offset = body_start + (len(raw_body) - len(body))
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
    p = Path(file_path)

    # 1. Binary files — skip immediately, no disk read needed
    if p.suffix.lower() in _BINARY_EXTENSIONS:
        return f"# [llm-diet] binary file skipped: {file_path}"

    # 2. File not found
    if not p.exists():
        return f"# [llm-diet] file not found: {file_path}"

    # 3. Empty file — return real content (empty string)
    raw = p.read_text(encoding="utf-8", errors="replace")
    if not raw:
        return raw

    graph = _load_graph()

    # HIT path — file is in the graph
    if graph is not None:
        nodes = _resolve(file_path, graph)
        if nodes:
            fn_blocks: list[str] = []
            for node in sorted(nodes, key=lambda n: n.get("line", 0)):
                code = node.get("code", "").rstrip()
                if code:
                    fn_blocks.append(_compress_fn(code))

            body = "\n\n".join(fn_blocks)
            original_size = len(raw)
            compressed_size = len(body)

            # 4. Compressed is larger than original — return raw instead
            if compressed_size >= original_size:
                return raw

            saved = original_size - compressed_size
            header = "\n".join([
                "# [compressed by llm-diet]",
                f"# file: {file_path}",
                f"# functions: {len(nodes)}",
                f"# tokens saved: ~{saved // 4} (original ~{original_size // 4}, compressed ~{compressed_size // 4})",
                "",
            ])
            return header + body

    # MISS path — file not in graph

    # strict mode: refuse to pass through unindexed files
    if _STRICT_MODE:
        return f"# [llm-diet] strict mode: {file_path} is not indexed. Run context-engine index to add it."

    # 5. Very large unindexed file — truncate to first 200 lines
    if len(raw) > _LARGE_FILE_CHARS:
        head = "\n".join(raw.splitlines()[:_LARGE_FILE_HEAD_LINES])
        return (
            head
            + "\n# [llm-diet] truncated: file not indexed and exceeds 50k chars."
            " Run context-engine index to include it."
        )

    return raw


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


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    import tempfile, sys

    _GRAPH_PATH = Path(".cecl/graph.json")
    _graph_cache = None
    _graph_loaded = False

    results: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition, detail))
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    print("Running edge-case tests for read_file...\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # 1. Binary file (no disk access needed — just extension check)
        bin_path = tmp / "image.png"
        bin_path.write_bytes(b"\x89PNG\r\n")
        result = read_file(str(bin_path))
        check(
            "Binary file skipped",
            result == f"# [llm-diet] binary file skipped: {bin_path}",
            repr(result[:60]),
        )

        # 2. File not found
        missing = tmp / "does_not_exist.py"
        result = read_file(str(missing))
        check(
            "File not found",
            result == f"# [llm-diet] file not found: {missing}",
            repr(result[:60]),
        )

        # 3. Empty file
        empty = tmp / "empty.py"
        empty.write_text("", encoding="utf-8")
        result = read_file(str(empty))
        check(
            "Empty file returns empty string",
            result == "",
            repr(result),
        )

        # 4. Compressed larger than original — should return raw
        #    A tiny file with one trivial function: compression overhead > savings
        tiny = tmp / "tiny.py"
        tiny.write_text("def f():\n    return 1\n", encoding="utf-8")
        original_text = tiny.read_text(encoding="utf-8")
        # No graph loaded for tmp dir, so this hits MISS path.
        # To test case 4, we need a HIT with bad compression ratio.
        # Simulate by temporarily patching _resolve to return a node whose
        # compressed output would be larger than the original.
        import context_engine.shadow_server as _mod
        _orig_resolve = _mod._resolve
        _orig_load = _mod._load_graph

        fake_code = "def f():\n    return 1\n"
        _mod._resolve = lambda fp, g: [{"type": "function", "file": str(tiny), "line": 1, "code": fake_code}]
        _mod._load_graph = lambda: {"nodes": [], "edges": []}

        result = read_file(str(tiny))
        # compressed body of fake_code = "def f():\n    return 1" (~21 chars)
        # original "def f():\n    return 1\n" = 22 chars
        # header alone is 100+ chars → compressed_size (body only) vs original
        # Let's use a deliberately very short original that body alone exceeds
        very_short = tmp / "vshort.py"
        very_short.write_text("x=1\n", encoding="utf-8")
        _mod._resolve = lambda fp, g: [{"type": "function", "file": str(very_short), "line": 1,
                                         "code": "def f(a, b, c, d, e):\n    return a+b+c+d+e\n"}]
        result = read_file(str(very_short))
        check(
            "Compressed >= original returns raw",
            not result.startswith("# [compressed by llm-diet]"),
            f"got {repr(result[:40])}",
        )

        _mod._resolve = _orig_resolve
        _mod._load_graph = _orig_load

        # 5. Large unindexed MISS file > 50k chars
        large = tmp / "large.py"
        large.write_text("x = 1\n" * 10000, encoding="utf-8")   # 60,000 chars, 10,000 lines
        _mod._graph_cache = None
        _mod._graph_loaded = False
        result = read_file(str(large))
        actual_lines = result.splitlines()
        check(
            "Large MISS file truncated to 200 lines + note",
            len(actual_lines) == 201 and "truncated" in actual_lines[-1],
            f"{len(actual_lines)} lines, last: {repr(actual_lines[-1][:60])}",
        )

        # 6. STRICT + HIT: should return compressed output, not the strict error
        # Use _mod.read_file so patches to _mod.* take effect in its globals.
        hit_file = tmp / "hit.py"
        hit_file.write_text(
            "def compute(x, y):\n" + "    z = x + y\n" * 20 + "    return z\n",
            encoding="utf-8",
        )
        _mod._STRICT_MODE = True
        _mod._resolve = lambda fp, g: [{"type": "function", "file": str(hit_file), "line": 1,
                                         "code": hit_file.read_text()}]
        _mod._load_graph = lambda: {"nodes": [], "edges": []}
        result = _mod.read_file(str(hit_file))
        check(
            "STRICT + HIT returns compressed output",
            result.startswith("# [compressed by llm-diet]"),
            repr(result[:60]),
        )

        # 7. STRICT + MISS: should return strict error, not raw content
        miss_file = tmp / "miss.py"
        miss_file.write_text("x = 42\n", encoding="utf-8")
        _mod._resolve = lambda fp, g: []
        result = _mod.read_file(str(miss_file))
        expected = f"# [llm-diet] strict mode: {miss_file} is not indexed. Run context-engine index to add it."
        check(
            "STRICT + MISS returns strict error",
            result == expected,
            repr(result[:80]),
        )

        _mod._STRICT_MODE = False
        _mod._resolve = _orig_resolve
        _mod._load_graph = _orig_load

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"Results: {passed}/{len(results)} passed")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _run_tests()
    else:
        main()
