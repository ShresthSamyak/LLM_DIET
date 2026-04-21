"""Smart context pruning: classify → score → dedup → inline → budget-cut.

Narrows the ranked node list from ~15 candidates down to ≤ 8 carefully
chosen nodes, inlining thin helpers as hint comments on their parent entry.
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass
from typing import Any, Literal

Node = dict[str, Any]
Category = Literal["CORE", "CRITICAL", "SUPPORTING", "NOISE"]


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

MAX_KEPT = 8
_THIN_WRAPPER_MAX_STMTS = 3
_NOISE_RATIO = 0.8                   # ≥80% log/print stmts → NOISE

_PRINT_NAMES: frozenset[str] = frozenset({"print", "pprint", "pp"})
_LOG_ATTRS: frozenset[str] = frozenset({
    "debug", "info", "warning", "warn", "error", "exception", "log", "critical",
})


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PruneResult:
    kept: list[Node]
    inline_hints: dict[str, list[str]]   # parent_id → hint lines
    categories: dict[str, Category]      # node_id → category (for telemetry)


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _parse_fn(code: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Parse a code snippet and return its top-level function/method def."""
    try:
        tree = ast.parse(textwrap.dedent(code).strip())
    except SyntaxError:
        return None
    if not tree.body:
        return None
    top = tree.body[0]
    if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return top
    return None


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    """Drop a leading docstring expression, if present."""
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _is_log_call(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return False
    func = stmt.value.func
    if isinstance(func, ast.Name):
        return func.id in _PRINT_NAMES
    if isinstance(func, ast.Attribute):
        return func.attr in _LOG_ATTRS
    return False


# ---------------------------------------------------------------------------
# Classification signals
# ---------------------------------------------------------------------------

def _is_noise_body(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when the function body is mostly prints/logs (no real logic)."""
    body = _strip_docstring(fn.body)
    if not body:
        return False
    noise = sum(1 for s in body if _is_log_call(s))
    return noise / len(body) >= _NOISE_RATIO


def _is_thin_wrapper(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True for short helpers with no control flow — candidates for inlining."""
    body = _strip_docstring(fn.body)
    if not body or len(body) > _THIN_WRAPPER_MAX_STMTS:
        return False
    for stmt in body:
        if isinstance(stmt, (ast.If, ast.Try, ast.For, ast.While,
                              ast.AsyncFor, ast.With, ast.AsyncWith)):
            return False
    return True


def importance_score(node: Node) -> int:
    """Deterministic debugging-importance score for a node.

    Higher is more important.  Based on presence of control flow, raises,
    non-trivial returns, and call density — penalised for log-noise or
    thin-wrapper shape.
    """
    code = node.get("code", "")
    if not code:
        # File/class nodes have no body to score — treat as neutral.
        return 0

    fn = _parse_fn(code)
    if fn is None:
        return 0

    score = 0
    for child in ast.walk(fn):
        if isinstance(child, (ast.If, ast.IfExp, ast.Try)):
            score += 3
        elif isinstance(child, ast.Raise):
            score += 4
        elif isinstance(child, ast.Return):
            if child.value is None or isinstance(child.value, (ast.Constant, ast.Name)):
                score += 1   # trivial return
            else:
                score += 2   # non-trivial return
        elif isinstance(child, ast.Call):
            score += 1

    if _is_noise_body(fn):
        score -= 5
    if _is_thin_wrapper(fn):
        score -= 3

    return score


def classify(node: Node, entry_set: set[str], depth: int, score: int) -> Category:
    """Bucket a node into one of CORE / CRITICAL / SUPPORTING / NOISE."""
    nid = node["id"]
    ntype = node.get("type", "")

    if nid in entry_set:
        return "CORE"

    # Non-code nodes (file/class): kept as structural breadcrumbs only.
    if ntype not in ("function", "method"):
        return "SUPPORTING" if depth <= 1 else "NOISE"

    # Shape-based classification takes precedence over score:
    #   - log-heavy bodies → NOISE regardless of call density
    #   - thin wrappers   → SUPPORTING (to be inlined, not emitted in full)
    fn = _parse_fn(node.get("code", ""))
    if fn is not None:
        if _is_noise_body(fn):
            return "NOISE"
        if _is_thin_wrapper(fn):
            return "SUPPORTING"

    if score <= 0:
        return "NOISE"
    if depth == 1 and score >= 5:
        return "CRITICAL"
    if depth == 1:
        return "SUPPORTING"
    if score >= 7:
        return "CRITICAL"
    if score >= 3:
        return "SUPPORTING"
    return "NOISE"


# ---------------------------------------------------------------------------
# Inline-hint generation
# ---------------------------------------------------------------------------

def inline_hint(node: Node) -> str | None:
    """Return a one-line hint for a thin-wrapper node, or None if unsuitable."""
    code = node.get("code", "")
    fn = _parse_fn(code)
    if fn is None or not _is_thin_wrapper(fn):
        return None

    body = _strip_docstring(fn.body)
    if not body:
        return None

    try:
        lines = [ast.unparse(s) for s in body]
    except Exception:
        return None

    name = fn.name
    if len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None:
        return f"# {name}: {lines[0].removeprefix('return ').strip()}"
    return f"# {name}: " + "; ".join(lines)


# ---------------------------------------------------------------------------
# Structural deduplication
# ---------------------------------------------------------------------------

class _AnonymiseNames(ast.NodeTransformer):
    """Rewrite all identifiers to a placeholder so isomorphic fns collide."""

    def visit_Name(self, node: ast.Name) -> ast.Name:
        return ast.copy_location(ast.Name(id="_", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.arg:
        return ast.copy_location(ast.arg(arg="_", annotation=None), node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        # Attribute access like `self.x` → `_.x` (keep attr names — they carry meaning)
        self.generic_visit(node)
        return node


def _structural_key(code: str) -> str:
    """Return a stable key that collapses name-only differences."""
    fn = _parse_fn(code)
    if fn is None:
        return code.strip()
    cloned = _AnonymiseNames().visit(fn)
    cloned.name = "_"
    ast.fix_missing_locations(cloned)
    try:
        return ast.unparse(cloned)
    except Exception:
        return code.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prune(
    ranked: list[Node],
    entry_ids: list[str],
    visited: dict[str, int],
    max_kept: int = MAX_KEPT,
) -> PruneResult:
    """Narrow *ranked* to ≤ *max_kept* nodes using classification + dedup.

    Pipeline:
      1. classify + score each node
      2. drop NOISE outright
      3. emit thin-wrapper SUPPORTING nodes as inline hints on entry nodes
      4. structurally dedupe CRITICAL / SUPPORTING (keep highest score)
      5. fill budget CORE → CRITICAL → SUPPORTING
    """
    entry_set = set(entry_ids)
    categories: dict[str, Category] = {}
    scored: list[tuple[Node, Category, int]] = []

    for node in ranked:
        depth = visited.get(node["id"], 99)
        score = importance_score(node)
        cat = classify(node, entry_set, depth, score)
        categories[node["id"]] = cat
        if cat == "NOISE":
            continue
        scored.append((node, cat, score))

    # ------------------------------------------------------------------
    # Step 1: inline thin-wrapper SUPPORTING helpers onto entry node(s).
    # ------------------------------------------------------------------
    inline: dict[str, list[str]] = {}
    consumed: set[str] = set()
    attach_target = entry_ids[0] if entry_ids else None

    for node, cat, _ in scored:
        if cat != "SUPPORTING":
            continue
        if node.get("type") not in ("function", "method"):
            continue
        hint = inline_hint(node)
        if hint and attach_target:
            inline.setdefault(attach_target, []).append(hint)
            consumed.add(node["id"])

    scored = [t for t in scored if t[0]["id"] not in consumed]

    # ------------------------------------------------------------------
    # Step 2: structural dedup inside CRITICAL and SUPPORTING.
    # ------------------------------------------------------------------
    def _dedup(items: list[tuple[Node, int]]) -> list[tuple[Node, int]]:
        best: dict[str, tuple[Node, int]] = {}
        for n, s in items:
            code = n.get("code", "")
            key = _structural_key(code) if code else n["id"]
            if key not in best or best[key][1] < s:
                best[key] = (n, s)
        # Preserve score order (desc).
        return sorted(best.values(), key=lambda t: -t[1])

    core = [n for n, c, _ in scored if c == "CORE"]
    critical = _dedup([(n, s) for n, c, s in scored if c == "CRITICAL"])
    supporting = _dedup([(n, s) for n, c, s in scored if c == "SUPPORTING"])

    # ------------------------------------------------------------------
    # Step 3: fill budget.
    # ------------------------------------------------------------------
    kept: list[Node] = list(core)
    for n, _ in critical:
        if len(kept) >= max_kept:
            break
        kept.append(n)
    for n, _ in supporting:
        if len(kept) >= max_kept:
            break
        kept.append(n)

    # ------------------------------------------------------------------
    # Step 4: if the only kept node is a bare file node (no code), expand
    # it by pulling in all function/method children from ranked.
    # This prevents "database connection" from returning just a filename stub.
    # ------------------------------------------------------------------
    if len(kept) == 1 and kept[0].get("type") == "file":
        file_id = kept[0]["id"]
        children = [
            n for n in ranked
            if n.get("type") in ("function", "method")
            and n.get("file") == file_id
        ]
        if children:
            kept = children[:max_kept]

    return PruneResult(kept=kept, inline_hints=inline, categories=categories)
