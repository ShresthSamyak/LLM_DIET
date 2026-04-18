"""Deterministic context ranking and formatting for LLM token efficiency."""

from __future__ import annotations

import ast
import re

_FILE_IMPORTANCE = {
    "auth": 3, "service": 2, "api": 2, "route": 2, "router": 2,
    "handler": 1, "util": 1, "helper": 1, "model": 1, "schema": 1,
}

_PURPOSE_VERBS = {
    "get": "fetch", "fetch": "fetch", "read": "fetch",
    "create": "create", "add": "create", "insert": "create", "post": "create",
    "update": "update", "patch": "update", "edit": "update",
    "delete": "delete", "remove": "delete",
    "validate": "validate", "check": "validate", "verify": "validate",
    "login": "authenticate", "auth": "authenticate", "token": "authenticate",
    "build": "build", "make": "build", "generate": "build",
    "parse": "parse", "decode": "parse", "encode": "parse",
    "send": "send", "emit": "send", "notify": "send",
    "run": "execute", "execute": "execute", "process": "execute",
}

_FRAMEWORK_SIGNALS = {
    "fastapi": {"APIRouter", "FastAPI", "Depends", "HTTPException"},
    "django": {"HttpResponse", "JsonResponse", "request.method"},
    "flask": {"Blueprint", "jsonify", "request.json"},
    "sqlalchemy": {"AsyncSession", "Session", "select(", "db.execute"},
    "jwt": {"jwt.encode", "jwt.decode", "create_access_token"},
    "async": {"async def", "await ", "asyncio"},
}


def _keyword_score(node: dict, query_tokens: set[str]) -> float:
    name_tokens = set(re.split(r"[_\W]+", node.get("name", "").lower()))
    file_tokens = set(re.split(r"[/_\W]+", node.get("file", "").lower()))
    code = node.get("code", "").lower()
    hits = len(query_tokens & name_tokens) * 3
    hits += len(query_tokens & file_tokens) * 1
    hits += sum(1 for t in query_tokens if t in code)
    return float(hits)


def _centrality_score(node: dict) -> float:
    calls = len(node.get("calls", []))
    callers = len(node.get("callers", []))
    return min(calls + callers * 1.5, 10.0)


def _file_score(node: dict) -> float:
    fpath = node.get("file", "").lower()
    score = 0.0
    for keyword, weight in _FILE_IMPORTANCE.items():
        if keyword in fpath:
            score += weight
    return score


def rank_nodes(nodes: list[dict], query: str) -> list[dict]:
    query_tokens = set(re.split(r"\W+", query.lower())) - {"", "a", "the", "in", "to", "for"}
    scored = []
    for node in nodes:
        score = (
            _keyword_score(node, query_tokens) * 2.0
            + _centrality_score(node) * 0.8
            + _file_score(node) * 1.2
        )
        scored.append((score, node))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [n for _, n in scored[:8]]


def _extract_args(code: str, name: str) -> str:
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                args = [a.arg for a in node.args.args if a.arg != "self"]
                defaults_offset = len(args) - len(node.args.defaults)
                parts = []
                for i, arg in enumerate(args):
                    di = i - defaults_offset
                    if di >= 0:
                        parts.append(f"{arg}=...")
                    else:
                        parts.append(arg)
                return ", ".join(parts)
    except SyntaxError:
        pass
    m = re.search(rf"def\s+{re.escape(name)}\s*\(([^)]*)\)", code)
    if m:
        raw = m.group(1).replace("\n", " ").strip()
        args = [a.strip().split(":")[0].strip() for a in raw.split(",") if a.strip() and a.strip() != "self"]
        return ", ".join(args)
    return ""


def _infer_purpose(name: str, node_type: str) -> str:
    tokens = re.split(r"[_\W]+", name.lower())
    for token in tokens:
        if token in _PURPOSE_VERBS:
            rest = [t for t in tokens if t != token]
            subject = "_".join(rest) if rest else "resource"
            return f"{_PURPOSE_VERBS[token]} {subject}"
    if node_type == "class":
        return "data model / service class"
    return "utility"


def summarize_node(node: dict) -> str:
    name = node.get("name", "unknown")
    ntype = node.get("type", "function")
    code = node.get("code", "")
    fpath = node.get("file", "")
    short_path = "/".join(fpath.replace("\\", "/").split("/")[-2:]) if fpath else ""

    if ntype == "class":
        purpose = _infer_purpose(name, "class")
        return f"{name} [{short_path}] -> {purpose}"

    args = _extract_args(code, name)
    purpose = _infer_purpose(name, "function")
    prefix = "async " if re.search(r"^\s*async def", code, re.MULTILINE) else ""
    return f"{prefix}{name}({args}) [{short_path}] -> {purpose}"


def build_flow(nodes: list[dict]) -> str:
    if not nodes:
        return "(no flow detected)"

    name_set = {n.get("name") for n in nodes}
    graph: dict[str, list[str]] = {
        n["name"]: [c for c in n.get("calls", []) if c in name_set and c is not None]
        for n in nodes if n.get("name")
    }

    # Find root: node with no callers within the set
    caller_targets = {c for targets in graph.values() for c in targets}
    roots = [n for n in graph if n not in caller_targets]
    start = roots[0] if roots else next(iter(graph), None)

    if not start:
        return " -> ".join(n.get("name", "?") for n in nodes[:4])

    chain: list[str] = []
    visited: set[str] = set()
    current: str | None = start
    while current and current not in visited and len(chain) < 4:
        chain.append(current)
        visited.add(current)
        nxt = graph.get(current, [])
        current = nxt[0] if nxt else None

    return " -> ".join(chain) if chain else start


def _detect_constraints(nodes: list[dict]) -> list[str]:
    combined = " ".join(n.get("code", "") for n in nodes)
    found = []
    for label, signals in _FRAMEWORK_SIGNALS.items():
        if any(sig in combined for sig in signals):
            found.append(label)
    return found


def _detect_missing(nodes: list[dict]) -> list[str]:
    defined = {n.get("name") for n in nodes}
    missing = set()
    for node in nodes:
        for call in node.get("calls", []):
            if call and call not in defined:
                missing.add(call)
    return sorted(missing)[:6]


def format_output(query: str, nodes: list[dict]) -> str:
    ranked = rank_nodes(nodes, query)
    flow = build_flow(ranked)
    summaries = "\n".join(summarize_node(n) for n in ranked)
    missing = _detect_missing(ranked)
    constraints = _detect_constraints(ranked)

    missing_str = ", ".join(missing) if missing else "none detected"
    constraints_str = ", ".join(constraints) if constraints else "unknown"

    return (
        f"=== TASK ===\n{query}\n\n"
        f"=== CORE FLOW ===\n{flow}\n\n"
        f"=== KEY FUNCTIONS ===\n{summaries}\n\n"
        f"=== MISSING ===\n{missing_str}\n\n"
        f"=== CONSTRAINTS ===\n{constraints_str}"
    )
