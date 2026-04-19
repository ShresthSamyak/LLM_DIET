from __future__ import annotations

import ast
import re

_STOPWORDS = frozenset({"a", "an", "the", "in", "to", "for", "of", "on", "at", "by", "is", "it"})

_ALIASES: dict[str, set[str]] = {
    "login": {"auth", "token", "password", "signin", "authenticate"},
    "auth": {"login", "token", "jwt", "authenticate", "signin"},
    "token": {"auth", "jwt", "access", "bearer", "refresh"},
    "user": {"account", "profile", "member"},
    "create": {"add", "insert", "post", "new"},
    "delete": {"remove", "drop", "destroy"},
    "get": {"fetch", "read", "retrieve", "find"},
    "update": {"patch", "edit", "modify"},
    "validate": {"check", "verify", "assert"},
    "hash": {"password", "bcrypt", "crypt"},
    "db": {"database", "session", "repo", "repository"},
}


def _is_test_file(fpath: str) -> bool:
    p = fpath.replace("\\", "/").lower()
    name = p.split("/")[-1]
    return name.startswith("test_") or name.endswith("_test.py") or "/test" in p or "/tests/" in p


def _enrich(nid: str, raw: dict, calls_map: dict, callers_map: dict) -> dict:
    node = dict(raw)
    node["name"] = nid.split(":")[-1]
    node["calls"] = [dst.split(":")[-1] for dst in calls_map.get(nid, [])]
    node["callers"] = [src.split(":")[-1] for src in callers_map.get(nid, [])]
    return node


def resolve_nodes(selected_ids: list[str], graph: dict) -> list[dict]:
    """
    Resolve node IDs to enriched dicts. When test-file nodes are in the pool,
    automatically pulls in the implementation nodes they call.
    """
    by_id: dict[str, dict] = {n["id"]: dict(n) for n in graph.get("nodes", []) if "id" in n}

    # Track full destination IDs (not just names) to enable cross-file expansion
    calls_map: dict[str, list[str]] = {}   # src_id -> [dst_id, ...]
    callers_map: dict[str, list[str]] = {} # dst_id -> [src_id, ...]
    for edge in graph.get("edges", []):
        if edge.get("type") == "calls":
            src, dst = edge["from"], edge["to"]
            calls_map.setdefault(src, []).append(dst)
            callers_map.setdefault(dst, []).append(src)

    resolved: dict[str, dict] = {}
    for nid in selected_ids:
        raw = by_id.get(nid)
        if not raw:
            continue
        resolved[nid] = _enrich(nid, raw, calls_map, callers_map)

    # For test-file nodes: follow calls into implementation files
    impl_ids: list[str] = []
    for nid, node in list(resolved.items()):
        if not _is_test_file(node.get("file", "")):
            continue
        for dst_id in calls_map.get(nid, []):
            if dst_id not in resolved and not _is_test_file(by_id.get(dst_id, {}).get("file", "")):
                impl_ids.append(dst_id)

    for nid in impl_ids:
        raw = by_id.get(nid)
        if raw:
            resolved[nid] = _enrich(nid, raw, calls_map, callers_map)

    # Narrowness fallback: if all nodes share one file, add their callees
    files = {n.get("file") for n in resolved.values()}
    if len(files) == 1:
        extra_ids = []
        for nid, node in list(resolved.items()):
            for dst_id in calls_map.get(nid, []):
                if dst_id not in resolved and by_id.get(dst_id, {}).get("type") in ("function", "method"):
                    extra_ids.append(dst_id)
                    if len(extra_ids) >= 4:
                        break
            if len(extra_ids) >= 4:
                break
        for nid in extra_ids:
            raw = by_id.get(nid)
            if raw:
                resolved[nid] = _enrich(nid, raw, calls_map, callers_map)

    return list(resolved.values())


def _tokens(text: str) -> set[str]:
    return set(re.split(r"[_\W]+", text.lower())) - _STOPWORDS - {""}


def _alias_tokens(query_tokens: set[str]) -> set[str]:
    expanded = set(query_tokens)
    for t in query_tokens:
        expanded |= _ALIASES.get(t, set())
    return expanded


def _short_path(fpath: str) -> str:
    parts = fpath.replace("\\", "/").split("/")
    return parts[-1] if parts else fpath


def _score(node: dict, query_tokens: set[str], alias_tokens: set[str]) -> int:
    if node.get("type") not in ("function", "method"):
        return 0
    name = node.get("name", "").lower()
    fpath = node.get("file", "").lower()
    code = node.get("code", "").lower()
    name_tokens = _tokens(name)

    score = 0
    # Exact / token match
    if name in query_tokens or name_tokens == query_tokens:
        score += 3
    score += len(query_tokens & name_tokens) * 2
    # Alias match
    score += len((alias_tokens - query_tokens) & name_tokens)
    # Substring match: query token appears inside function name
    for t in alias_tokens:
        if len(t) >= 3 and t in name:
            score += 1
    # File path match
    if any(k in fpath for k in query_tokens):
        score += 2
    # Docstring / body keyword match (light weight)
    for t in query_tokens:
        if len(t) >= 4 and t in code:
            score += 1
            break
    return score


def _strip_comments(code: str) -> str:
    lines = []
    for line in code.splitlines():
        cleaned = re.sub(r'\s*#.*$', "", line).rstrip()
        if cleaned.strip():
            lines.append(cleaned)
    return "\n".join(lines)


def _strip_docstrings(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
    return code


def _compress(code: str, max_lines: int = 18) -> str:
    code = _strip_docstrings(code)
    code = _strip_comments(code)
    lines = [l for l in code.splitlines() if l.strip()]
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["    ..."]
    return "\n".join(lines)


def _fallback_select(nodes: list[dict], top_n: int) -> list[dict]:
    """Return top_n function nodes by call-graph centrality when scoring finds nothing."""
    fn_nodes = [n for n in nodes if n.get("type") in ("function", "method")]
    ranked = sorted(
        fn_nodes,
        key=lambda n: len(n.get("calls", [])) + len(n.get("callers", [])),
        reverse=True,
    )
    return ranked[:top_n]


def rank_and_select(nodes: list[dict], query: str, top_n: int = 3) -> list[dict]:
    query_tokens = _tokens(query)
    alias_tokens = _alias_tokens(query_tokens)

    scored = []
    for node in nodes:
        s = _score(node, query_tokens, alias_tokens)
        if s == 0:
            continue
        scored.append((s, node))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [n for _, n in scored[:top_n]]

    if not top:
        return _fallback_select(nodes, top_n)

    top_names = {n.get("name") for n in top}
    by_name = {n.get("name"): n for n in nodes if n.get("name")}
    additions: list[dict] = []
    caller_count = 0

    for node in top:
        for callee in node.get("calls", []):
            if callee and callee not in top_names and callee in by_name:
                if _score(by_name[callee], query_tokens, alias_tokens) >= 2:
                    additions.append(by_name[callee])
                    top_names.add(callee)
        for caller in node.get("callers", []):
            if caller_count >= 2:
                break
            if caller and caller not in top_names and caller in by_name:
                if _score(by_name[caller], query_tokens, alias_tokens) >= 2:
                    additions.append(by_name[caller])
                    top_names.add(caller)
                    caller_count += 1

    seen: set[str] = set()
    unique: list[dict] = []
    for n in top + additions:
        name = n.get("name")
        if name and name not in seen:
            seen.add(name)
            unique.append(n)

    re_scored = sorted(
        unique,
        key=lambda n: _score(n, query_tokens, alias_tokens),
        reverse=True,
    )
    return re_scored[:top_n]


def format_output(_query: str, nodes: list[dict]) -> str:
    if not nodes:
        return "NO_CONTEXT_FOUND"

    by_file: dict[str, list[dict]] = {}
    for node in nodes:
        key = _short_path(node.get("file", "unknown"))
        by_file.setdefault(key, []).append(node)

    blocks: list[str] = []
    total = 0

    for fname, fnodes in by_file.items():
        blocks.append(f"[{fname}]")
        for node in fnodes:
            compressed = _compress(node.get("code", ""))
            lines = compressed.splitlines()
            if total + len(lines) > 55:
                remaining = 55 - total
                if remaining <= 0:
                    break
                lines = lines[:remaining] + ["    ..."]
            blocks.extend(lines)
            blocks.append("")
            total += len(lines) + 1

    return "\n".join(l for l in blocks if l.strip()).strip()
