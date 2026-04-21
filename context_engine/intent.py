"""Intent detection, flow analysis, failure extraction, and structured formatting."""

from __future__ import annotations

import ast
import re
import textwrap
from collections import Counter, deque
from typing import Any, Mapping

Node = dict[str, Any]
Edge = dict[str, str]

# ---------------------------------------------------------------------------
# 1. Intent detection
# ---------------------------------------------------------------------------

# Phrase patterns checked first (most specific signal).
_INTENT_PHRASES: dict[str, tuple[str, ...]] = {
    "debug": (
        "not working", "doesn't work", "does not work", "not work",
        "why is", "why does", "why isn't", "can't", "cannot",
        "not found", "keeps failing", "attribute error", "import error",
        "key error", "type error", "index error", "null pointer",
    ),
    "explain": (
        "how does", "how do", "how is", "what is", "what does",
        "walk me through", "show me how", "help me understand",
        "what happens", "explain how",
    ),
    "generate": (
        "add a", "add new", "create a", "create new", "implement a",
        "implement new", "write a", "build a", "how to add", "how to create",
        "how do i add", "how do i create",
    ),
}

# Single-word fallbacks checked in priority order.
_INTENT_WORDS: dict[str, frozenset[str]] = {
    "debug": frozenset({
        "fix", "bug", "broken", "fail", "failing", "crash", "error",
        "issue", "wrong", "exception", "traceback", "problem", "trouble",
        "stuck", "weird", "unexpected", "undefined",
    }),
    "explain": frozenset({
        "how", "explain", "flow", "works", "overview", "trace",
        "describe", "understand", "purpose",
    }),
    "generate": frozenset({
        "add", "create", "implement", "write", "build",
        "generate", "scaffold", "extend", "new",
    }),
}

# Combined filter set used by parse_query() to strip intent trigger words
# from the keyword list before code search.
INTENT_FILTER_WORDS: frozenset[str] = frozenset(
    word
    for words in _INTENT_WORDS.values()
    for word in words
) | frozenset({
    # Explain-specific words that aren't general code identifiers.
    "explain", "understand", "overview", "trace", "describe", "purpose",
})


def detect_intent(query: str) -> str:
    """Return one of: ``debug`` | ``explain`` | ``generate`` | ``lookup``.

    Detection order: phrase patterns → single-word triggers → fallback (lookup).
    Priority across intents: debug > explain > generate > lookup.
    """
    q = query.lower()
    tokens = set(q.split())

    for intent in ("debug", "explain", "generate"):
        for phrase in _INTENT_PHRASES.get(intent, ()):
            if phrase in q:
                return intent

    for intent in ("debug", "explain", "generate"):
        if tokens & _INTENT_WORDS[intent]:
            return intent

    return "lookup"


# ---------------------------------------------------------------------------
# 2. Flow extraction
# ---------------------------------------------------------------------------

def build_flow(
    nodes: list[Node],
    edges: list[Edge],
    entry_ids: list[str],
) -> list[str]:
    """Return bare function/method names in topological call order.

    Restricted to the kept function/method nodes and ``calls`` edges between
    them.  Entry points seed the queue first.  Cycles are appended at the end.
    """
    kept: dict[str, Node] = {
        n["id"]: n
        for n in nodes
        if n.get("type") in ("function", "method")
    }
    if not kept:
        return []

    succ: dict[str, list[str]] = {nid: [] for nid in kept}
    in_deg: dict[str, int] = {nid: 0 for nid in kept}
    for edge in edges:
        src, dst, etype = edge["from"], edge["to"], edge.get("type", "")
        if etype == "calls" and src in kept and dst in kept:
            succ[src].append(dst)
            in_deg[dst] += 1

    # Seed queue: entry points with no internal callers first.
    queue: deque[str] = deque()
    queued: set[str] = set()
    for eid in entry_ids:
        if eid in kept and in_deg[eid] == 0:
            queue.append(eid)
            queued.add(eid)
    for nid in kept:
        if nid not in queued and in_deg[nid] == 0:
            queue.append(nid)
            queued.add(nid)

    ordered: list[str] = []
    visited: set[str] = set()
    while queue:
        nid = queue.popleft()
        if nid in visited:
            continue
        visited.add(nid)
        _, _, name = nid.rpartition(":")
        ordered.append(name)
        for callee in succ[nid]:
            in_deg[callee] -= 1
            if in_deg[callee] == 0 and callee not in visited:
                queue.append(callee)

    # Append cycle members last.
    for nid in kept:
        if nid not in visited:
            _, _, name = nid.rpartition(":")
            ordered.append(name)

    return ordered


# ---------------------------------------------------------------------------
# 3. Failure extraction
# ---------------------------------------------------------------------------

def extract_failures(nodes: list[Node]) -> list[dict[str, str]]:
    """Scan code snippets for AST-level failure patterns.

    Detected patterns
    -----------------
    * ``raise <Exc>``     — explicit exception raises.
    * ``except <Exc>``    — exception handlers (potential silent swallowers).
    * ``if <guard>: raise / return False/None``  — validation checks.

    Returns a deduplicated list of ``{"node", "kind", "detail"}`` dicts.
    """
    results: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def _add(node_name: str, kind: str, detail: str) -> None:
        key = (node_name, kind, detail)
        if key not in seen:
            seen.add(key)
            results.append({"node": node_name, "kind": kind, "detail": detail})

    for node in nodes:
        code = node.get("code", "")
        if not code:
            continue
        _, _, name = node["id"].rpartition(":")

        try:
            tree = ast.parse(textwrap.dedent(code).strip())
        except SyntaxError:
            continue

        for child in ast.walk(tree):
            if isinstance(child, ast.Raise) and child.exc is not None:
                try:
                    exc_text = ast.unparse(child.exc)
                except Exception:
                    exc_text = "..."
                _add(name, "raises", exc_text)

            elif isinstance(child, ast.ExceptHandler) and child.type is not None:
                try:
                    exc_type = ast.unparse(child.type)
                except Exception:
                    exc_type = "Exception"
                _add(name, "catches", exc_type)

            elif isinstance(child, ast.If):
                body_is_guard = any(
                    isinstance(s, ast.Raise)
                    or (
                        isinstance(s, ast.Return)
                        and (
                            s.value is None
                            or (isinstance(s.value, ast.Constant)
                                and s.value.value in (False, None))
                        )
                    )
                    for s in child.body
                )
                if body_is_guard:
                    try:
                        cond = ast.unparse(child.test)
                    except Exception:
                        cond = "..."
                    _add(name, "check", cond)

    return results


# ---------------------------------------------------------------------------
# 4. Debug hints
# ---------------------------------------------------------------------------

_HINT_RULES: list[tuple[frozenset[str], str]] = [
    (frozenset({"secret", "key", "jwt", "bearer"}),
     "Check SECRET_KEY / JWT_SECRET is consistent across all services"),
    (frozenset({"token", "jwt", "exp", "expir", "expiry"}),
     "Verify token expiry (exp claim) and clock skew between services"),
    (frozenset({"password", "hash", "bcrypt", "argon", "pbkdf"}),
     "Confirm password hashing algorithm matches on the verify side"),
    (frozenset({"db", "database", "sql", "postgres", "mysql", "sqlite"}),
     "Check database connection, pool limits, and query parameters"),
    (frozenset({"permission", "role", "access", "guard", "authorize"}),
     "Verify user role/permission is being set and checked correctly"),
    (frozenset({"email", "smtp", "mail"}),
     "Check email format validation and SMTP/relay configuration"),
    (frozenset({"cors", "origin"}),
     "Review CORS origin whitelist and allowed-methods configuration"),
    (frozenset({"env", "config", "setting", "environ"}),
     "Ensure required environment variables are present and correctly typed"),
]


def debug_hints(
    failures: list[dict[str, str]],
    keywords: list[str],
) -> list[str]:
    """Return up to 6 actionable debug hints based on failures and keywords."""
    signal: set[str] = set(keywords)
    for f in failures:
        signal.update(f["detail"].lower().split())

    hints: list[str] = []
    for trigger_set, hint in _HINT_RULES:
        if trigger_set & signal:
            hints.append(hint)

    for f in failures:
        if len(hints) >= 6:
            break
        if f["kind"] == "raises":
            hints.append(
                f"{f['node']} raises {f['detail']} — trace the call site"
            )
        elif f["kind"] == "catches":
            hints.append(
                f"{f['node']} silently catches {f['detail']} — inspect handler"
            )

    return hints[:6]


# ---------------------------------------------------------------------------
# 5. Generate suggestions
# ---------------------------------------------------------------------------

_GENERATE_TEMPLATES: dict[str, list[str]] = {
    "auth": [
        "Create endpoint (e.g. POST /auth/token or /login)",
        "Fetch user record from DB by username or email",
        "Verify submitted password against stored hash",
        "Sign and return a JWT (set exp, sub, iat claims)",
        "Return refresh token if long-lived sessions are needed",
        "Add rate-limiting and account-lockout on repeated failures",
    ],
    "user": [
        "Create endpoint (e.g. POST /users or /register)",
        "Validate and sanitize input fields (email, username, password)",
        "Check for duplicate email/username in the database",
        "Hash password with bcrypt/argon2 before persisting",
        "Return 201 with the created user object",
        "Send a confirmation email if verification is required",
    ],
    "db": [
        "Define the ORM model / schema for the new entity",
        "Create a database migration for the new table/columns",
        "Implement a repository with CRUD operations",
        "Add indexes for frequently queried fields",
        "Wire the repository into the service layer",
    ],
    "default": [
        "Define the data model / schema",
        "Implement the core business logic in a service",
        "Expose via endpoint, CLI command, or service method",
        "Add input validation and structured error handling",
        "Write unit tests covering success and failure paths",
    ],
}


def suggest_implementation(keywords: list[str]) -> list[str]:
    """Return ordered implementation steps for a generate-mode query."""
    kw = set(keywords)
    if kw & {"login", "auth", "authenticate", "signin", "jwt", "token", "password"}:
        return _GENERATE_TEMPLATES["auth"]
    if kw & {"user", "account", "profile", "register", "signup"}:
        return _GENERATE_TEMPLATES["user"]
    if kw & {"db", "database", "sql", "model", "schema", "migration"}:
        return _GENERATE_TEMPLATES["db"]
    return _GENERATE_TEMPLATES["default"]


# ---------------------------------------------------------------------------
# 6. Graph-aware generate helpers
# ---------------------------------------------------------------------------

_ROUTER_TOKENS: frozenset[str] = frozenset({
    "router", "route", "routes", "api", "endpoint", "endpoints",
    "view", "views", "controller", "controllers", "handler", "handlers",
    "server", "app", "cli", "command", "commands",
})

_LAYER_SIGNALS: list[tuple[str, tuple[str, ...]]] = [
    ("Password Layer", ("password", "hash", "verify", "bcrypt", "argon", "pbkdf", "crypt")),
    ("Token Layer", ("token", "jwt", "access", "refresh", "bearer", "sign", "decode", "encode", "claim")),
    ("User Layer", ("user", "account", "profile", "register", "signup", "login", "auth")),
    ("DB Layer", ("db", "database", "session", "query", "orm", "repo", "repository")),
    ("Middleware Layer", ("middleware", "guard", "permission", "role", "scope", "policy")),
]


def find_integration_points(nodes: list[Node]) -> list[str]:
    """Return unique file paths that look like router/API/endpoint modules."""
    seen: set[str] = set()
    result: list[str] = []
    for node in nodes:
        fpath = node.get("file", "")
        if not fpath or fpath in seen:
            continue
        stem = re.split(r"[/\\]", fpath)[-1]
        stem = re.sub(r"\.(py|js|ts)$", "", stem, flags=re.IGNORECASE).lower()
        if any(tok in stem for tok in _ROUTER_TOKENS):
            seen.add(fpath)
            result.append(fpath)
    return result


def group_components(nodes: list[Node]) -> dict[str, list[str]]:
    """Bucket function/method nodes into functional layers by name pattern."""
    groups: dict[str, list[str]] = {}
    for node in nodes:
        if node.get("type") not in ("function", "method"):
            continue
        _, _, name = node["id"].rpartition(":")
        name_lower = name.lower()
        placed = False
        for layer_name, signals in _LAYER_SIGNALS:
            if any(sig in name_lower for sig in signals):
                groups.setdefault(layer_name, []).append(name)
                placed = True
                break
        if not placed:
            groups.setdefault("Other", []).append(name)
    return groups


def _detect_framework(nodes: list[Node]) -> str:
    """Detect web framework from code snippets. Scans all supplied nodes."""
    for node in nodes:
        code = node.get("code", "")
        if not code:
            continue
        if any(sig in code for sig in ("APIRouter", "Depends(", "HTTPException", "fastapi")):
            return "fastapi"
        if "fastapi" in code.lower():
            return "fastapi"
        if any(sig in code for sig in ("Blueprint", "@app.route", "Flask(")):
            return "flask"
        if "flask" in code.lower():
            return "flask"
        if any(sig in code for sig in ("HttpResponse", "render(request", "models.Model")):
            return "django"
        if "django" in code.lower():
            return "django"
    return "generic"


_ASYNC_PATTERN = re.compile(r"^\s*async def ", re.MULTILINE)
_AWAIT_PATTERN = re.compile(r"^\s+await ", re.MULTILINE)


def _detect_async(nodes: list[Node]) -> bool:
    """Return True if any node contains real async code (not just docstring mentions)."""
    for node in nodes:
        code = node.get("code", "")
        if _ASYNC_PATTERN.search(code) or _AWAIT_PATTERN.search(code):
            return True
    return False


def _find_fn(nodes: list[Node], fallback: str, *signals: str) -> tuple[str, bool]:
    """Return (name, found_in_graph).

    Searches function/method nodes for any name containing a signal token.
    When not found, returns (fallback, False) so callers can surface MISSING.
    """
    for node in nodes:
        if node.get("type") not in ("function", "method"):
            continue
        _, _, name = node["id"].rpartition(":")
        if any(sig in name.lower() for sig in signals):
            return name, True
    return fallback, False


def _short_path(fpath: str) -> str:
    """Return last two path segments, forward-slash separated."""
    parts = re.split(r"[/\\]", fpath.strip())
    return "/".join(p for p in parts[-2:] if p)


def _gen_auth_snippet(
    framework: str,
    is_async: bool,
    verify_fn: str,
    token_fn: str,
    user_fn: str,
    db_fn: str,
) -> str:
    a = "async " if is_async else ""
    aw = "await " if is_async else ""
    if framework == "fastapi":
        return (
            f'router = APIRouter()\n'
            f'\n'
            f'\n'
            f'@router.post("/login")\n'
            f'{a}def login(data: LoginRequest, db: AsyncSession = Depends({db_fn})):\n'
            f'    user = {aw}{user_fn}(db, data.email)\n'
            f'    if not user or not {verify_fn}(data.password, user.hashed_password):\n'
            f'        raise HTTPException(status_code=401, detail="Invalid credentials")\n'
            f'    token = {token_fn}(subject=str(user.id))\n'
            f'    return {{"access_token": token, "token_type": "bearer"}}'
        )
    if framework == "flask":
        return (
            f'auth_bp = Blueprint("auth", __name__)\n'
            f'\n'
            f'\n'
            f'@auth_bp.post("/login")\n'
            f'def login():\n'
            f'    data = request.get_json()\n'
            f'    user = {user_fn}(data["email"])\n'
            f'    if not user or not {verify_fn}(data["password"], user.hashed_password):\n'
            f'        abort(401)\n'
            f'    token = {token_fn}(user.id)\n'
            f'    return jsonify({{"access_token": token, "token_type": "bearer"}})'
        )
    return (
        f'{a}def login(email: str, password: str):\n'
        f'    user = {aw}{user_fn}(email)\n'
        f'    if not user or not {verify_fn}(password, user.hashed_password):\n'
        f'        raise AuthenticationError("Invalid credentials")\n'
        f'    return {token_fn}(user.id)'
    )


def _gen_user_snippet(
    framework: str,
    is_async: bool,
    user_fn: str,
    db_fn: str,
) -> str:
    a = "async " if is_async else ""
    aw = "await " if is_async else ""
    if framework == "fastapi":
        return (
            f'router = APIRouter()\n'
            f'\n'
            f'\n'
            f'@router.post("/users", status_code=201)\n'
            f'{a}def create_user(data: CreateUserRequest, db: AsyncSession = Depends({db_fn})):\n'
            f'    if {aw}{user_fn}(db, data.email):\n'
            f'        raise HTTPException(status_code=409, detail="Email already registered")\n'
            f'    user = {aw}save_user(db, data)\n'
            f'    return user'
        )
    return (
        f'{a}def create_user(email: str, password: str):\n'
        f'    if {aw}{user_fn}(email):\n'
        f'        raise ValueError("Email already registered")\n'
        f'    return {aw}save_user(email=email, hashed_password=hash_password(password))'
    )


def _gen_db_snippet(is_async: bool) -> str:
    if is_async:
        return (
            'class NewEntity(Base):\n'
            '    __tablename__ = "new_entities"\n'
            '    id = Column(Integer, primary_key=True)\n'
            '\n'
            'async def create_entity(db: AsyncSession, data: dict) -> NewEntity:\n'
            '    obj = NewEntity(**data)\n'
            '    db.add(obj)\n'
            '    await db.commit()\n'
            '    await db.refresh(obj)\n'
            '    return obj'
        )
    return (
        'class NewEntity(Base):\n'
        '    __tablename__ = "new_entities"\n'
        '    id = Column(Integer, primary_key=True)\n'
        '\n'
        'def create_entity(db: Session, data: dict) -> NewEntity:\n'
        '    obj = NewEntity(**data)\n'
        '    db.add(obj)\n'
        '    db.commit()\n'
        '    db.refresh(obj)\n'
        '    return obj'
    )


def _gen_generic_snippet(name: str, framework: str, is_async: bool) -> str:
    fn_name = re.sub(r"[^a-z0-9_]", "_", name.lower()).strip("_")
    a = "async " if is_async else ""
    aw = "await " if is_async else ""
    if framework == "fastapi":
        return (
            f'@router.post("/{fn_name}")\n'
            f'{a}def {fn_name}(data: dict):\n'
            f'    result = {aw}process_{fn_name}(data)\n'
            f'    return result'
        )
    return (
        f'{a}def {fn_name}(data):\n'
        f'    result = {aw}process(data)\n'
        f'    return result'
    )


def generate_code_snippet(
    keywords: list[str],
    kept_nodes: list[Node],
    all_nodes: list[Node],
    framework: str,
    is_async: bool,
) -> tuple[str, list[str]]:
    """Generate code grounded in actual graph nodes; return (snippet, missing_deps).

    Uses ``all_nodes`` (full graph) for existence checks so that functions which
    exist in the graph but weren't in the pruned result set are NOT flagged as
    missing.  Uses ``kept_nodes`` (query result) first to pick the most contextually
    relevant function name, falling back to a search across ``all_nodes``.
    """
    kw = set(keywords)

    def _find_best(fallback: str, *signals: str) -> tuple[str, bool]:
        """Find in kept first, then all_nodes; check existence against all_nodes."""
        name, found = _find_fn(kept_nodes, fallback, *signals)
        if not found:
            name, found = _find_fn(all_nodes, fallback, *signals)
        return name, found

    verify_fn, verify_found = _find_best("verify_password", "verify", "check_password", "validate_password")
    token_fn, token_found = _find_best("create_access_token", "create_access", "generate_token", "create_token", "sign_token")
    user_fn, user_found = _find_best("get_user_by_email", "get_user", "find_user", "fetch_user", "lookup_user")
    db_fn, db_found = _find_best("get_db", "get_db", "get_session", "db_session", "get_connection")

    missing: list[str] = []

    if kw & {"login", "auth", "authenticate", "signin", "jwt", "token", "password"}:
        if not user_found:
            missing.append(f"{user_fn}(db, email)  # retrieve user record from DB")
        if not verify_found:
            missing.append(f"{verify_fn}(plain, hashed)  # password verification")
        if not token_found:
            missing.append(f"{token_fn}(subject)  # JWT generation")
        if not db_found and framework in ("fastapi", "generic"):
            missing.append(f"{db_fn}()  # async DB session dependency")
        return _gen_auth_snippet(framework, is_async, verify_fn, token_fn, user_fn, db_fn), missing

    if kw & {"user", "account", "profile", "register", "signup"}:
        if not user_found:
            missing.append(f"{user_fn}(db, email)  # check for existing user")
        if not db_found and framework == "fastapi":
            missing.append(f"{db_fn}()  # async DB session dependency")
        return _gen_user_snippet(framework, is_async, user_fn, db_fn), missing

    if kw & {"db", "database", "sql", "model", "schema", "migration"}:
        return _gen_db_snippet(is_async), missing

    return _gen_generic_snippet(keywords[0] if keywords else "feature", framework, is_async), missing


# ---------------------------------------------------------------------------
# 6b. Integration target + import generation
# ---------------------------------------------------------------------------

def _file_to_module(fpath: str) -> str:
    """Convert an absolute file path to a dotted Python import path.

    Anchoring priority:
    1. If "app" is a path segment, anchor there (FastAPI/Django project convention).
    2. Otherwise find the first all-lowercase Python package name segment.

    Example: ``C:/Users/HP/project/app/core/security.py`` → ``app.core.security``
    Example: ``C:/Users/HP/project/context_engine/cli.py`` → ``context_engine.cli``
    """
    normalized = fpath.replace("\\", "/")
    parts = normalized.split("/")
    if parts:
        parts[-1] = re.sub(r"\.pyi?$", "", parts[-1])
    # Prefer "app" as anchor — avoids long system-path prefixes
    if "app" in parts:
        idx = parts.index("app")
        return ".".join(p for p in parts[idx:] if p)
    for i, part in enumerate(parts):
        if re.match(r"^[a-z][a-z0-9_]*$", part):
            return ".".join(p for p in parts[i:] if p)
    return ".".join(p for p in parts if p)


def find_integration_target(
    all_nodes: list[Node],
    keywords: list[str],
) -> dict[str, str]:
    """Return the best existing file to add generated code to, or suggest a new one.

    Returns ``{"action": "add"|"create", "file": "<path>"}``.
    """
    router_files: list[str] = []
    for node in all_nodes:
        if node.get("type") != "file":
            continue
        fpath = node["id"]
        stem = re.split(r"[/\\]", fpath)[-1]
        stem = re.sub(r"\.(py|js|ts)$", "", stem, flags=re.IGNORECASE).lower()
        if any(tok in stem for tok in _ROUTER_TOKENS):
            router_files.append(fpath)

    kw = set(keywords)
    if router_files:
        for f in router_files:
            f_lower = f.lower()
            if kw & {"login", "auth", "signin"} and any(t in f_lower for t in ("auth", "login", "security")):
                return {"action": "add", "file": _short_path(f)}
        return {"action": "add", "file": _short_path(router_files[0])}

    if kw & {"login", "auth", "authenticate", "signin"}:
        return {"action": "create", "file": "app/api/routes/auth.py"}
    if kw & {"user", "account", "profile", "register"}:
        return {"action": "create", "file": "app/api/routes/users.py"}
    return {"action": "create", "file": "app/api/routes/new.py"}


def generate_schema(keywords: list[str], framework: str) -> str:
    """Return a Pydantic BaseModel class definition for the request body.

    Only emitted for FastAPI (the only framework where Pydantic request models
    are idiomatic).  Returns an empty string for all other frameworks.
    """
    if framework != "fastapi":
        return ""
    kw = set(keywords)
    if kw & {"login", "auth", "authenticate", "signin"}:
        return (
            "class LoginRequest(BaseModel):\n"
            "    email: str\n"
            "    password: str"
        )
    if kw & {"user", "register", "signup", "account"}:
        return (
            "class CreateUserRequest(BaseModel):\n"
            "    email: str\n"
            "    password: str\n"
            "    username: str | None = None"
        )
    return ""


# Known third-party package prefixes — used to separate framework imports from local ones.
_THIRD_PARTY_PREFIXES: frozenset[str] = frozenset({
    "fastapi", "pydantic", "sqlalchemy", "starlette", "uvicorn",
    "flask", "django", "aiohttp", "httpx", "requests",
    "celery", "redis", "boto3", "jose", "passlib", "bcrypt",
})


def generate_imports(
    framework: str,
    is_async: bool,
    fn_names: list[str],
    all_nodes: list[Node],
    has_schema: bool = False,
) -> list[str]:
    """Generate properly grouped import statements.

    Group 1 — third-party (fastapi, pydantic, sqlalchemy, …), alphabetical.
    Group 2 — local project imports (one ``from <module> import …`` per file).
    Groups separated by a blank line.
    """
    third_party: list[str] = []
    if framework == "fastapi":
        third_party.append("from fastapi import APIRouter, Depends, HTTPException")
        if has_schema:
            third_party.append("from pydantic import BaseModel")
        if is_async:
            third_party.append("from sqlalchemy.ext.asyncio import AsyncSession")
    elif framework == "flask":
        third_party.append("from flask import Blueprint, abort, jsonify, request")
    elif has_schema:
        third_party.append("from pydantic import BaseModel")

    # Build fn_name → source_file map from graph
    file_map: dict[str, str] = {}
    for node in all_nodes:
        if node.get("type") not in ("function", "method"):
            continue
        _, _, name = node["id"].rpartition(":")
        if name in fn_names and name not in file_map:
            fpath = node.get("file", "")
            if fpath:
                file_map[name] = fpath

    # Group by module path, emit one from-import per module
    module_fns: dict[str, list[str]] = {}
    for name, fpath in file_map.items():
        module = _file_to_module(fpath)
        module_fns.setdefault(module, []).append(name)

    local: list[str] = []
    for module in sorted(module_fns):
        # Skip any module that resolves to a known third-party package
        pkg = module.split(".")[0]
        if pkg in _THIRD_PARTY_PREFIXES:
            continue
        fns = ", ".join(sorted(module_fns[module]))
        local.append(f"from {module} import {fns}")

    lines: list[str] = []
    lines.extend(third_party)
    if third_party and local:
        lines.append("")
    lines.extend(local)
    return lines


# ---------------------------------------------------------------------------
# 6c. Missing implementations + debug fix generation
# ---------------------------------------------------------------------------

# Minimal working implementations for commonly missing auth dependencies.
# Each entry: fn_name → (sync_impl, async_impl)
_MISSING_IMPL_TEMPLATES: dict[str, tuple[str, str]] = {
    "get_user_by_email": (
        "def get_user_by_email(db: Session, email: str):\n"
        "    return db.query(User).filter(User.email == email).first()",
        "async def get_user_by_email(db: AsyncSession, email: str):\n"
        "    result = await db.execute(select(User).where(User.email == email))\n"
        "    return result.scalar_one_or_none()",
    ),
    "verify_password": (
        "def verify_password(plain_password: str, hashed_password: str) -> bool:\n"
        '    return pwd_context.verify(plain_password, hashed_password)',
        "def verify_password(plain_password: str, hashed_password: str) -> bool:\n"
        '    return pwd_context.verify(plain_password, hashed_password)',
    ),
    "create_access_token": (
        "def create_access_token(subject: str) -> str:\n"
        "    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)\n"
        '    payload = {"sub": subject, "exp": expire}\n'
        "    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)",
        "def create_access_token(subject: str) -> str:\n"
        "    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)\n"
        '    payload = {"sub": subject, "exp": expire}\n'
        "    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)",
    ),
    "get_db": (
        "def get_db():\n"
        "    db = SessionLocal()\n"
        "    try:\n"
        "        yield db\n"
        "    finally:\n"
        "        db.close()",
        "async def get_db() -> AsyncGenerator[AsyncSession, None]:\n"
        "    async with AsyncSessionLocal() as session:\n"
        "        yield session",
    ),
}


def generate_missing_impl(missing_deps: list[str], is_async: bool) -> str:
    """Return concatenated minimal implementations for each missing dependency."""
    impls: list[str] = []
    for dep in missing_deps:
        fn_name = dep.split("(")[0].strip()
        if fn_name in _MISSING_IMPL_TEMPLATES:
            sync_impl, async_impl = _MISSING_IMPL_TEMPLATES[fn_name]
            impls.append(async_impl if is_async else sync_impl)
    return "\n\n".join(impls)


# Debug fix rules: (trigger_words, check, root_cause, fix_code)
_DEBUG_FIX_RULES: list[tuple[frozenset[str], str, str, str]] = [
    (
        frozenset({"jwt", "token", "bearer", "secret", "key", "exp", "expir", "claim"}),
        "Token expiry and SECRET_KEY consistency across services",
        "Token expired, or signed/verified with mismatched SECRET_KEY",
        'ALGORITHM = "HS256"\nACCESS_TOKEN_EXPIRE_MINUTES = 30\n\n'
        '# Inspect a live token:\npayload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])\n'
        'if payload["exp"] < datetime.utcnow().timestamp():\n'
        '    raise HTTPException(status_code=401, detail="Token expired")',
    ),
    (
        frozenset({"password", "hash", "bcrypt", "argon", "pbkdf", "verify", "credential"}),
        "Password hash algorithm and context config on both hash + verify sides",
        "Password hashed with different settings or algorithm than verification",
        'from passlib.context import CryptContext\n\n'
        'pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")\n\n'
        'def verify_password(plain: str, hashed: str) -> bool:\n'
        '    return pwd_context.verify(plain, hashed)',
    ),
    (
        frozenset({"cors", "origin", "allow", "preflight"}),
        "CORSMiddleware present in app startup; allow_origins covers frontend URL",
        "CORS policy blocking pre-flight or credential requests from the frontend",
        'from fastapi.middleware.cors import CORSMiddleware\n\n'
        'app.add_middleware(\n'
        '    CORSMiddleware,\n'
        '    allow_origins=["http://localhost:3000"],\n'
        '    allow_credentials=True,\n'
        '    allow_methods=["*"],\n'
        '    allow_headers=["*"],\n'
        ')',
    ),
    (
        frozenset({"db", "database", "session", "connect", "pool", "sql", "sqlite"}),
        "DB connection pool size, session lifecycle, and async session factory config",
        "Session not committed, not closed, or connection pool limit reached",
        "async with AsyncSessionLocal() as db:\n"
        "    async with db.begin():\n"
        "        result = await db.execute(\n"
        "            select(User).where(User.email == email)\n"
        "        )\n"
        "    # session commits + closes automatically on context exit",
    ),
    (
        frozenset({"permission", "role", "forbidden", "403", "scope", "access", "guard"}),
        "User role populated on login and permission check logic correct",
        "Permission denied — role not assigned, or check condition is inverted",
        'user = await get_user_by_email(db, email)\n'
        'print(f"roles: {user.roles}")\n'
        'if "admin" not in user.roles:\n'
        '    raise HTTPException(status_code=403, detail="Forbidden")',
    ),
]


def generate_debug_fix(
    failures: list[dict[str, str]],
    keywords: list[str],
) -> tuple[list[str], list[str], str]:
    """Return (checks, root_causes, fix_code) for debug output.

    Matches the combined keyword+failure signal against ``_DEBUG_FIX_RULES``.
    Falls back to a generic trace snippet when no rule fires.
    """
    signal: set[str] = set(keywords)
    for f in failures:
        signal.update(f["detail"].lower().split())

    for trigger_set, check, root_cause, fix in _DEBUG_FIX_RULES:
        if trigger_set & signal:
            return [check], [root_cause], fix

    # Fallback: generate a minimal trace wrapper around the first failure site.
    if failures:
        f = failures[0]
        if f["kind"] == "raises":
            fix = (
                f"try:\n    result = {f['node']}(...)\n"
                f"except Exception as e:\n    print(f'Error in {f['node']}: {{e}}')\n    raise"
            )
        elif f["kind"] == "catches":
            fix = (
                f"import logging\nlogger = logging.getLogger(__name__)\n\n"
                f"try:\n    ...\nexcept {f['detail']} as e:\n"
                f"    logger.exception('Unexpected error in {f['node']}')\n    raise"
            )
        else:
            fix = f"assert <condition>, '<condition> must hold before calling {f['node']}()'"
        return (
            [f"Inspect failure site: {f['node']}"],
            [f"{f['node']} {f['kind']}: {f['detail']}"],
            fix,
        )

    return [], [], ""


# ---------------------------------------------------------------------------
# 7. Output formatters
# ---------------------------------------------------------------------------

_DIVIDER = "-" * 72


def _bare(node_id: str) -> str:
    _, _, name = node_id.rpartition(":")
    return name or node_id


def _token_line(result: Mapping[str, Any]) -> str:
    rt = result.get("token_estimate_raw", 0)
    ct = result.get("token_estimate", 0)
    if rt > 0 and ct <= rt:
        suffix = f"-{100 * (rt - ct) // rt}%"
    else:
        suffix = "no compression"
    return f"tokens        : ~{ct} (raw ~{rt}, {suffix})"


def _format_debug(
    result: Mapping[str, Any],
    flow: list[str],
    failures: list[dict[str, str]],
) -> str:
    lines: list[str] = [
        "intent        : DEBUG",
        f"keywords      : {', '.join(result.get('keywords', []))}",
        "",
    ]

    if result.get("entry_points"):
        lines.append("ENTRY POINTS:")
        for ep in result["entry_points"]:
            lines.append(f"  {_bare(ep)}")
        lines.append("")

    # Annotated failure path
    if flow:
        failure_map: dict[str, list[str]] = {}
        for f in failures:
            if f["kind"] == "raises":
                note = f"raises {f['detail']}"
            elif f["kind"] == "catches":
                note = f"catches {f['detail']}"
            else:
                note = f"guards on `{f['detail']}`"
            failure_map.setdefault(f["node"], []).append(note)

        lines.append("FAILURE PATH:")
        for fn in flow:
            notes = failure_map.get(fn, [])
            suffix = "  ->  " + ", ".join(notes) if notes else "  ->  OK"
            lines.append(f"  {fn}{suffix}")
        lines.append("")

    checks, root_causes, fix_code = generate_debug_fix(failures, result.get("keywords", []))

    if checks:
        lines.append("CHECK:")
        for c in checks:
            lines.append(f"  - {c}")
        lines.append("")

    if root_causes:
        lines.append("ROOT CAUSE (LIKELY):")
        for rc in root_causes:
            lines.append(f"  - {rc}")
        lines.append("")

    if fix_code:
        lines += ["FIX:", _DIVIDER, fix_code, _DIVIDER, ""]

    nodes_sel = result.get("nodes_selected", [])
    lines += [
        f"nodes         : {len(nodes_sel)}",
        _token_line(result),
        "",
    ]

    if result.get("context"):
        lines += [_DIVIDER, result["context"], _DIVIDER]

    return "\n".join(lines)


def _format_explain(
    result: Mapping[str, Any],
    flow: list[str],
) -> str:
    kws = result.get("keywords", [])
    domain = kws[0].upper() if kws else "CODE"

    lines: list[str] = [
        "intent        : EXPLAIN",
        f"keywords      : {', '.join(kws)}",
        "",
        f"{domain} FLOW:",
    ]

    if flow:
        for i, step in enumerate(flow, 1):
            lines.append(f"  {i}. {step}")
    else:
        lines.append("  (no flow data - run `context-engine index` first)")

    lines += [
        "",
        f"nodes         : {len(result.get('nodes_selected', []))}",
        _token_line(result),
        "",
    ]

    if result.get("context"):
        lines += [_DIVIDER, result["context"], _DIVIDER]

    return "\n".join(lines)


def _format_generate(
    result: Mapping[str, Any],
    kept_nodes: list[Node],
    all_nodes: list[Node],
) -> str:
    kws = result.get("keywords", [])
    framework = _detect_framework(all_nodes)
    is_async = _detect_async(all_nodes)
    groups = group_components(kept_nodes)
    target = find_integration_target(all_nodes, kws)
    schema = generate_schema(kws, framework)
    snippet, missing = generate_code_snippet(kws, kept_nodes, all_nodes, framework, is_async)

    # Collect graph-verified function names referenced in the snippet for imports.
    fn_names_in_snippet: list[str] = [
        node["id"].rpartition(":")[2]
        for node in all_nodes
        if node.get("type") in ("function", "method")
        and node["id"].rpartition(":")[2] in snippet
    ]
    import_lines = generate_imports(framework, is_async, fn_names_in_snippet, all_nodes, has_schema=bool(schema))

    concurrency = "async" if is_async else "sync"
    lines: list[str] = [
        "INTENT:",
        "GENERATE",
        "",
        f"framework     : {framework}  ({concurrency})",
        "",
    ]

    if groups:
        lines.append("AUTH STACK:" if kws and set(kws) & {"login", "auth", "authenticate", "signin", "token", "jwt"} else "DETECTED STACK:")
        for layer, fns in groups.items():
            lines.append(f"  {layer}:")
            for fn in fns:
                lines.append(f"    - {fn}()")
        lines.append("")

    action_verb = "Add to" if target["action"] == "add" else "Create"
    lines += [
        "INTEGRATION TARGET:",
        f"  {action_verb}: {target['file']}",
        "",
    ]

    if import_lines:
        lines += ["IMPORTS:", _DIVIDER]
        lines.extend(import_lines)
        lines += [_DIVIDER, ""]

    if schema:
        lines += ["SCHEMA:", _DIVIDER, schema, _DIVIDER, ""]

    lines += ["GENERATED CODE:", _DIVIDER, snippet, _DIVIDER, ""]

    if missing:
        missing_code = generate_missing_impl(missing, is_async)
        if missing_code:
            lines += ["MISSING IMPLEMENTATION:", _DIVIDER, missing_code, _DIVIDER, ""]
        else:
            # No template available — fall back to listing
            lines.append("MISSING (implement before deploying):")
            for dep in missing:
                lines.append(f"  - {dep}")
            lines.append("")

    # Integration guidance
    action_verb = "Add to" if target["action"] == "add" else "Create"
    insert_lines = [
        "INSERTION HINT:",
        f"  1. {action_verb}: {target['file']}",
    ]
    if framework == "fastapi":
        insert_lines.append("  2. Register in main app:  app.include_router(router, prefix='/auth')")
    insert_lines.append("")
    lines.extend(insert_lines)

    if result.get("context"):
        lines += ["EXISTING CODE:", _DIVIDER, result["context"], _DIVIDER]

    return "\n".join(lines)


def _format_lookup(result: Mapping[str, Any]) -> str:
    lines: list[str] = [
        "intent        : LOOKUP",
        f"keywords      : {', '.join(result.get('keywords', []))}",
        "",
    ]

    eps = result.get("entry_points", [])
    if eps:
        lines.append(f"entry points  : {len(eps)}")
        for ep in eps:
            lines.append(f"  * {_bare(ep)}")
        lines.append("")

    cat_counts = Counter(result.get("categories", {}).values())
    lines.append(
        f"nodes         : {len(result.get('nodes_selected', []))}  "
        + "  ".join(f"{k}={v}" for k, v in sorted(cat_counts.items()))
    )

    inline_count = sum(len(v) for v in result.get("inline_hints", {}).values())
    if inline_count:
        lines.append(f"inline hints  : {inline_count}")

    lines += ["", _token_line(result), ""]

    if result.get("context"):
        lines += [_DIVIDER, result["context"], _DIVIDER]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 8. Public entry point
# ---------------------------------------------------------------------------

def format_intent_output(
    result: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> str:
    """Produce intent-aware CLI output for a query result.

    Dispatches to one of four formatters:

    * ``debug``    → CALL FLOW + POTENTIAL FAILURES + DEBUG HINTS + code
    * ``explain``  → numbered FLOW + code
    * ``generate`` → EXISTING COMPONENTS + SUGGESTED IMPLEMENTATION + code
    * ``lookup``   → entry points + stats + code  (current behaviour)
    """
    intent = result.get("intent", "lookup")
    node_map: dict[str, Any] = {n["id"]: n for n in graph.get("nodes", [])}
    graph_nodes: list[Node] = list(node_map.values())
    kept: list[Node] = [
        node_map[nid]
        for nid in result.get("nodes_selected", [])
        if nid in node_map
    ]
    edges: list[Edge] = graph.get("edges", [])
    entry_ids: list[str] = result.get("entry_points", [])

    flow = build_flow(kept, edges, entry_ids)
    failures = extract_failures(kept)

    if intent == "debug":
        return _format_debug(result, flow, failures)
    if intent == "explain":
        return _format_explain(result, flow)
    if intent == "generate":
        return _format_generate(result, kept, graph_nodes)
    return _format_lookup(result)
