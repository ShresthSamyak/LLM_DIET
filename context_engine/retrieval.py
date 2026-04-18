"""Context retrieval engine: query → relevant code context."""

from __future__ import annotations

import re
from collections import deque
from typing import Any, TypedDict

from .compressor import compress_code
from .intent import INTENT_FILTER_WORDS, detect_intent
from .pruner import PruneResult, importance_score, prune

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Node = dict[str, Any]
Edge = dict[str, str]


class Graph(TypedDict):
    nodes: list[Node]
    edges: list[Edge]


class QueryResult(TypedDict):
    intent: str
    keywords: list[str]
    entry_points: list[str]
    nodes_selected: list[str]
    categories: dict[str, str]
    inline_hints: dict[str, list[str]]
    token_estimate: int
    token_estimate_raw: int
    context: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# _DEBUG_TRIGGERS kept for reference; intent detection now lives in intent.py.
_DEBUG_TRIGGERS: frozenset[str] = frozenset({"fix", "bug", "error", "issue", "debug", "crash", "fail", "broken"})
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "it", "be", "do", "does", "me", "my", "we", "us", "i", "with",
    "this", "that", "have", "has", "not", "are", "was", "but", "get", "set",
    "work", "works", "working", "show", "tell", "want", "need", "help",
})

_MAX_NODES = 12          # hard cap on nodes passed to the pruner
_MAX_DEPTH = 2
_RELEVANCE_THRESHOLD = 2  # minimum keyword-relevance score for non-adjacent nodes

# Keywords whose presence in a file path marks the module as high-relevance.
# Keep this list NARROW — generic tokens like 'token' or 'session' appear in
# unrelated file names and would grant false module credit.
_MODULE_KEYWORDS: frozenset[str] = frozenset({
    "auth", "login", "security",
    "user", "account", "permission", "credential",
})
_MODULE_HIGH_SCORE = 3   # module score threshold that triggers a per-node boost


# ---------------------------------------------------------------------------
# Step 1: Query understanding
# ---------------------------------------------------------------------------

def parse_query(query: str) -> dict[str, Any]:
    """Return intent + keywords extracted from a raw query string.

    Intent is one of ``debug | explain | generate | lookup`` — detected by
    :func:`intent.detect_intent` using phrase patterns then single-word
    triggers in priority order.

    Keywords are lower-cased tokens stripped of stop words and intent trigger
    words so that the code-search stage only sees meaningful symbol names.
    """
    intent = detect_intent(query)
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", query.lower())
    _filter = _STOP_WORDS | INTENT_FILTER_WORDS
    keywords = [t for t in tokens if t not in _filter]
    seen: set[str] = set()
    unique_kw: list[str] = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_kw.append(kw)
    return {"intent": intent, "keywords": unique_kw}

# ---------------------------------------------------------------------------
# Step 2: Entry point detection — fuzzy tokens + synonyms + fan-in
# ---------------------------------------------------------------------------

# Files whose names tend to host HTTP/CLI/UI entry points.
_ENTRY_FILE_TOKENS: frozenset[str] = frozenset({
    "auth", "routes", "route", "views", "view", "api", "handlers", "handler",
    "controllers", "controller", "endpoints", "endpoint", "server", "app",
    "main", "cli", "command", "commands", "resolvers", "resolver",
})

# File-name stems that receive an *extra* strong boost as entry-point hosts.
# These are the most likely homes for auth/security entry points specifically.
_ENTRY_BOOST_FILES: frozenset[str] = frozenset({
    "auth", "authentication", "authorize", "authorization",
    "security", "login", "logout", "signin", "signup",
    "middleware", "guard", "permission", "credentials",
    "routes", "router",
})

# Path segments that boost relevance — signals the node lives in a core module.
_IMPORTANT_DIRS: frozenset[str] = frozenset({"auth", "service", "services", "core", "api"})

# Path segments that indicate low-signal test/fixture files.
_TEST_PATH_TOKENS: frozenset[str] = frozenset({"test", "tests", "mock", "mocks", "fixture", "fixtures"})

# Domain-specific path keywords that are allowed to qualify a file as a
# relevant module for entry-point selection.
#
# Critically, generic terms that overlap with auth synonyms but mean something
# different in a path context are EXCLUDED:
#   - "session"  → usually app/db/session.py (DB session), not an auth session
#   - "data"     → data models, not auth
#   - "service"  → generic service layer
#   - "store"    → storage layer
#
# A path must contain one of THESE tokens (and also be weight>=2) to pass
# the hard module gate in ``_is_relevant_module``.
_ENTRY_MODULE_KEYWORDS: frozenset[str] = frozenset({
    "auth", "authentication", "authorize", "authorization",
    "security", "login", "logout", "signin", "signup",
    "jwt", "token", "credential", "credentials", "password",
    "permission", "permissions", "guard", "middleware",
    "user", "account", "identity",
})

# Rule-based keyword expansion map.
#
# Each key is a canonical query token.  Values are *first-level* synonyms —
# terms that are semantically equivalent or very closely related.  A second
# expansion pass then fans out from those synonyms to produce a weaker
# "secondary" tier (see ``_expand_keywords``).
#
# Extend this table freely; no other code needs to change.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    # ── Authentication / identity ────────────────────────────────────────────
    "login":          ("auth", "authenticate", "signin", "token", "session", "jwt"),
    "signin":         ("auth", "authenticate", "login", "token", "session"),
    "logout":         ("signout", "auth", "session"),
    "auth":           ("login", "authenticate", "signin", "token", "security", "jwt"),
    "authenticate":   ("login", "auth", "signin", "token", "credential"),
    "jwt":            ("token", "auth", "bearer", "claim"),
    "token":          ("jwt", "auth", "session", "bearer", "credential"),
    "session":        ("token", "cookie", "auth", "login"),
    "security":       ("auth", "permission", "guard", "middleware"),
    "permission":     ("security", "auth", "guard", "role", "access"),
    "credential":     ("auth", "token", "password", "secret"),
    "password":       ("credential", "hash", "encrypt"),
    # ── User / account ──────────────────────────────────────────────────────
    "user":           ("account", "profile", "member"),
    "account":        ("user", "profile"),
    "profile":        ("user", "account"),
    # ── Errors / failures ───────────────────────────────────────────────────
    "bug":            ("error", "exception", "failure", "issue", "defect"),
    "error":          ("exception", "fail", "failure", "bug", "err"),
    "exception":      ("error", "fail", "bug"),
    "fail":           ("error", "exception", "failure"),
    "failure":        ("error", "fail", "exception", "bug"),
    # ── Database ────────────────────────────────────────────────────────────
    "db":             ("database", "sql", "query", "postgres", "mysql", "sqlite"),
    "database":       ("db", "sql", "query", "postgres", "mysql", "store"),
    "sql":            ("db", "database", "query", "postgres", "mysql"),
    "postgres":       ("db", "database", "sql"),
    "mysql":          ("db", "database", "sql"),
    "sqlite":         ("db", "database", "sql"),
    # ── Cache / queue ───────────────────────────────────────────────────────
    "cache":          ("redis", "memcache", "store", "ttl"),
    "redis":          ("cache", "queue", "store"),
    "queue":          ("task", "job", "worker", "celery", "redis"),
    # ── CRUD operations ─────────────────────────────────────────────────────
    "delete":         ("remove", "destroy", "drop"),
    "remove":         ("delete", "destroy"),
    "create":         ("add", "insert", "make", "new", "init"),
    "add":            ("create", "insert"),
    "update":         ("edit", "modify", "patch", "put"),
    "edit":           ("update", "modify"),
    "fetch":          ("get", "retrieve", "load", "read", "pull"),
    "get":            ("fetch", "retrieve", "read"),
    "save":           ("store", "persist", "write", "commit"),
    "store":          ("save", "persist", "cache"),
    "query":          ("search", "find", "lookup", "filter", "select"),
    "search":         ("query", "find", "lookup", "filter"),
    # ── Code processing ─────────────────────────────────────────────────────
    "parse":          ("decode", "deserialize", "lex", "tokenize"),
    "render":         ("format", "display", "emit", "serialize"),
    "validate":       ("check", "verify", "sanitize"),
    "verify":         ("validate", "check"),
    "compress":       ("shrink", "reduce", "minify"),
    "prune":          ("trim", "cut", "filter"),
    # ── Config / logging ────────────────────────────────────────────────────
    "config":         ("setting", "env", "configuration", "option"),
    "setting":        ("config", "env", "option"),
    "log":            ("logger", "logging", "trace", "audit"),
    "logger":         ("log", "logging"),
}

_FAN_IN_HIGH = 3                 # ≥N incoming call edges → high fan-in
_MAX_ENTRIES = 5                 # top-5 entry points
_SCORE_KEYWORD = 3               # exact keyword token match on symbol name
_SCORE_FILE_KW = 2               # exact keyword token match on file path
_SCORE_SYNONYM_NAME = 2          # synonym hit on symbol name
_SCORE_SYNONYM_FILE = 1          # synonym hit on file path only
_SCORE_BOOST_FILE = 3            # file is in _ENTRY_BOOST_FILES (auth.py, routes, …)
_SCORE_IMPORTANT_DIR = 2         # node lives in auth/service/core/api
_SCORE_FILE_HEURISTIC = 1        # file name resembles an entry-point module
_SCORE_FAN_IN = 2                # high incoming-call count
_PENALTY_TEST_FILE = -3          # node lives in a test/mock/fixture path
_ENTRY_MIN_SCORE = 3             # hard floor: a candidate must reach this to qualify

# Regex fragments for camelCase splitting.
_CAMEL_SPLIT = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
# Strip a language extension whenever it's followed by a separator or end
# of string — so `auth.py:login` loses the `py` noise token.
_EXT_STRIP = re.compile(
    r"\.(py|pyi|pyx|js|ts|tsx|jsx|rb|go|java|rs)(?=[:./]|$)",
    re.IGNORECASE,
)


def _tokenize(identifier: str) -> list[str]:
    """Split snake_case / camelCase / dotted / path-like strings into lower tokens."""
    identifier = _EXT_STRIP.sub("", identifier)
    tokens: list[str] = []
    for word in re.split(r"[^a-zA-Z0-9]+", identifier):
        if not word:
            continue
        parts = _CAMEL_SPLIT.findall(word) or [word]
        tokens.extend(p.lower() for p in parts)
    return tokens


def _expand_keywords(
    keywords: list[str],
) -> tuple[set[str], set[str], set[str]]:
    """Return ``(primary, synonyms, secondary)`` token sets.

    Three tiers of signal strength
    --------------------------------
    * **primary** — the user's own keywords, tokenised.  These must appear in
      a node's name or file path for the entry-point hard gate to pass.
    * **synonyms** — first-level expansions from ``_SYNONYMS``.  They add
      score bonuses but cannot pass the hard gate alone.
    * **secondary** — second-level expansions (synonyms of synonyms).  They
      provide a weak additional signal (+1) for code that uses indirect
      terminology (e.g. ``jwt`` or ``session`` when the user queried
      ``login``).  They cannot affect the hard gate at all.

    Example (query = "login")
    -------------------------
    primary   = {"login"}
    synonyms  = {"auth", "authenticate", "signin", "token", "session", "jwt"}
    secondary = {"bearer", "claim", "cookie", "credential", "security", …}
    """
    primary: set[str] = set()
    for kw in keywords:
        primary.update(_tokenize(kw))

    # Level-1 expansion.
    synonyms: set[str] = set()
    for kw in primary:
        for syn in _SYNONYMS.get(kw, ()):
            if syn not in primary:
                synonyms.add(syn)

    # Level-2 expansion: fan out from synonyms, excluding already-known tokens.
    secondary: set[str] = set()
    for syn in synonyms:
        for syn2 in _SYNONYMS.get(syn, ()):
            if syn2 not in primary and syn2 not in synonyms:
                secondary.add(syn2)

    return primary, synonyms, secondary


# Terms too generic to carry meaningful signal — weight 0 when reached via expansion.
_GENERIC_TERMS: frozenset[str] = frozenset({
    "get", "set", "find", "filter", "select", "check", "run", "do",
    "make", "new", "init", "build", "load", "read", "write", "store",
    "item", "data", "value", "result", "info", "obj", "list", "dict",
})


def _weighted_keywords(raw_keywords: list[str]) -> dict[str, int]:
    """Return a mapping of ``token -> weight`` for the full expanded keyword set.

    Weight tiers
    ------------
    * ``3`` — user's exact words (highest confidence).
    * ``2`` — strong first-level synonyms (close semantic match).
    * ``1`` — second-level / related concepts (weak signal).
    * ``0`` — generic terms (filtered out; not included in result).

    Example (query = "login")
    -------------------------
    ``{"login": 3, "auth": 2, "jwt": 2, "token": 2, "session": 2,
       "authenticate": 2, "signin": 2, "bearer": 1, "cookie": 1, ...}``
    """
    weights: dict[str, int] = {}

    # Exact user tokens — weight 3.
    primary: set[str] = set()
    for kw in raw_keywords:
        for tok in _tokenize(kw):
            primary.add(tok)
            weights[tok] = 3

    # First-level synonyms — weight 2 (skip generic terms).
    synonyms: set[str] = set()
    for kw in primary:
        for syn in _SYNONYMS.get(kw, ()):
            if syn not in primary and syn not in _GENERIC_TERMS:
                synonyms.add(syn)
                if weights.get(syn, 0) < 2:
                    weights[syn] = 2

    # Second-level synonyms — weight 1 (skip generic terms).
    for syn in synonyms:
        for syn2 in _SYNONYMS.get(syn, ()):
            if syn2 not in primary and syn2 not in synonyms and syn2 not in _GENERIC_TERMS:
                if syn2 not in weights:   # don't downgrade a higher-tier token
                    weights[syn2] = 1

    return weights


def _compute_fan_in(edges: list[Edge]) -> dict[str, int]:
    """Count incoming ``calls`` edges per node — proxy for importance."""
    fan_in: dict[str, int] = {}
    for edge in edges:
        if edge.get("type") == "calls":
            fan_in[edge["to"]] = fan_in.get(edge["to"], 0) + 1
    return fan_in


def _is_test_path(file_part: str) -> bool:
    """Return True when the file path contains a test/mock/fixture segment."""
    path_tokens = set(_tokenize(file_part))
    return bool(path_tokens & _TEST_PATH_TOKENS)


def _is_important_dir(file_part: str) -> bool:
    """Return True when the file path contains a high-signal directory name."""
    segments = {seg.lower() for seg in re.split(r"[/\\]", file_part) if seg}
    return bool(segments & _IMPORTANT_DIRS)


def _is_relevant_module(file_part: str, kw_weights: dict[str, int]) -> bool:
    """Hard module gate: True only if the file path contains a domain-specific
    strong keyword.

    Two conditions must BOTH hold for a file to qualify:

    1. The path token is **strong** (weight >= 2 in kw_weights) — excludes
       weak second-level expansions.
    2. The path token is in ``_ENTRY_MODULE_KEYWORDS`` — excludes terms that
       are weight-2 synonyms but carry a different meaning in a path context.

    Example that this fixes
    -----------------------
    Query = "login"  →  "session" gets weight 2 (synonym).
    ``app/db/session.py`` path tokens = {"app", "db", "session"}.
    ``session`` IS weight-2 but IS NOT in ``_ENTRY_MODULE_KEYWORDS``  →  DROPPED.
    ``core/security.py`` path tokens include ``security`` which IS in the
    allowlist  →  PASSES.
    """
    file_tokens = set(_tokenize(file_part))
    # Intersection of weight-2+ keywords AND the strict domain allowlist.
    qualifying_kws = {
        k for k, w in kw_weights.items()
        if w >= 2 and k in _ENTRY_MODULE_KEYWORDS
    }
    return bool(file_tokens & qualifying_kws)


def _score_node(
    node: Node,
    kw_weights: dict[str, int],
    fan_in: dict[str, int],
) -> int:
    """Score a node's suitability as an entry point using weighted keywords.

    Hard gate
    ---------
    A node must have at least one token match with weight >= 2 (i.e. an exact
    user keyword or a strong synonym) in its symbol name OR file path.
    Weight-1 (related concept) or weight-0 (generic) matches alone cannot
    pass the gate, preventing distant noise from becoming an entry point.

    Scoring rubric
    --------------
    Name/path match scores = ``weight * position_multiplier``:
    * Symbol name hit:  ``weight * 1.5`` (rounded down)  — name is strongest signal.
    * File path hit:    ``weight * 1``                   — useful but weaker.
    Path-level boosts applied on top:
    * ``+3`` file stem in _ENTRY_BOOST_FILES (auth.py, security.py, routes…).
    * ``+2`` file lives in important directory (auth/service/core/api).
    * ``+1`` file name resembles a generic entry-point module.
    * ``+2`` high incoming-call count (fan-in >= 3).
    * ``-3`` inside a test/mock/fixture path.

    Returns ``0`` when the hard gate is not satisfied.
    """
    if node["type"] not in ("function", "method", "class", "file"):
        return 0

    nid: str = node["id"]
    file_part, _, sym_part = nid.rpartition(":")
    if not file_part:
        file_part = nid
        sym_part = ""

    name_tokens = set(_tokenize(sym_part))
    file_tokens = set(_tokenize(file_part))

    # Compute the best weight matched in name and file separately.
    name_matches = {t: kw_weights[t] for t in name_tokens if t in kw_weights}
    file_matches = {t: kw_weights[t] for t in file_tokens if t in kw_weights}

    best_name_w = max(name_matches.values(), default=0)
    best_file_w = max(file_matches.values(), default=0)

    # ── Hard gate ────────────────────────────────────────────────────────────
    # Require weight >= 2 (exact user word or strong synonym) somewhere.
    if best_name_w < 2 and best_file_w < 2:
        return 0
    # ─────────────────────────────────────────────────────────────────────────

    # Sum weighted scores: name hits count 1.5×, file hits count 1×.
    name_score = sum(int(w * 1.5) for w in name_matches.values())
    file_score = sum(w for w in file_matches.values())
    score = name_score + file_score

    # Path-level boosts / penalties.
    if _is_test_path(file_part):
        score += _PENALTY_TEST_FILE

    file_stem_tokens = set(_tokenize(file_part.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]))
    if file_stem_tokens & _ENTRY_BOOST_FILES:
        score += _SCORE_BOOST_FILE
    elif _is_important_dir(file_part):
        score += _SCORE_IMPORTANT_DIR
    elif file_tokens & _ENTRY_FILE_TOKENS:
        score += _SCORE_FILE_HEURISTIC

    if node["type"] in ("function", "method") and fan_in.get(nid, 0) >= _FAN_IN_HIGH:
        score += _SCORE_FAN_IN

    return score


def _type_preference(ntype: str) -> int:
    """Sort tiebreak: prefer concrete code symbols over files/classes."""
    return {"function": 0, "method": 0, "class": 1, "file": 2}.get(ntype, 3)


def _fallback_entries(
    nodes: list[Node],
    edges: list[Edge],
    max_entries: int,
) -> list[str]:
    """Return highest-degree function/method nodes when nothing matched."""
    degree: dict[str, int] = {}
    for edge in edges:
        degree[edge["from"]] = degree.get(edge["from"], 0) + 1
        degree[edge["to"]] = degree.get(edge["to"], 0) + 1

    code_nodes = [n for n in nodes if n["type"] in ("function", "method")]
    if not code_nodes:
        # Graph without any code — fall back to file nodes.
        code_nodes = [n for n in nodes if n["type"] == "file"]
    code_nodes.sort(key=lambda n: (-degree.get(n["id"], 0), n["id"]))
    return [n["id"] for n in code_nodes[:max_entries]]


def find_entry_points(
    keywords: list[str],
    nodes: list[Node],
    edges: list[Edge] | None = None,
    max_entries: int = _MAX_ENTRIES,
) -> list[str]:
    """Return the top-*max_entries* (default 5) strictly-qualified entry nodes.

    Strict qualification rules
    --------------------------
    A node **must** satisfy the hard gate in :func:`_score_node` to be
    considered: its symbol name **or** file path must contain a primary
    keyword token.  Synonym-only matches from unrelated modules are rejected.
    The final score must also reach ``_ENTRY_MIN_SCORE`` (currently 3).

    Scoring rubric (see :func:`_score_node` for full details)
    ----------------------------------------------------------
    * ``+3`` per exact keyword token hit on the **symbol name**.
    * ``+2`` per exact keyword token hit on the **file path**.
    * ``+2`` per synonym hit on the symbol name.
    * ``+1`` per synonym hit on the file path.
    * ``+3`` strong boost: file is auth.py / security.py / middleware / routes / …
    * ``+2`` if the file lives in an important directory (auth/service/core/api).
    * ``+1`` if the file name resembles a generic entry-point module.
    * ``+2`` if the symbol has high incoming-call count (fan-in ≥ 3).
    * ``-3`` if the path contains a test/mock/fixture segment.

    No-keyword behaviour
    --------------------
    When *keywords* is empty the query gives us nothing to anchor on; the
    fallback returns the highest-degree functions so traversal can still start
    somewhere meaningful.

    When *keywords* are present but **no node passes the hard gate**, an empty
    list is returned rather than falling back to random high-degree nodes.
    The caller (``run_query``) must handle this gracefully.
    """
    edges_list = edges or []
    fan_in = _compute_fan_in(edges_list)
    kw_weights = _weighted_keywords(keywords)

    if not kw_weights:
        return _fallback_entries(nodes, edges_list, max_entries)

    # The hard module gate is only useful when the query contains auth/security
    # domain keywords — it keeps "login" queries from matching unrelated files.
    # For general code queries ("compress_code", "parse_query") the gate would
    # eliminate every node, so we bypass it entirely in that case.
    _strong_domain_kws = {
        k for k, w in kw_weights.items()
        if w >= 2 and k in _ENTRY_MODULE_KEYWORDS
    }
    apply_module_gate = bool(_strong_domain_kws)

    scored: list[tuple[int, int, str]] = []
    for node in nodes:
        nid = node["id"]
        file_part, _, _ = nid.rpartition(":")
        if not file_part:
            file_part = nid

        # ── Hard module pre-filter (auth-domain queries only) ───────────────
        if apply_module_gate and not _is_relevant_module(file_part, kw_weights):
            continue
        # ──────────────────────────────────────────────────────────────────

        s = _score_node(node, kw_weights, fan_in)
        if s >= _ENTRY_MIN_SCORE:
            scored.append((-s, _type_preference(node["type"]), node["id"]))

    if not scored:
        return []

    scored.sort()
    picked: list[str] = []
    seen: set[str] = set()
    for _, _, nid in scored:
        if nid in seen:
            continue
        seen.add(nid)
        picked.append(nid)
        if len(picked) >= max_entries:
            break
    return picked


# ---------------------------------------------------------------------------
# Step 3: Graph traversal
# ---------------------------------------------------------------------------

def _build_adjacency(edges: list[Edge]) -> tuple[dict[str, list[tuple[str, str]]], dict[str, list[tuple[str, str]]]]:
    """Build forward and reverse adjacency maps from the edge list.

    Returns ``(outgoing, incoming)`` where each value is a list of
    ``(neighbour_id, edge_type)`` tuples.
    """
    outgoing: dict[str, list[tuple[str, str]]] = {}
    incoming: dict[str, list[tuple[str, str]]] = {}
    for edge in edges:
        src, dst, etype = edge["from"], edge["to"], edge["type"]
        outgoing.setdefault(src, []).append((dst, etype))
        incoming.setdefault(dst, []).append((src, etype))
    return outgoing, incoming


def traverse_graph(
    entry_ids: list[str],
    nodes: list[Node],
    edges: list[Edge],
    max_depth: int = _MAX_DEPTH,
) -> dict[str, int]:
    """BFS from each entry node following calls, imports, and reverse-calls.

    Returns a mapping of ``node_id → minimum depth`` reached.  Depth 0 means
    the node is an entry point; depth 1 means one hop away, etc.
    """
    node_ids: set[str] = {n["id"] for n in nodes}
    outgoing, incoming = _build_adjacency(edges)

    # Edge types to follow in each direction.
    _follow_out: frozenset[str] = frozenset({"calls", "imports", "contains", "has_method"})
    _follow_in: frozenset[str] = frozenset({"calls"})   # reverse-calls for debugging

    visited: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()

    for eid in entry_ids:
        if eid in node_ids:
            queue.append((eid, 0))
            visited[eid] = 0

    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue

        # Forward edges
        for neighbour, etype in outgoing.get(current, []):
            if etype in _follow_out and neighbour not in visited:
                visited[neighbour] = depth + 1
                queue.append((neighbour, depth + 1))

        # Reverse call edges (who calls *current*)
        for neighbour, etype in incoming.get(current, []):
            if etype in _follow_in and neighbour not in visited:
                visited[neighbour] = depth + 1
                queue.append((neighbour, depth + 1))

    return visited


# ---------------------------------------------------------------------------
# Step 4: Relevance ranking + filtering
# ---------------------------------------------------------------------------

def _file_of(node_id: str) -> str:
    """Return the file portion of a node ID (everything before the last ``:``)."""
    file_part, sep, _ = node_id.rpartition(":")
    return file_part if sep else node_id


def _module_scores(
    visited: dict[str, int],
    node_map: dict[str, Node],
    kw_weights: dict[str, int],
) -> dict[str, int]:
    """Compute a per-file relevance score for every module touched by traversal.

    Rubric
    ------
    * ``+3``  file path contains a module-level keyword (auth/login/security/…).
    * ``+2``  file path contains any query keyword (weight >= 2).
    * ``+2``  the file contains ≥2 visited nodes whose symbol names hit a
              keyword with weight >= 2.
    """
    strong_kws = {k for k, w in kw_weights.items() if w >= 2}

    relevant_fn_count: dict[str, int] = {}
    for nid in visited:
        node = node_map.get(nid)
        if node is None or node.get("type") not in ("function", "method"):
            continue
        file = _file_of(nid)
        _, _, sym = nid.rpartition(":")
        name_tokens = set(_tokenize(sym))
        if name_tokens & strong_kws:
            relevant_fn_count[file] = relevant_fn_count.get(file, 0) + 1

    all_files: set[str] = {_file_of(nid) for nid in visited}

    scores: dict[str, int] = {}
    for file in all_files:
        s = 0
        file_tokens = set(_tokenize(file))
        path_segments = {seg.lower() for seg in re.split(r"[/\\]", file) if seg}

        if path_segments & _MODULE_KEYWORDS:
            s += 3
        if file_tokens & strong_kws:
            s += 2
        if relevant_fn_count.get(file, 0) >= 2:
            s += 2
        scores[file] = s
    return scores


def _keyword_relevance_score(
    node: Node,
    kw_weights: dict[str, int],
    high_score_neighbours: set[str],
    module_boost: int = 0,
) -> int:
    """Weighted keyword relevance score for traversal filtering.

    Uses the same ``kw_weights`` dict as ``_score_node`` so that the
    relative importance of exact vs. expanded keywords is consistent
    across both entry-point selection and traversal filtering.

    Rubric
    ------
    * name token hit:  ``weight`` (from kw_weights).
    * file path hit:   ``weight // 2`` (path is weaker signal than name).
    * code body hit:   ``weight // 2``, capped at 2 total.
    * neighbour of high-scoring node: ``+1``.
    * module boost: ``+2`` (relevant module) / ``-3`` (unrelated module).
    * no match at all: ``-2``.
    """
    nid: str = node["id"]
    file_part, _, sym_part = nid.rpartition(":")
    if not file_part:
        file_part = nid
        sym_part = ""

    name_tokens = set(_tokenize(sym_part))
    file_tokens = set(_tokenize(file_part))
    code_text = (node.get("code") or "").lower()

    name_score = sum(kw_weights[t] for t in name_tokens if t in kw_weights)
    file_score = sum(kw_weights[t] // 2 for t in file_tokens if t in kw_weights)
    code_score = min(
        sum(w // 2 for kw, w in kw_weights.items() if kw in code_text), 2
    )

    has_any_match = bool(name_score or file_score or code_score)

    score = name_score + file_score + code_score
    score += module_boost

    if nid in high_score_neighbours:
        score += 1

    if not has_any_match:
        score -= 2

    return score


def rank_nodes(
    visited: dict[str, int],
    entry_ids: list[str],
    all_nodes: list[Node],
    keywords: list[str] | None = None,
    max_nodes: int = _MAX_NODES,
    threshold: int = _RELEVANCE_THRESHOLD,
) -> list[Node]:
    """Return up to *max_nodes* candidate nodes ordered by relevance.

    Three-stage filtering
    ---------------------
    1. **Module-level scoring** — every file touched by traversal gets a
       module score (+3 path keyword, +2 primary keyword, +2 multi-function).
       High-scoring modules boost each of their nodes by +2; low-scoring
       (unrelated) modules penalise each node by -3.

    2. **Keyword-relevance gate** — using the module-adjusted score:
       * Entry points: always kept.
       * Same-file nodes (any depth): kept if score >= threshold.
       * Cross-module 1-hop nodes: kept if score >= threshold.
       * Cross-module depth-2+ nodes: **hard-dropped** (max 1 cross-module hop).
       * Any node with score < threshold AND not in an entry-point file: dropped.

    3. **Structural priority sort** — survivors are ordered by
       (depth, node type, -importance_score) so the pruner receives a
       well-ranked shortlist and can focus on shape-based logic.

    Keyword-relevance rubric (see :func:`_keyword_relevance_score`)
    ---------------------------------------------------------------
    * ``+3``  symbol name contains keyword
    * ``+2``  file path contains keyword
    * ``+2``  docstring / code body contains keyword
    * ``+1``  synonym hit
    * ``+1``  adjacent to a high-relevance node
    * ``+2``  node lives in a high-relevance module
    * ``-3``  node lives in an unrelated module
    * ``-2``  no keyword match at all
    """
    entry_set = set(entry_ids)
    node_map: dict[str, Node] = {n["id"]: n for n in all_nodes}

    kws: list[str] = keywords or []
    kw_weights: dict[str, int] = _weighted_keywords(kws)

    # Files that contain at least one entry point — used for the hard filter.
    entry_files: set[str] = {_file_of(eid) for eid in entry_ids}

    # ------------------------------------------------------------------
    # Stage 1: module-level scoring.
    # ------------------------------------------------------------------
    mod_scores: dict[str, int] = _module_scores(visited, node_map, kw_weights)

    def _module_boost(node_id: str) -> int:
        """Per-node module adjustment.

        +2  node is in a high-relevance module (module_score >= _MODULE_HIGH_SCORE).
        +2  node is in an auth cluster (file has >= 2 high-scoring neighbours).
        -5  node is in a completely unrelated module (module_score == 0).
            The -5 penalty exceeds the maximum single-token name score (weight
            3 * 1.5 = 4 in _score_node; weight 3 in _keyword_relevance_score),
            making it impossible for a lone keyword hit to survive unpenalised.
        """
        if not kw_weights:
            return 0
        ms = mod_scores.get(_file_of(node_id), 0)
        boost = 0
        if ms >= _MODULE_HIGH_SCORE:
            boost += 2
        elif ms == 0:
            boost -= 5   # stronger than any single-token name hit
        if _file_of(node_id) in cluster_files:
            boost += 2
        return boost

    HIGH_SCORE_CUTOFF = 4  # must be defined before cluster detection uses it

    # Stage 1b: auth-cluster detection.
    # A file qualifies as a "cluster" when >= 2 of its visited nodes score
    # above HIGH_SCORE_CUTOFF on raw keyword signal alone (no module boost yet).
    # All nodes in a cluster file receive an additional +2 in _module_boost,
    # lifting the cohesive auth surface above the noise floor.
    _pre_cluster_scores: dict[str, int] = {
        nid: _keyword_relevance_score(node_map[nid], kw_weights, set(), module_boost=0)
        for nid in visited
        if nid in node_map
    }
    _file_high_count: dict[str, int] = {}
    for nid, s in _pre_cluster_scores.items():
        if s >= HIGH_SCORE_CUTOFF:
            f = _file_of(nid)
            _file_high_count[f] = _file_high_count.get(f, 0) + 1

    cluster_files: set[str] = {
        f for f, cnt in _file_high_count.items() if cnt >= 2
    }

    # Stage 2a: raw keyword scores with module boost + cluster boost applied.
    raw_scores: dict[str, int] = {}
    for nid in visited:
        node = node_map.get(nid)
        if node is None:
            continue
        raw_scores[nid] = _keyword_relevance_score(
            node, kw_weights, set(),
            module_boost=_module_boost(nid),
        )

    high_score_ids: set[str] = {
        nid for nid, s in raw_scores.items() if s >= HIGH_SCORE_CUTOFF
    }

    # ------------------------------------------------------------------
    # Stage 2b: priority sort + gate.
    # ------------------------------------------------------------------
    def _priority(node_id: str) -> int:
        """Lower value = higher priority in the final sort."""
        if node_id in entry_set:
            return 0
        depth = visited.get(node_id, 99)
        ntype = node_map.get(node_id, {}).get("type", "")
        is_code = ntype in ("function", "method")
        same_file = _file_of(node_id) in entry_files
        # Same-file nodes always rank ahead of cross-module nodes at same depth.
        if depth == 1:
            return 1 if (is_code and same_file) else (2 if same_file else 3)
        if depth == 2:
            return 4 if (is_code and same_file) else (5 if same_file else 6)
        return 10

    def _sort_key(node_id: str) -> tuple[int, int]:
        node = node_map.get(node_id, {})
        return (_priority(node_id), -importance_score(node))

    kept_ids: list[str] = []
    for nid in sorted(visited.keys(), key=_sort_key):
        node = node_map.get(nid)
        if node is None:
            continue

        depth = visited.get(nid, 99)
        in_entry_file = _file_of(nid) in entry_files
        is_entry = nid in entry_set

        if kw_weights:
            if not in_entry_file and depth > 1:
                continue

            if not is_entry:
                rel_score = _keyword_relevance_score(
                    node, kw_weights, high_score_ids,
                    module_boost=_module_boost(nid),
                )
                if rel_score < threshold and not in_entry_file:
                    continue
                if rel_score < 0:
                    continue

        kept_ids.append(nid)
        if len(kept_ids) >= max_nodes:
            break

    return [node_map[nid] for nid in kept_ids if nid in node_map]


# ---------------------------------------------------------------------------
# Step 5: Context assembly
# ---------------------------------------------------------------------------

def build_context(
    ranked_nodes: list[Node],
    entry_ids: list[str],
    keywords: list[str] | None = None,
    compress: bool = True,
    inline_hints: dict[str, list[str]] | None = None,
) -> str:
    """Assemble code snippets ordered by entry → callees → callers.

    Each function/method snippet is passed through :func:`compress_code` so
    the emitted context contains only debug-relevant lines plus a
    ``# calls:`` dependency hint.  When *inline_hints* is provided, each
    entry-node block gets its helper hints appended (one per line).  File
    and class nodes without code contribute a comment stub only.
    """
    entry_set = set(entry_ids)
    kws = keywords or []
    hints = inline_hints or {}

    # Split into ordered buckets.
    entries: list[Node] = []
    rest: list[Node] = []
    for node in ranked_nodes:
        (entries if node["id"] in entry_set else rest).append(node)

    ordered = entries + rest
    seen_ids: set[str] = set()
    parts: list[str] = []

    for node in ordered:
        nid = node["id"]
        if nid in seen_ids:
            continue
        seen_ids.add(nid)

        code: str = node.get("code", "").strip()
        ntype: str = node.get("type", "")

        if code:
            if compress and ntype in ("function", "method"):
                code = compress_code(code, kws).strip()
            if not code:
                continue
            header = f"# [{ntype}] {nid}"
            block = f"{header}\n{code}"
            if nid in hints:
                block += "\n" + "\n".join(hints[nid])
            parts.append(block)
        else:
            parts.append(f"# [{ntype}] {nid}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Step 6: Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token count: characters / 4 (GPT-style approximation)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_query(query: str, graph: Graph, compress: bool = True) -> QueryResult:
    """Full retrieval pipeline: query string + graph → QueryResult.

    Pipeline: parse → **expand** → entry-points → BFS traversal → rank →
    **prune** (classify, dedup, inline) → compress → assemble.

    Keyword expansion
    -----------------
    After parsing, raw keywords (e.g. ``["login"]``) are expanded using the
    synonym table before being passed to any downstream stage.  This means
    the entry-point hard gate, module scoring, and traversal filtering all
    operate on the *full* expanded set::

        "login"  →  ["login", "auth", "token", "jwt", "session", "authenticate", …]

    The ``QueryResult.keywords`` field still holds the **original** user
    keywords so the caller can display what the user actually typed.
    ``token_estimate_raw`` measures the pre-compression/pre-prune baseline.
    """
    parsed = parse_query(query)
    intent: str = parsed["intent"]
    raw_keywords: list[str] = parsed["keywords"]   # what the user literally typed

    # ------------------------------------------------------------------ #
    # Expand keywords BEFORE any downstream stage runs.                   #
    # This is the integration point that makes synonym matching work.     #
    # Without this, "login" never matches auth.py:create_access_token.   #
    # ------------------------------------------------------------------ #
    # Build weighted keyword map and derive the operative keyword list from it.
    # Tokens with weight 0 are excluded; higher-weight tokens come first.
    kw_weights_global = _weighted_keywords(raw_keywords)
    keywords: list[str] = sorted(
        kw_weights_global, key=lambda k: -kw_weights_global[k]
    )

    nodes: list[Node] = graph["nodes"]
    edges: list[Edge] = graph["edges"]

    entry_ids = find_entry_points(keywords, nodes, edges)

    # find_entry_points returns [] when keywords were provided but no node
    # satisfied the hard gate (name OR file must contain a primary keyword).
    # Proceeding with an empty entry list would produce meaningless traversal;
    # return an empty result instead so the caller can surface a clear signal.
    if not entry_ids:
        return QueryResult(
            intent=intent,
            keywords=keywords,
            entry_points=[],
            nodes_selected=[],
            categories={},
            inline_hints={},
            token_estimate=1,
            token_estimate_raw=1,
            context="",
        )

    visited = traverse_graph(entry_ids, nodes, edges)
    ranked = rank_nodes(visited, entry_ids, nodes, keywords=keywords)

    # Narrow candidates into a minimal, categorised set.
    result: PruneResult = prune(ranked, entry_ids, visited)

    context_text = build_context(
        result.kept,
        entry_ids,
        keywords,
        compress=compress,
        inline_hints=result.inline_hints,
    )
    # Baseline: untouched candidates, no compression, no inlining.
    raw_text = build_context(ranked, entry_ids, keywords, compress=False)

    return QueryResult(
        intent=intent,
        keywords=raw_keywords,    # show user's original words, not the expanded set
        entry_points=entry_ids,
        nodes_selected=[n["id"] for n in result.kept],
        categories=dict(result.categories),
        inline_hints=dict(result.inline_hints),
        token_estimate=estimate_tokens(context_text),
        token_estimate_raw=estimate_tokens(raw_text),
        context=context_text,
    )
