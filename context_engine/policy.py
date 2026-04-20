"""Escape policy: confidence scoring, path normalization, gate logic, session state."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

_NEED_PATTERN = re.compile(
    r"NEED:\s*(.+?)\s*REASON:\s*(.+)", re.IGNORECASE | re.DOTALL
)
_SESSION_DIR = Path(".llm_diet")
_LOG_DIR = _SESSION_DIR / "logs"
_SESSION_FILE = _SESSION_DIR / "session.json"
_MAX_ESCAPE_ATTEMPTS = 2
_GATED_TOOLS = frozenset({"Read", "Edit", "Write", "MultiEdit"})
_ALLOWED_TOOLS = frozenset({"Grep", "Glob", "Bash", "LS"})


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def normalize_path(p: str) -> str:
    return p.replace("\\", "/").lower().strip()


def basename(p: str) -> str:
    return normalize_path(p).split("/")[-1]


def paths_match(a: str, b: str) -> bool:
    na, nb = normalize_path(a), normalize_path(b)
    return na == nb or basename(na) == basename(nb)


# ---------------------------------------------------------------------------
# NEED / REASON extraction
# ---------------------------------------------------------------------------

def extract_need(text: str) -> tuple[str, str] | None:
    """Return (normalized_filename, reason) or None."""
    m = _NEED_PATTERN.search(text)
    if not m:
        return None
    filename = normalize_path(m.group(1).strip().split()[0])
    reason = m.group(2).strip()
    return filename, reason


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def compute_confidence(ranked_nodes: list[dict]) -> tuple[float, str]:
    """
    Return (confidence 0.0–1.0, mode).
    Inputs: coverage (endpoint/auth/helper present), file diversity, score gap.
    """
    if not ranked_nodes:
        return 0.0, "open"

    from .ranker import _is_endpoint, _is_auth_fn, _is_helper

    has_endpoint = any(_is_endpoint(n) for n in ranked_nodes)
    has_auth = any(_is_auth_fn(n) for n in ranked_nodes)
    has_helper = any(_is_helper(n) for n in ranked_nodes)
    files = {n.get("file", "") for n in ranked_nodes}
    diverse = len(files) > 1

    score = 0.0
    score += 0.35 if has_endpoint else 0.0
    score += 0.30 if has_auth else 0.0
    score += 0.20 if has_helper else 0.0
    score += 0.15 if diverse else 0.0

    if score > 0.8:
        mode = "restricted"
    elif score > 0.5:
        mode = "guided"
    else:
        mode = "open"

    return round(score, 3), mode


# ---------------------------------------------------------------------------
# Neighbor expansion
# ---------------------------------------------------------------------------

def get_related_files(filepath: str, max_extra: int = 3) -> set[str]:
    """Return files in the same directory (up to max_extra)."""
    p = Path(filepath)
    if not p.parent.exists():
        return set()
    siblings = [
        str(f)
        for f in p.parent.iterdir()
        if f.suffix == ".py" and f != p and not f.name.startswith("__")
    ]
    return set(siblings[:max_extra])


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class PolicySession:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    query: str = ""
    mode: str = "open"
    confidence: float = 0.0
    allowed_files: list[str] = field(default_factory=list)
    denied_reads: list[str] = field(default_factory=list)
    escape_attempts: dict[str, int] = field(default_factory=dict)
    escapes: list[dict] = field(default_factory=list)

    # ---- persistence -------------------------------------------------------

    @classmethod
    def load(cls) -> "PolicySession":
        if _SESSION_FILE.exists():
            try:
                data = json.loads(_SESSION_FILE.read_text(encoding="utf-8"))
                return cls(**data)
            except Exception:
                pass
        return cls()

    def save(self) -> None:
        _SESSION_DIR.mkdir(exist_ok=True)
        _SESSION_FILE.write_text(
            json.dumps(self.__dict__, indent=2), encoding="utf-8"
        )

    def save_log(self) -> None:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = _LOG_DIR / f"{self.session_id}.json"
        log_path.write_text(
            json.dumps(self.__dict__, indent=2), encoding="utf-8"
        )

    # ---- helpers -----------------------------------------------------------

    def is_allowed(self, filepath: str) -> bool:
        return any(paths_match(filepath, f) for f in self.allowed_files)

    def allow_file(self, filepath: str) -> None:
        norm = normalize_path(filepath)
        if not self.is_allowed(norm):
            self.allowed_files.append(norm)

    def allow_with_neighbors(self, filepath: str) -> None:
        self.allow_file(filepath)
        for neighbor in get_related_files(filepath):
            self.allow_file(neighbor)

    def record_deny(self, filepath: str) -> None:
        norm = normalize_path(filepath)
        self.denied_reads.append(norm)
        self.escape_attempts[norm] = self.escape_attempts.get(norm, 0) + 1

    def record_escape(self, filepath: str, reason: str) -> None:
        norm = normalize_path(filepath)
        attempt = self.escape_attempts.get(norm, 0)
        self.escapes.append({"file": norm, "reason": reason, "attempt": attempt})


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

def gate(session: PolicySession, filepath: str, last_message: str) -> tuple[str, str]:
    """
    Apply two-phase gate.
    Returns ("allow", "") or ("block", reason_message).
    """
    norm = normalize_path(filepath)

    if session.mode == "open":
        return "allow", ""

    if session.is_allowed(filepath):
        return "allow", ""

    attempts = session.escape_attempts.get(norm, 0)

    if attempts >= _MAX_ESCAPE_ATTEMPTS:
        return "block", f"Max escape attempts reached for {basename(norm)}."

    parsed = extract_need(last_message)
    if parsed and paths_match(parsed[0], filepath):
        _, reason = parsed
        session.record_escape(filepath, reason)
        session.allow_with_neighbors(filepath)
        session.save()
        return "allow", ""

    if attempts == 0:
        session.record_deny(filepath)
        session.save()
        return (
            "block",
            f"File outside scope: {basename(norm)}. "
            f"If required, respond with:\n"
            f"NEED: {basename(norm)}\n"
            f"REASON: <one sentence explaining why>",
        )

    session.record_deny(filepath)
    session.save()
    return "block", f"Invalid NEED/REASON for {basename(norm)}. Access denied."
