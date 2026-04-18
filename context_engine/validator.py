"""Pre-apply validation: syntax correctness and duplicate symbol detection."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


def _top_level_names(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def validate_syntax(code: str, filename: str = "<generated>") -> ValidationResult:
    """Return ValidationResult with ok=False if code has a SyntaxError."""
    if not code.strip():
        return ValidationResult(ok=True)
    try:
        ast.parse(code)
        return ValidationResult(ok=True)
    except SyntaxError as exc:
        return ValidationResult(
            ok=False,
            errors=[f"SyntaxError line {exc.lineno}: {exc.msg}"],
        )


def validate_no_duplicates(existing_code: str, additions: str) -> ValidationResult:
    """Ensure additions do not redefine top-level names already in existing_code."""
    if not existing_code.strip() or not additions.strip():
        return ValidationResult(ok=True)
    try:
        existing_names = _top_level_names(ast.parse(existing_code))
        new_names = _top_level_names(ast.parse(additions))
    except SyntaxError:
        return ValidationResult(ok=True)  # syntax validator handles this

    duplicates = existing_names & new_names
    if duplicates:
        return ValidationResult(
            ok=False,
            errors=[f"Would overwrite existing definitions: {', '.join(sorted(duplicates))}"],
        )
    return ValidationResult(ok=True)


def validate_patch(original: str, patched: str, filename: str = "<patched>") -> ValidationResult:
    """Run all validations on the patched file content."""
    return validate_syntax(patched, filename)
