"""Pipeline orchestrator for the `apply` command."""

from __future__ import annotations

from pathlib import Path

from .patcher import _extract_create_content, apply_file_diff, backup_file, generate_diff, parse_diff
from .planner import plan
from .ranker import format_output, rank_nodes
from .retrieval import run_query
from .validator import validate_no_duplicates, validate_patch

_COMPRESS_SYSTEM = """\
You are a codebase context optimizer, not a code generator.

Your job is to extract the MINIMUM necessary context from a codebase to solve the given task, while minimizing token usage.

## INSTRUCTIONS

1. Detect intent:
   * DEBUG -> include function + callers + error paths
   * GENERATE -> include helpers + dependencies + patterns
   * EXPLAIN -> include high-level flow only

2. Select context:
   * Max 5-8 nodes total
   * Prefer core functions over helpers
   * Follow call graph (1 hop max)

3. Compress aggressively:
   * DO NOT include full code unless absolutely necessary
   * Convert functions into: function_name(args) -> purpose
   * Remove implementation details
   * Keep only signatures + role

4. Highlight:
   * Core flow (how things connect)
   * Missing pieces (important!)
   * Likely issues (for debugging)

5. Output format STRICTLY:

=== TASK === <short rewritten task>

=== CORE FLOW ===
A -> B -> C

=== KEY FUNCTIONS ===
func1(args) -> purpose
func2(args) -> purpose

=== MISSING === <missing dependencies or functions>

=== POSSIBLE ISSUES === <only if DEBUG>

=== CONSTRAINTS ===
<framework / async / auth etc>

=== OUTPUT INSTRUCTION ===
Return only code. No explanation.

## RULES
* Be concise. Every token matters.
* No unnecessary text.
* No explanations outside defined sections.
* Do NOT generate final code solution.
"""


def _deterministic_context(query: str, result: dict) -> str:
    """Rank and format nodes using the deterministic ranker (no LLM)."""
    nodes: list[dict] = result.get("nodes", [])
    if not nodes:
        return ""
    return format_output(query, nodes)


def compress_context(query: str, result: dict) -> str:
    """
    Compress retrieval output into minimal LLM context.
    Uses deterministic ranker first; optionally refines with Haiku.
    Falls back gracefully if anthropic is unavailable.
    """
    deterministic = _deterministic_context(query, result)
    if not deterministic.strip():
        return ""

    try:
        import anthropic
    except ImportError:
        return deterministic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=_COMPRESS_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"User query: {query}\n\nCode graph + functions:\n{deterministic}",
        }],
    )
    return message.content[0].text.strip()


def _extract_fn_lists(result: dict) -> tuple[list[str], list[str]]:
    """Return (known_functions, missing_functions) from a query result."""
    kept: list[dict] = result.get("nodes", [])
    known = [n["name"] for n in kept if n.get("name")]
    missing: list[str] = result.get("missing_deps", [])
    return known, missing


def run_apply(
    query: str,
    graph: dict,
    project_root: Path,
    dry_run: bool = False,
    yes: bool = False,
) -> int:
    """
    Full apply pipeline: query → plan → diff → validate → apply.
    Returns exit code (0 = success, 1 = error).
    """
    from typer import echo, confirm

    backup_dir = project_root / ".cecl" / "backups"

    # Step 1: Context extraction + compression
    echo("Step 1/5  Extracting and compressing context...")
    result = run_query(query, graph)
    context = compress_context(query, result)
    known_fns, missing_fns = _extract_fn_lists(result)

    if not context.strip():
        echo("  No relevant context found in graph - proceeding with empty context.")

    # Step 2: Planning
    echo("Step 2/5  Planning changes (Haiku)...")
    try:
        steps = plan(query, context)
    except RuntimeError as exc:
        echo(f"  Planning failed: {exc}", err=True)
        return 1

    echo(f"  {len(steps)} step(s) planned:")
    for s in steps:
        echo(f"    [{s['action'].upper()}] {s['file']} - {s['reason']}")

    # Step 3: Diff generation
    echo("Step 3/5  Generating diffs (Sonnet)...")
    try:
        diff_text = generate_diff(
            query=query,
            plan=steps,
            compressed_context=context,
            known_functions=known_fns,
            missing_functions=missing_fns,
            project_root=project_root,
        )
    except RuntimeError as exc:
        echo(f"  Diff generation failed: {exc}", err=True)
        return 1

    diffs = parse_diff(diff_text)
    if not diffs:
        echo("  No diffs produced by LLM.", err=True)
        return 1

    echo(f"  {len(diffs)} file diff(s) parsed.")

    # Step 4: Validation
    echo("Step 4/5  Validating diffs...")
    valid = True
    patch_ops: list[tuple] = []  # (FileDiff, Path, str, str)

    for diff in diffs:
        original = ""
        patched = ""
        target = project_root / diff.relative_path
        try:
            target, original, patched = apply_file_diff(diff, project_root)
        except FileNotFoundError as exc:
            if diff.is_new_file:
                target = project_root / diff.relative_path
                original = ""
                patched = _extract_create_content(diff.hunks)
            else:
                echo(f"  ERROR: {exc}", err=True)
                valid = False
                continue

        vr = validate_patch(original, patched, diff.relative_path)
        if not vr.ok:
            echo(f"  SYNTAX ERROR in {diff.relative_path}:", err=True)
            for err in vr.errors:
                echo(f"    {err}", err=True)
            valid = False
            continue

        if original:
            vdup = validate_no_duplicates(original, patched[len(original):] if len(patched) > len(original) else "")
            if not vdup.ok:
                echo(f"  DUPLICATE SYMBOL in {diff.relative_path}:", err=True)
                for err in vdup.errors:
                    echo(f"    {err}", err=True)
                valid = False
                continue

        echo(f"  OK  {diff.relative_path}")
        patch_ops.append((diff, target, original, patched))

    if not valid:
        echo("Validation failed - no files written.", err=True)
        return 1

    # Dry-run: show diffs only
    if dry_run:
        echo("\n-- DRY RUN: diff preview --")
        echo(diff_text)
        echo("-- No files written (--dry-run) --")
        return 0

    # Step 5: Apply
    if not yes:
        files_list = ", ".join(str(t) for _, t, _, _ in patch_ops)
        confirmed = confirm(f"\nApply changes to: {files_list}?")
        if not confirmed:
            echo("Aborted.")
            return 0

    echo("Step 5/5  Applying changes...")
    for _diff, target, original, patched in patch_ops:
        if original:
            bak = backup_file(target, backup_dir)
            echo(f"  Backed up {target.name} -> {bak.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(patched, encoding="utf-8")
        action = "Created" if not original else "Updated"
        echo(f"  {action} {target}")

    echo(f"\nDone. {len(patch_ops)} file(s) modified. Backups in {backup_dir}")
    return 0
