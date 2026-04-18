"""CLI entry points for context_engine."""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import typer

from .graph_builder import build_graph
from .intent import format_intent_output
from .parser import parse_file
from .retrieval import run_query

app = typer.Typer(
    name="context-engine",
    help="Codebase ingestion and call-graph builder.",
    add_completion=False,
)

IGNORE_DIRS: frozenset[str] = frozenset(
    {"venv", "node_modules", ".git", "__pycache__", ".cecl"}
)

OUTPUT_DIR = ".cecl"
OUTPUT_FILE = "graph.json"


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(levelname)s  %(message)s",
        level=level,
        stream=sys.stderr,
    )


def _collect_py_files(root: Path) -> list[Path]:
    """Walk root recursively, skipping ignored directories."""
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


@app.command()
def index(
    directory: Path = typer.Argument(
        default=Path("."),
        help="Root directory to scan (default: current directory).",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Scan Python files and build a call graph saved to .cecl/graph.json."""
    _configure_logging(verbose)
    logger = logging.getLogger(__name__)

    logger.info("Scanning %s for Python files…", directory)
    py_files = _collect_py_files(directory)
    if not py_files:
        typer.echo("No Python files found.", err=True)
        raise typer.Exit(code=1)

    logger.info("Found %d files — parsing…", len(py_files))

    results = []
    errors = 0
    for path in py_files:
        result = parse_file(path)
        if result is None:
            errors += 1
        else:
            results.append(result)

    logger.info("Parsed %d files (%d skipped due to errors).", len(results), errors)

    graph = build_graph(results)

    out_dir = directory / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUTPUT_FILE
    out_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")

    node_counts = Counter(n["type"] for n in graph["nodes"])
    edge_counts = Counter(e["type"] for e in graph["edges"])
    typer.echo(f"Graph saved -> {out_path}")
    typer.echo(
        f"  nodes : {len(graph['nodes'])}  "
        + "  ".join(f"{t}={c}" for t, c in sorted(node_counts.items()))
    )
    typer.echo(
        f"  edges : {len(graph['edges'])}  "
        + "  ".join(f"{t}={c}" for t, c in sorted(edge_counts.items()))
    )


@app.command()
def query(
    query_str: str = typer.Argument(..., metavar="QUERY", help="Natural-language query, e.g. 'fix login bug'."),
    graph_path: Path = typer.Option(
        Path(".cecl/graph.json"),
        "--graph", "-g",
        help="Path to graph.json produced by `index`.",
    ),
    raw: bool = typer.Option(False, "--raw", help="Print full JSON result instead of formatted output."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Query the code graph and return minimal relevant context."""
    _configure_logging(verbose)

    if not graph_path.exists():
        typer.echo(f"Graph not found at {graph_path}. Run `context-engine index` first.", err=True)
        raise typer.Exit(code=1)

    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        typer.echo(f"Failed to parse {graph_path}: {exc}", err=True)
        raise typer.Exit(code=1)

    result = run_query(query_str, graph)

    if raw:
        typer.echo(json.dumps(result, indent=2))
        return

    typer.echo(format_intent_output(result, graph))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    app()


if __name__ == "__main__":
    main()
