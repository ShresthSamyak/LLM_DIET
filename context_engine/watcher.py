"""Filesystem watcher — rebuilds .cecl/graph.json on every relevant file change.

Usage (via CLI):
    context-engine watch .
    context-engine watch /path/to/repo
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .graph_builder import build_graph
from .js_parser import JS_EXTENSIONS, parse_js_file
from .parser import parse_file

logger = logging.getLogger(__name__)

_WATCHED_EXTENSIONS = frozenset({".py"}) | JS_EXTENSIONS
_IGNORE_DIRS = frozenset({"venv", ".venv", "__pycache__", ".git", "node_modules", ".cecl"})
_OUTPUT_DIR = ".cecl"
_OUTPUT_FILE = "graph.json"
_DEBOUNCE_SECONDS = 2.0


# ---------------------------------------------------------------------------
# Index helper (mirrors cli.py index logic without Typer)
# ---------------------------------------------------------------------------

def _collect_source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.suffix not in _WATCHED_EXTENSIONS:
            continue
        if any(part in _IGNORE_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def rebuild_graph(root: Path) -> tuple[int, int, float]:
    """Parse all source files under *root* and write graph.json.

    Returns (node_count, edge_count, elapsed_ms).
    """
    t0 = time.monotonic()
    files = _collect_source_files(root)

    results = []
    for path in files:
        if path.suffix == ".py":
            r = parse_file(path)
        else:
            r = parse_js_file(path)
        if r is not None:
            results.append(r)

    graph = build_graph(results)

    import json
    out_dir = root / _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / _OUTPUT_FILE).write_text(
        json.dumps(graph, indent=2), encoding="utf-8"
    )

    elapsed_ms = (time.monotonic() - t0) * 1000
    return len(graph["nodes"]), len(graph["edges"]), elapsed_ms


# ---------------------------------------------------------------------------
# Watchdog event handler with debounce
# ---------------------------------------------------------------------------

class _SourceChangeHandler(FileSystemEventHandler):
    def __init__(self, root: Path) -> None:
        super().__init__()
        self._root = root
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def _is_relevant(self, path_str: str) -> bool:
        p = Path(path_str)
        if p.suffix not in _WATCHED_EXTENSIONS:
            return False
        return not any(part in _IGNORE_DIRS for part in p.parts)

    def _schedule_rebuild(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(_DEBOUNCE_SECONDS, self._do_rebuild)
            self._timer.daemon = True
            self._timer.start()

    def _do_rebuild(self) -> None:
        try:
            nodes, edges, ms = rebuild_graph(self._root)
            print(f"Graph updated: {nodes} nodes, {edges} edges ({ms:.0f}ms)", flush=True)
        except Exception as exc:
            print(f"Graph rebuild failed: {exc}", flush=True)
            logger.exception("Rebuild error")

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_relevant(event.src_path):
            self._schedule_rebuild()

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_relevant(event.src_path):
            self._schedule_rebuild()

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_relevant(event.src_path):
            self._schedule_rebuild()

    def on_moved(self, event: FileSystemEvent) -> None:
        src_ok = self._is_relevant(event.src_path)
        dst_ok = self._is_relevant(event.dest_path)  # type: ignore[attr-defined]
        if not event.is_directory and (src_ok or dst_ok):
            self._schedule_rebuild()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def watch(root: Path) -> None:
    """Block forever, rebuilding the graph whenever source files change."""
    root = root.resolve()
    print(f"Watching {root}", flush=True)
    print(f"Extensions : {', '.join(sorted(_WATCHED_EXTENSIONS))}", flush=True)
    print(f"Debounce   : {_DEBOUNCE_SECONDS}s", flush=True)
    print("Press Ctrl-C to stop.\n", flush=True)

    # Initial build so the graph is fresh when the watcher starts.
    try:
        nodes, edges, ms = rebuild_graph(root)
        print(f"Initial build: {nodes} nodes, {edges} edges ({ms:.0f}ms)", flush=True)
    except Exception as exc:
        print(f"Initial build failed: {exc}", flush=True)

    handler = _SourceChangeHandler(root)
    observer = Observer()
    observer.schedule(handler, str(root), recursive=True)
    observer.start()

    try:
        while observer.is_alive():
            observer.join(timeout=1)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        print("\nWatcher stopped.", flush=True)
