#!/usr/bin/env python3
"""
UserPromptSubmit hook.
Runs context engine on the incoming query, computes confidence,
writes session state and injects a context summary into the prompt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from context_engine.policy import PolicySession, compute_confidence, normalize_path
from context_engine.ranker import format_output, rank_and_select, resolve_nodes
from context_engine.retrieval import run_query

_GRAPH_PATH = Path(".cecl/graph.json")


def _load_graph() -> dict | None:
    if not _GRAPH_PATH.exists():
        return None
    try:
        return json.loads(_GRAPH_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    payload = json.loads(sys.stdin.read())
    query: str = payload.get("prompt", "")

    graph = _load_graph()
    if not graph or not query.strip():
        sys.exit(0)

    # Run retrieval
    result = run_query(query, graph)
    selected: list[str] = result.get("nodes_selected", [])
    all_ids = [n["id"] for n in graph.get("nodes", []) if "id" in n]
    pool_ids = selected if selected else all_ids
    node_dicts = resolve_nodes(pool_ids, graph)
    ranked = rank_and_select(node_dicts, query)

    # Build session
    session = PolicySession()
    session.query = query
    session.confidence, session.mode = compute_confidence(ranked)

    # Seed allowed files from ranked nodes
    for node in ranked:
        fpath = node.get("file", "")
        if fpath:
            session.allow_file(fpath)

    session.save()
    session.save_log()

    # Inject compressed context as system hint (stdout appended to prompt)
    if ranked:
        context = format_output(query, ranked)
        out = {
            "prompt_suffix": (
                f"\n\n<llm-diet-context>\n{context}\n</llm-diet-context>\n"
                f"<!-- mode={session.mode} confidence={session.confidence} "
                f"files={[normalize_path(f).split('/')[-1] for f in session.allowed_files]} -->"
            )
        }
        sys.stdout.write(json.dumps(out))

    sys.exit(0)


if __name__ == "__main__":
    main()
