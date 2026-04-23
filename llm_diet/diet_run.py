"""diet-run — launch Claude Code with the llm-diet shadow MCP server active.

The shadow server intercepts all read_file calls and returns compressed
call-graph output. The built-in Read tool is disallowed so Claude cannot
bypass the shadow server by reading files directly.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    # Parse optional working directory argument
    workdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    workdir = workdir.resolve()

    # Check 1: graph must exist
    graph_path = workdir / ".cecl" / "graph.json"
    if not graph_path.exists():
        print(f"Error: no graph found at {graph_path}")
        print("Run `context-engine index .` first to build the call graph.")
        sys.exit(1)

    # Check 2: context_engine must be importable
    try:
        import importlib.util
        if importlib.util.find_spec("context_engine") is None:
            raise ImportError
    except ImportError:
        print("Error: context_engine is not installed.")
        print("Run `pip install llm-diet` to install it.")
        sys.exit(1)

    # Check 3: .mcp.json must exist (shadow server must be registered)
    mcp_config = workdir / ".mcp.json"
    if not mcp_config.exists():
        print(f"Error: no .mcp.json found at {mcp_config}")
        print("Run `context-engine install` to register the shadow MCP server.")
        sys.exit(1)

    # Build environment: inherit everything, add LLM_DIET_STRICT=1
    env = os.environ.copy()
    env["LLM_DIET_STRICT"] = "1"

    # Resolve claude executable via PATH (bare "claude" fails on Windows)
    claude_exe = shutil.which("claude")
    if claude_exe is None:
        print("Error: claude not found on PATH. Install Claude Code first.")
        sys.exit(1)

    # Build claude command:
    #   --mcp-config .mcp.json     load the shadow MCP server
    #   --strict-mcp-config        ignore all other MCP servers (shadow only)
    #   --disallowed-tools Read    block built-in Read so all file reads go
    #                              through the shadow server's read_file tool
    cmd = [
        claude_exe,
        "--mcp-config", str(mcp_config),
        "--disallowed-tools", "Read,Bash,Glob,Grep",
    ]

    print(f"[diet-run] launching: {' '.join(cmd)}")

    if sys.platform == "win32":
        result = subprocess.run(cmd, env=env, shell=False)
        sys.exit(result.returncode)
    else:
        os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
