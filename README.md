# llm-diet

[![PyPI version](https://img.shields.io/pypi/v/llm-diet)](https://pypi.org/project/llm-diet/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/llm-diet)](https://pypi.org/project/llm-diet/)
[![License: MIT](https://img.shields.io/github/license/ShresthSamyak/LLM_DIET)](LICENSE)
[![Downloads](https://img.shields.io/badge/downloads-PyPI-brightgreen)](https://pypi.org/project/llm-diet/)

**Give Claude the right context upfront. Fewer turns, faster answers, lower cost.**

Deterministic context retrieval for AI coding tools. Parses your repo into a call graph, scores every function against your query, and injects the top matches — before Claude starts reasoning. No embeddings, no vector DB, no LLM calls in the retrieval path.

## The Problem

Every Claude Code session starts blind. Claude explores your entire codebase before answering — reading files, listing directories, running commands. That exploration costs tokens and time.

```
Without llm-diet:
  Your prompt → Claude explores codebase → finds relevant code → answers
  Cost: $0.19 for a simple bug fix session

With llm-diet:
  Your prompt + injected context → Claude answers faster
  Cost: $0.035 for the same depth of answer
```

Real test on a 1,240-node project: 5x cheaper per session.

## Benchmark

**This repo (185 nodes, 46k tokens)**

| Query | Baseline | With llm-diet | Reduction | Time |
|-------|----------|---------------|-----------|------|
| fix authentication bug | 46,661 | 434 | 99.1% | 187ms |
| add a new API endpoint | 46,661 | 106 | 99.8% | 203ms |
| debug memory leak | 46,661 | 428 | 99.1% | 172ms |
| add logging to the pipeline | 46,661 | 58 | 99.9% | 141ms |

**46,661 → 275 tokens average. 176ms overhead.**

**FastAPI repo (946k tokens — repo never seen before)**

| Query | Baseline | With llm-diet | Reduction | Time |
|-------|----------|---------------|-----------|------|
| fix authentication bug | 946,210 | 87 | >99.9% | 359ms |
| add a new API endpoint | 946,210 | 120 | >99.9% | 359ms |
| how does the database connection work | 946,210 | 130 | >99.9% | 391ms |
| debug memory leak | 946,210 | 436 | >99.9% | 1062ms |
| add input validation | 946,210 | 244 | >99.9% | 375ms |
| explain the caching logic | 946,210 | 114 | >99.9% | 313ms |
| fix error handling | 946,210 | 221 | >99.9% | 406ms |
| add logging to the pipeline | 946,210 | 136 | >99.9% | 328ms |

**946,210 → 186 tokens injected. 5x cheaper sessions in practice.**

## How It Works

```
repo files (.py, .js, .ts, .jsx, .tsx)
   ↓
AST parser  (no LLM — pure tree-sitter)
   ↓
call graph  (.cecl/graph.json)
   ↓
query  →  keyword expansion  →  BFS traversal  →  top 5 functions
   ↓
injected into Claude before reasoning starts
```
Same query + same graph = same result. Deterministic by design.

## MCP Shadow Server (new in 0.1.7)

By default, Claude Code explores your codebase after receiving injected context — reading files directly even when we've already told it what's relevant.

The shadow server fixes this at the transport layer. When `context-engine install` detects a built graph, it registers a local MCP server in `.mcp.json`. Claude Code routes all `read_file` calls through this server, which returns compressed call-graph versions instead of raw files.

**Real numbers on a 40-node project (coupon-hunter-poc):**

| File | Original | Compressed | Reduction |
|------|----------|------------|-----------|
| playwright_amazon.py | 6,590 chars | 872 chars | 86% |
| orchestrator.py | 10,492 chars | 2,169 chars | 79% |
| connectors/playwright_amazon.py | 3,067 chars | 631 chars | 79% |

**Overall: 32,856 → 10,044 chars across all indexed files (69% reduction, ~5,700 tokens saved per full codebase read)**

Files not in the graph pass through unchanged. Binary files are skipped. Large unindexed files (>50k chars) are truncated to 200 lines with a note to run `context-engine index`.

## diet-run (new in 0.1.8)

`diet-run` is a CLI wrapper that launches Claude Code in fully enforced Low Bandwidth Mode:

```bash
diet-run                    # run in current directory
diet-run /path/to/project   # run in specific directory
```

What it does:
- Sets `LLM_DIET_STRICT=1` — unindexed files return an error instead of raw content
- Passes `--mcp-config .mcp.json` — shadow server is the only file reader
- Passes `--disallowed-tools Read` — Claude's built-in Read tool is blocked
- Requires `.cecl/graph.json` and `.mcp.json` to exist before launching

Run `context-engine install` first to set up the shadow server, then use `diet-run` instead of `claude` to open sessions.

## Quick Start

```bash
pip install llm-diet
context-engine install    # indexes repo + configures your AI tool
# open Claude Code and start coding
```

## Commands

| Command | Description |
|---------|-------------|
| `context-engine install` | Index repo and configure Claude Code / Cursor / Windsurf |
| `context-engine index .` | (Re)build the call graph |
| `context-engine query "fix auth bug"` | See what would be injected for a query |
| `context-engine apply "add endpoint"` | Plan → diff → validate → patch (needs `ANTHROPIC_API_KEY`) |
| `context-engine watch .` | Auto-reindex on file save |

## Platform Support

| Platform | Integration | Token reduction |
|----------|-------------|-----------------|
| Claude Code | `UserPromptSubmit` hook — dynamic injection on every prompt | Full (186 tokens avg) |
| Cursor | Static rules file written to `.cursor/rules/` | Guides AI; no dynamic injection |
| Windsurf | Static rules file written to `.windsurf/rules/` | Guides AI; no dynamic injection |

Full token reduction verified on Claude Code. Cursor/Windsurf dynamic injection on the roadmap.

## Why Not RAG?

| | llm-diet | Embeddings / RAG | code-review-graph |
|-|----------|-----------------|-------------------|
| Retrieval method | AST + call graph | Vector similarity | AST + SQLite |
| LLM calls to retrieve | 0 | 1+ | 0 |
| Deterministic | Yes | No | Yes |
| Setup | `pip install` + `index` | Model + DB infra | `pip install` + `build` |
| Languages | Python, JS, TS, JSX, TSX | Any | 23 languages |
| Autonomous apply | Yes | No | No |
| Works offline | Yes | No | Yes |

We do less. What we do, we do surgically.

## Contributing

Good first issues:
- **Dynamic injection for Cursor/Windsurf** — extend beyond Claude Code's `UserPromptSubmit`
- **More language parsers** — add Go, Rust, Java following the `FileParseResult` interface in `parser.py`
- **Better keyword expansion** — improve domain-specific term mapping in `retrieval.py`

Open an issue or send a PR.

## License

MIT
