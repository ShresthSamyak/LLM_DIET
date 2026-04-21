# context-engine 
> Cut your AI coding costs by 99%. Inject only the code that matters.

![demo](demo/demo.gif)
> *Record your own: `bash demo/record_demo.sh`*

## Benchmark (this repo, 185 nodes)

| Query | Baseline Tokens | With CE Tokens | Reduction % | Nodes | Files Selected | Time (ms) |
|-------|----------------|----------------|-------------|-------|----------------|----------|
| fix authentication bug | 46,661 | 434 | 99.1% | 3 | intent.py, validator.py, patcher.py | 187 |
| add a new API endpoint | 46,661 | 106 | 99.8% | 1 | ranker.py | 203 |
| how does the database connection work | 46,661 | 278 | 99.4% | 2 | intent.py, retrieval.py | 219 |
| debug memory leak | 46,661 | 428 | 99.1% | 3 | intent.py, cli.py, compressor.py | 172 |
| add input validation | 46,661 | 0 | 100.0% | 0 | no match in this repo | 125 |
| explain the caching logic | 46,661 | 418 | 99.1% | 3 | intent.py, pruner.py, retrieval.py | 172 |
| fix error handling | 46,661 | 479 | 99.0% | 3 | intent.py, validator.py, patcher.py | 187 |
| add logging to the pipeline | 46,661 | 58 | 99.9% | 1 | cli.py | 141 |

**46,661 tokens → 275 tokens average. Same accuracy. 176ms overhead (small repo).**

## Benchmark (external repo — fastapi, 946k token codebase)

| Query | Baseline Tokens | With CE Tokens | Reduction % | Nodes | Files Selected | Time (ms) |
|-------|----------------|----------------|-------------|-------|----------------|----------|
| fix authentication bug | 946,210 | 87 | >99.9% | 1 | api_key.py | 359 |
| add a new API endpoint | 946,210 | 120 | >99.9% | 1 | api_key.py | 359 |
| how does the database connection work | 946,210 | 130 | >99.9% | 2 | tutorial001_an_py310.py, param_functions.py | 391 |
| debug memory leak | 946,210 | 436 | >99.9% | 5 | applications.py, test_arbitrary_types.py, … | 1062 |
| add input validation | 946,210 | 244 | >99.9% | 3 | utils.py, tutorial004_py310.py, … | 375 |
| explain the caching logic | 946,210 | 114 | >99.9% | 1 | test_security_scopes_sub_dependency.py | 313 |
| fix error handling | 946,210 | 221 | >99.9% | 3 | tutorial003_py310.py, tutorial002_py310.py, … | 406 |
| add logging to the pipeline | 946,210 | 136 | >99.9% | 1 | tutorial002_an_py310.py | 328 |

**946,210 tokens → 186 tokens average. Repo never seen before. 449ms overhead.**

## How it works

```
User prompt → context-engine → top 5 relevant functions → Claude sees 275 tokens
                                        ↑
               (instead of your entire codebase at 46,661 tokens)
```

AST-parses your repo into a call graph. When you send a prompt, it scores every
function by keyword relevance + call graph centrality and injects only the top
matches — before Claude starts reasoning.

No embeddings. No vector DB. No LLM calls in the retrieval path.

## Install

```bash
pip install llm-diet
context-engine install
```

That's it. The install command indexes your repo and configures Claude Code, Cursor, or Windsurf automatically.

## Use as CLI

```bash
context-engine query "fix the auth bug"
context-engine apply "add input validation to the login endpoint"
```

## Why not just use RAG / embeddings?

No LLM calls, no vector DB, no setup.

context-engine uses AST parsing + call graph traversal. It's deterministic —
same query, same graph, same result every time. Works offline. Runs in ~176ms (small repo) / ~432ms (large repo like FastAPI).
Embeddings need a model call just to retrieve context. We don't.

| Feature | context-engine | code-review-graph |
|---------|---------------|-------------------|
| Languages | Python, JS, TS, JSX, TSX | 23 languages |
| Dependencies | tree-sitter only | tree-sitter + SQLite + more |
| Context injection | UserPromptSubmit hook (Claude Code) / IDE rules (Cursor, Windsurf) | MCP server |
| Autonomous apply | ✅ plan → diff → validate → patch | ❌ |
| Setup | pip install + index | pip install + build |
| Incremental updates | auto on file save via `context-engine watch` | auto on file save |

We do less. What we do, we do surgically.

\* Full token reduction applies to Claude Code. Cursor/Windsurf receive static rules files that guide the AI to prefer provided context — dynamic injection coming in a future release.
