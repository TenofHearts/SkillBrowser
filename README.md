# Skill Search Agent

Local skill discovery and reading MVP for long-horizon LLM agents.

The project stores skills as local `skill.yaml` metadata plus `skill.md` instructional content. The current implementation focuses on the reader-first pipeline:

```text
skill_search -> skill_read -> apply skill instructions
```

Executable skill invocation is planned but not implemented yet.

## Current Capabilities

- Load and validate local skill specs from `data/skills`.
- Parse Markdown skill documents into named sections.
- Search skills with an in-memory dependency-light hybrid scorer: BM25, per-view token-vector cosine ranking, reciprocal-rank fusion, and request capability/type hints.
- Read a full skill document or a specific section with a token budget.
- Build a SQLite registry containing skills, documents, sections, and generated search views.
- Run small retrieval, local hard-query, and ToolRet evaluation datasets.

## Project Layout

- `src/core/` contains the general search, scoring, selection, sectioning, and view-building algorithms.
- `src/benchmarks/` contains benchmark adapters and metric code for retrieval and ToolRet.
- Top-level files in `src/` contain the CLI, agent loop, schema, loading, reading, registry, config, and LLM client code.

## Setup

```powershell
uv sync
```

## CLI

Validate local skills:

```powershell
uv run skill-agent validate-skills --skill-dir data/skills
```

Build the SQLite registry:

```powershell
uv run skill-agent build-index --skill-dir data/skills --index-dir data/indexes
```

Search skills:

```powershell
uv run skill-agent search "extract text from a pdf" --top-k 5 --skill-dir data/skills
```

Programmatic search requests also accept optional context fields while remaining compatible with the CLI shape:

```python
SkillSearchRequest(
    query="handle this paper",
    task_context="Need a structured analysis of a research PDF after text extraction.",
    required_capabilities=["extract_claim", "extract_method"],
    input_types=["paper_text"],
    output_types=["structured_text"],
)
```

Read a skill section:

```powershell
uv run skill-agent read research.paper_claim_method_finding --section procedure --max-tokens 2000 --skill-dir data/skills
```

Run retrieval evaluation:

```powershell
uv run skill-agent eval-retrieval --skill-dir tests/fixtures/skills --dataset tests/fixtures/retrieval_eval.jsonl --top-k 1
```

Run the local hard-query retrieval benchmark:

```powershell
uv run skill-agent eval-retrieval --skill-dir data/skills --dataset data/eval/local_hard_retrieval.jsonl --top-k 3
```

Run the LLM skill-selection agent with a deterministic mock model:

```powershell
uv run skill-agent run-agent "extract text from a PDF" --skill-dir data/skills --top-k 3 --llm mock
```

Run the agent with an OpenAI-compatible hosted model endpoint. The recommended default model for meaningful agent/tool-selection tests is `Qwen/Qwen3.5-397B-A17B`; lower-cost Qwen variants or GLM endpoints can be used by changing `model` in `config.toml`.

```powershell
Copy-Item config.example.toml config.toml
# Edit config.toml with your provider URL, API key, model, and benchmark defaults.
uv run skill-agent run-agent "extract text from a PDF" --skill-dir data/skills --top-k 3 --llm openai-compatible --config config.toml
```

`--skill-dir` is accepted either before or after the subcommand.

Run ToolRet retrieval-only evaluation against ToolRet query/tool exports:

```powershell
uv run skill-agent eval-toolret --queries path/to/toolret_queries.jsonl --tools path/to/toolret_tools.jsonl --top-k 10 --limit 30
```

The ToolRet command evaluates `SkillSearcher` directly, not the LLM agent. JSONL and JSON exports work without extra dependencies; parquet exports require `pandas` with a parquet engine such as `pyarrow`.

```toml
[toolret]
queries = "path/to/toolret_queries.jsonl"
tools = "path/to/toolret_tools.jsonl"
first_stage_candidates = ""
limit = 30
top_k = 10
category = "all"
use_instruction = true
baseline = "hybrid"
candidate_pool_size = 100
rankgpt_window_size = 20
rankgpt_step_size = 10
```

To compare instruction-aware and query-only retrieval, run once with `--use-instruction` and once with `--no-instruction`.

Run a ToolRet-style LLM agent reranking baseline:

```powershell
uv run skill-agent eval-toolret --queries path/to/toolret_queries.jsonl --tools path/to/toolret_tools.jsonl --first-stage-candidates path/to/nv_embed_candidates.jsonl --baseline rankgpt --llm openai-compatible --top-k 10 --candidate-pool-size 100
```

`--baseline rankgpt` implements a RankGPT-style zero-shot LLM reranker over first-stage candidates. To match the ToolRet paper most closely, pass first-stage candidates produced by NV-Embed-v1. If `--first-stage-candidates` is omitted, the command uses this repo's hybrid retriever as the first-stage fallback and reports token usage, latency, and parse failures for the LLM reranking calls.

## Tests

```powershell
uv run pytest -q
```

## Project Status

The project is in an early MVP state:

- Milestone 1 is mostly complete: schema, loader, Markdown reader, SQLite registry, and validation tests exist.
- Milestone 2 is partial: multi-view text is generated and persisted, and in-memory BM25 plus token-vector view indexes exist; persistent BM25, FAISS/dense indexing, id maps, and reloadable index files are not implemented.
- Milestone 3 is partial: search returns ranked skill cards from BM25/vector RRF candidates with normalized score breakdowns and request capability/type hints; persistent filters, learned dense retrieval, reranking, and search logs are not implemented.
- Milestone 4 is partial: `skill_read` behavior exists, but a dedicated context builder and read logs are not implemented.
- Milestone 5 now has an initial LLM-backed agent loop for search/read/final-answer workflows.
- Milestone 6 has retrieval and ToolRet benchmark adapters.
- Milestone 7 is not implemented: optional skill invocation remains future work.
