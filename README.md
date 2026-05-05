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
- Search skills with an in-memory lexical/capability scorer.
- Read a full skill document or a specific section with a token budget.
- Build a SQLite registry containing skills, documents, sections, and generated search views.
- Run small retrieval, local hard-query, and optional GatewayBench-lite evaluation datasets.

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

Run a GatewayBench-lite JSONL export:

```powershell
uv run skill-agent eval-gatewaybench-lite --dataset path/to/gatewaybench.jsonl --top-k 5 --limit 100
```

`--skill-dir` is accepted either before or after the subcommand.

## Tests

```powershell
uv run pytest -q
```

## Project Status

The project is in an early MVP state:

- Milestone 1 is mostly complete: schema, loader, Markdown reader, SQLite registry, and validation tests exist.
- Milestone 2 is partial: multi-view text is generated and persisted, but persistent BM25, FAISS/dense indexing, id maps, and reloadable index files are not implemented.
- Milestone 3 is partial: search returns ranked skill cards with score breakdowns, but it is not yet a true hybrid search engine with RRF, dense retrieval, filters, or search logs.
- Milestone 4 is partial: `skill_read` behavior exists, but a dedicated context builder and read logs are not implemented.
- Milestones 5-7 are not implemented: agent loop, full evaluation pipeline, and optional skill invocation remain future work.
