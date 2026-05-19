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
- Search skills with an in-memory hybrid scorer: BM25 sparse retrieval, optional learned dense embeddings over per-view skill text, reciprocal-rank fusion, and request capability/type hints.
- Read a full skill document or a specific section with a token budget.
- Build a SQLite registry containing skills, documents, sections, and generated search views.
- Run small retrieval and local hard-query evaluation datasets.
- Run SRA-Bench retrieval with this project's Hybrid agent and pass the results into the SR-Agents infer/evaluate pipeline.

## Project Layout

- `src/core/` contains the general search, scoring, selection, sectioning, and view-building algorithms.
- `src/benchmarks/` contains benchmark adapters and metric code for retrieval and SRA-Bench integration.
- `benchmarks/SR-Agents/` is a git submodule for the upstream SRA-Bench/SR-Agents benchmark.
- `scripts/` contains convenience scripts for SRA-Bench prepare, retrieval, and end-to-end evaluation runs.
- Top-level files in `src/` contain the CLI, agent loop, schema, loading, reading, registry, config, and LLM client code.

## Setup

```powershell
uv sync
```

Install optional embedding dependencies for the machine you are on:

```powershell
# CPU-only machines
uv sync --extra cpu

# NVIDIA GPU machines on Windows/Linux
uv sync --extra cu128
```

The `cpu` and `cu128` extras are mutually exclusive. The CUDA extra uses PyTorch's CUDA 12.8 wheel index on Windows/Linux; macOS falls back to the standard PyPI wheel.

## CLI

Validate local skills:

```powershell
uv run skill-agent validate-skills --skill-dir data/skills
```

Build the SQLite registry:

```powershell
uv run skill-agent build-index --skill-dir data/skills --index-dir data/indexes
```

Build the registry and persist dense view vectors:

```powershell
uv run skill-agent build-index --skill-dir data/skills --index-dir data/indexes --retrieval-mode hybrid --embedding-backend hf-transformers
```

Search skills:

```powershell
uv run skill-agent search "extract text from a pdf" --top-k 5 --skill-dir data/skills
```

Compare retrieval modes:

```powershell
# BM25-only sparse baseline
uv run skill-agent search "extract text from a pdf" --top-k 5 --retrieval-mode bm25

# Dense-only with a local Hugging Face encoder
uv run skill-agent search "extract text from a pdf" --top-k 5 --retrieval-mode dense --embedding-backend hf-transformers

# Hybrid BM25 + dense retrieval
uv run skill-agent search "extract text from a pdf" --top-k 5 --retrieval-mode hybrid --embedding-backend hf-transformers
```

For fast tests or smoke checks, `--embedding-backend fake` uses a deterministic local fake embedder. It is not a real retrieval model.

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

The same retrieval flags work for evaluation:

```powershell
uv run skill-agent eval-retrieval --skill-dir data/skills --dataset data/eval/local_hard_retrieval.jsonl --top-k 3 --retrieval-mode hybrid --embedding-backend hf-transformers
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

## SRA-Bench / SR-Agents

The upstream benchmark is tracked as a submodule:

```powershell
git submodule update --init --recursive
uv run python scripts/sra_bench.py prepare
```

Run this project's Hybrid retriever on one SRA-Bench dataset and write an SR-Agents-compatible retrieval file:

```powershell
uv run python scripts/sra_bench.py retrieve --dataset theoremqa --top-k 50 --config config.toml
```

The retrieval file lands under `data/eval/sra/results/retrieval/` and can be consumed by SR-Agents' native `infer` and `evaluate` stages. For a full retrieve -> infer -> evaluate run against an OpenAI-compatible endpoint:

```powershell
uv run python scripts/sra_bench.py run `
  --dataset theoremqa `
  --model gpt-4o-mini `
  --api-base https://api.openai.com/v1 `
  --top-k 50 `
  --provider-k 1 `
  --engine direct
```

PowerShell wrappers are also available:

```powershell
.\scripts\sra_prepare.ps1
.\scripts\sra_retrieve_hybrid.ps1 -Dataset theoremqa -TopK 50
.\scripts\sra_run_hybrid_eval.ps1 -Dataset theoremqa -Model gpt-4o-mini -ApiBase https://api.openai.com/v1
```

For quick local smoke tests, add `--limit 5` or pass `-Limit 5` to the retrieval wrapper.

The full run uses this repo for retrieval and the SR-Agents submodule for benchmark inference/evaluation. ToolQA requires the external ToolQA corpus described in the upstream SR-Agents README.

## Tests

```powershell
uv run pytest -q
```

## Project Status

The project is in an early MVP state:

- Milestone 1 is mostly complete: schema, loader, Markdown reader, SQLite registry, and validation tests exist.
- Milestone 2 is partial: multi-view text is generated and persisted, and in-memory BM25 plus optional dense view embeddings exist; persistent BM25, FAISS indexing, id maps, and reloadable vector files are not implemented.
- Milestone 3 is partial: search returns ranked skill cards from BM25, sparse-view, and optional dense RRF candidates with normalized score breakdowns and request capability/type hints; persistent filters, reranking, and search logs are not implemented.
- Milestone 4 is partial: `skill_read` behavior exists, but a dedicated context builder and read logs are not implemented.
- Milestone 5 now has an initial LLM-backed agent loop for search/read/final-answer workflows.
- Milestone 6 has retrieval and SRA-Bench benchmark adapters.
- Milestone 7 is not implemented: optional skill invocation remains future work.
