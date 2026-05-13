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
- Run small retrieval, local hard-query, and ToolRet evaluation datasets.

## Project Layout

- `src/core/` contains the general search, scoring, selection, sectioning, and view-building algorithms.
- `src/benchmarks/` contains benchmark adapters and metric code for retrieval and ToolRet.
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

Run ToolRet retrieval-only evaluation against ToolRet query/tool exports:

```powershell
uv run skill-agent eval-toolret --queries path/to/toolret_queries.jsonl --tools path/to/toolret_tools.jsonl --top-k 10 --limit 30
```

The ToolRet command evaluates `SkillSearcher` directly by default. Most run options can live in `config.toml`, so repeated experiments can usually be launched with:

```powershell
uv run skill-agent eval-toolret --config config.toml
```

JSONL and JSON exports work without extra dependencies; parquet exports require `pandas` with a parquet engine such as `pyarrow`. Put generated result JSON and checkpoints under `data/eval/toolret/results/`; that folder is ignored by git.

```toml
[embedding]
enabled = true
backend = "hf-transformers"
model = "data/models/BAAI/bge-base-en-v1.5"
batch_size = 8
max_length = 512
device = ""
cache_dir = "data/eval/toolret/embedding_cache"

[search]
weight_lexical = 1.0
weight_sparse_view = 0.35
weight_dense = 0.45
weight_rrf = 0.2
weight_capability = 0.25
weight_usage = 0.15
weight_input_type = 0.2
weight_output_type = 0.2
weight_penalty = 0.4

[toolret]
queries = "path/to/toolret_queries.jsonl"
tools = "path/to/toolret_tools.jsonl"
first_stage_candidates = ""
limit = 30
top_k = 10
category = "all"
use_instruction = true
retrieval_mode = "hybrid"
baseline = "hybrid"
llm = "mock"
first_stage_model = "BAAI/bge-base-en-v1.5"
first_stage_backend = "auto"
embed_batch_size = 8
embed_max_length = 512
embed_device = ""
candidate_pool_size = 100
rankgpt_window_size = 20
rankgpt_step_size = 10
workers = 1
output = "data/eval/toolret/results/toolret_result.json"
checkpoint = "data/eval/toolret/results/toolret_result.checkpoint.jsonl"
resume = false
```

To compare instruction-aware and query-only retrieval, run once with `--use-instruction` and once with `--no-instruction`.

Run the ToolRet paper's LLM agent reranking baseline:

```powershell
uv run skill-agent eval-toolret --queries path/to/toolret_queries.jsonl --tools path/to/toolret_tools.jsonl --first-stage-candidates path/to/nv_embed_candidates.jsonl --baseline toolret-rankgpt --llm openai-compatible --top-k 10 --candidate-pool-size 100
```

`--baseline toolret-rankgpt` implements the ToolRet paper's RankGPT-style zero-shot LLM reranker over first-stage candidates produced by NV-Embed-v1. It requires `--first-stage-candidates` and intentionally does not fall back to this repo's hybrid retriever.

Compare the paper-style LLM pipeline against the SkillSpec hybrid searcher in one run:

```powershell
uv run skill-agent eval-toolret --queries path/to/toolret_queries.jsonl --tools path/to/toolret_tools.jsonl --first-stage-candidates path/to/nv_embed_candidates.jsonl --baseline compare --llm openai-compatible --top-k 10 --candidate-pool-size 100
```

This comparison uses `SkillSpec`-derived tool documents for both sides. The hybrid side uses `SkillSearcher`; the LLM side uses the provided first-stage candidates followed by RankGPT-style reranking with the configured LLM.

Build practical local first-stage candidates from the same `SkillSpec` representation:

```powershell
uv run skill-agent build-toolret-candidates --queries path/to/toolret_queries.jsonl --tools path/to/toolret_tools.jsonl --output path/to/bge_candidates.jsonl --top-k 100 --model BAAI/bge-base-en-v1.5 --embedding-backend hf-transformers --max-length 512
```

This local-friendly path uses a standard Hugging Face encoder with mean pooling and writes the same candidate JSONL format consumed by `--first-stage-candidates`. It is not the ToolRet paper's NV-Embed-v1 first stage, but it is practical on consumer GPUs and useful for local comparisons.

Build paper-faithful NV-Embed-v1 first-stage candidates from the same `SkillSpec` representation:

```powershell
uv run skill-agent build-toolret-candidates --queries path/to/toolret_queries.jsonl --tools path/to/toolret_tools.jsonl --output path/to/nv_embed_candidates.jsonl --top-k 100 --model nvidia/NV-Embed-v1
```

This command loads `nvidia/NV-Embed-v1` locally with Hugging Face Transformers and `trust_remote_code=True`, embeds queries and `SkillSpec`-derived tool documents, and writes candidate JSONL that can be passed to `--first-stage-candidates`. NV-Embed-v1 is a gated 7B model, so the local environment must have `torch`, `transformers`, and the Hugging Face model terms accepted/authenticated before this command can run. Authenticate with `uv run hf auth login` after accepting access to the model on Hugging Face. Model files are cached under `~/.cache/huggingface/hub/`, and remote model code is cached under `~/.cache/huggingface/modules/transformers_modules/`.

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
- Milestone 6 has retrieval and ToolRet benchmark adapters.
- Milestone 7 is not implemented: optional skill invocation remains future work.
