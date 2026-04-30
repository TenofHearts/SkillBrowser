# Project Plan: Capability-Aware Local Skill Search for Long-Horizon LLM Agents

## 1. Project Goal

Build a local skill discovery and context-loading framework where an LLM agent can solve long-horizon tasks by dynamically searching, reading, and applying skills from a local skill library instead of loading all skill descriptions into the system prompt.

Skills are not necessarily executable tools. A skill can be:

```text
- an instructional document
- a workflow guide
- a reference note
- a code recipe
- a prompt pattern
- a tool usage guide
- an executable tool wrapper
- a hybrid of documentation and executable interface
```

Executable tools are a subclass of skills, not the default assumption.

The target behavior is:

1. A user provides a complex long-horizon task.
2. The LLM performs planning and reasoning.
3. When the LLM realizes that specialized capability or procedural knowledge is needed, it calls an internal framework tool named `skill_search`.
4. `skill_search` performs hybrid search over a local skill library.
5. The search tool returns a small set of relevant compact skill cards.
6. The LLM selects the most relevant skill to inspect.
7. The LLM reads the selected skill document by calling `skill_read`, loading relevant sections into context.
8. The LLM applies the skill instructions during reasoning.
9. If the selected skill explicitly declares an executable interface, the LLM may optionally invoke it through the `skill_invoke` interface.
10. The framework records discovery, reading, application, optional invocation, and evaluation logs.

The core pipeline is:

```text
skill_search → skill_read → apply skill instructions in context
```

With the optional branch:

```text
skill_search → skill_read → skill_invoke
                           only if execution_available = true
```

The central design principle is:

> Do not treat skills as static prompt text.
> Treat skills as searchable, readable, versioned, and measurable local procedural knowledge.
> Executable tools are an optional subclass, not the default.

---

## 2. Core Research Question

Can dynamic skill search and skill reading reduce context pollution and improve task performance compared with loading all skill descriptions into the agent prompt?

Sub-questions:

1. Can hybrid retrieval find relevant skills from a large local skill library?
2. Can LLMs select the correct skill documents to read from retrieved candidates?
3. Does reading selected skill sections improve task performance compared with loading no skills or all skills?
4. Does dynamic skill loading reduce prompt tokens without reducing task success rate?
5. Does the framework scale better than all-skill prompt loading as the skill library grows?
6. Can the system support multi-step long-horizon tasks where skills are discovered and read during reasoning rather than preloaded?

---

## 3. System Overview

The framework explicitly distinguishes five phases:

```text
1. Skill discovery      — finding relevant skills via search
2. Skill reading        — loading skill document content
3. Skill context loading — inserting skill sections into conversation
4. Skill application    — LLM applies skill instructions in reasoning
5. Optional skill execution — invoking executable skills if declared
```

The reader-first pipeline:

```text
User Task
  ↓
LLM Agent Planner
  ↓
Need specialized capability / procedural knowledge?
  ↓ yes
Internal Tool Call: skill_search
  ↓
Capability-Aware Hybrid Search
  ↓
Candidate Skill Cards
  ↓
LLM selects skill to inspect
  ↓
Internal Tool Call: skill_read
  ↓
Skill Context Builder loads relevant skill sections
  ↓
LLM applies skill instructions in reasoning
  ↓
Optional: if execution_available = true
    Internal Tool Call: skill_invoke
  ↓
LLM Continues Planning
  ↓
Final Answer / Artifact
```

The search engine should not merely search for textually similar skills. It should search for skills that can satisfy the capability needs implied by the current task state.

---

## 4. Recommended Technical Stack

### 4.1 Language and Runtime

Use Python as the main implementation language. Use `uv` as the virtual environment manager

Recommended version:

```text
Python >= 3.11
```

Reason:

- Strong ecosystem for retrieval, embeddings, databases, evaluation, and LLM agent orchestration.
- Easy integration with local scripts and skill execution.
- Suitable for rapid experimental iteration.

---

### 4.2 Embedding and Reranking Models

For the MVP:

```text
Embedding model:
  BAAI/bge-small-en-v1.5
  or
  intfloat/e5-base-v2

Reranker:
  optional in MVP
  later: BAAI/bge-reranker-base
```

The MVP should support pluggable embedding backends:

```text
local sentence-transformers
OpenAI-compatible embedding API
custom embedding function
```

---

### 4.3 LLM Interface

The framework should not depend on one specific LLM provider.

Implement a generic LLM client interface:

```python
class BaseLLMClient:
    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        ...
```

Implement adapters later:

```text
OpenAI-compatible API
local vLLM / Ollama / LLaMAFactory API
mock LLM for unit tests
```

---

## 5. Repository Structure

Recommended structure:

```text
skill-search-agent/
├── README.md
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   ├── embedding.yaml
│   └── eval.yaml
│
├── data/
│   ├── skills/
│   │   ├── pdf_extract/
│   │   │   ├── skill.yaml
│   │   │   ├── skill.md
│   │   │   └── main.py
│   │   ├── paper_claim_method_finding/
│   │   │   ├── skill.yaml
│   │   │   └── skill.md
│   │   ├── csv_analyze/
│   │   │   ├── skill.yaml
│   │   │   ├── skill.md
│   │   │   └── main.py
│   │   └── ...
│   │
│   ├── indexes/
│   │   ├── faiss_description.index
│   │   ├── faiss_capability.index
│   │   ├── bm25.pkl
│   │   └── id_map.json
│   │
│   └── eval/
│       ├── toolret/
│       ├── synthetic_tasks/
│       └── results/
│
├── src/
│   └── skill_search_agent/
│       ├── __init__.py
│       │
│       ├── db/
│       │   ├── models.py
│       │   ├── session.py
│       │   └── migrations.py
│       │
│       ├── schema/
│       │   ├── skill.py
│       │   ├── search.py
│       │   ├── reading.py
│       │   ├── invocation.py
│       │   └── logs.py
│       │
│       ├── indexing/
│       │   ├── loader.py
│       │   ├── normalizer.py
│       │   ├── embedder.py
│       │   ├── bm25_index.py
│       │   ├── vector_index.py
│       │   └── build_index.py
│       │
│       ├── retrieval/
│       │   ├── query_parser.py
│       │   ├── hybrid_search.py
│       │   ├── fusion.py
│       │   ├── filters.py
│       │   └── reranker.py
│       │
│       ├── reading/
│       │   ├── reader.py
│       │   ├── section_parser.py
│       │   ├── context_builder.py
│       │   └── compression.py
│       │
│       ├── agent/
│       │   ├── planner.py
│       │   ├── tool_specs.py
│       │   ├── skill_selector.py
│       │   └── loop.py
│       │
│       ├── execution/
│       │   ├── registry.py
│       │   ├── executor.py
│       │   ├── sandbox.py
│       │   └── verifier.py
│       │
│       ├── eval/
│       │   ├── metrics.py
│       │   ├── toolret_adapter.py
│       │   ├── synthetic_benchmark.py
│       │   └── run_eval.py
│       │
│       └── cli.py
│
├── tests/
│   ├── test_skill_schema.py
│   ├── test_skill_document_schema.py
│   ├── test_indexing.py
│   ├── test_hybrid_search.py
│   ├── test_skill_read.py
│   ├── test_context_builder.py
│   ├── test_skill_invocation.py  (optional executable skill tests)
│   └── test_eval_metrics.py
│
└── logs/
    ├── runs/
    └── eval/
```

Reading module responsibilities:

```text
reader.py:
  - load full skill document
  - load requested section
  - enforce max_tokens
  - return structured SkillReadResponse

section_parser.py:
  - parse markdown skill.md into sections
  - map headings to known section names

context_builder.py:
  - build skill context blocks to insert into the LLM conversation
  - include skill metadata, section content, usage notes, and caveats
  - avoid loading unnecessary sections

compression.py:
  - optional summarization or truncation for long skill documents
```

The Skill Context Builder should support levels:

```text
Level 0: compact search card
Level 1: overview
Level 2: specific sections
Level 3: full skill document
```

---

## 6. Database Design

Use SQLite for the MVP.

The database should store:

1. Skill metadata.
2. Skill capabilities.
3. Skill input/output schemas.
4. Skill examples.
5. Skill dependencies.
6. Skill execution statistics.
7. Skill invocation logs.
8. Search logs.
9. Agent trajectory logs.

---

## 7. Skill Metadata Schema

Each skill should be defined by a `skill.yaml` file.

The schema supports documentation-first skills, executable skills, and hybrid skills.

### 7.1 Key Fields

#### `skill_type`

Allowed values:

```text
instructional
workflow
reference
prompt_pattern
code_recipe
tool_usage_guide
tool_wrapper
hybrid
```

#### `interaction`

```yaml
interaction:
  mode: read_then_apply
  readable: true
  executable: false
  default_read_level: overview
```

Allowed `interaction.mode` values:

```text
read_then_apply
read_then_generate_code
read_then_execute
execute_directly
reference_only
```

#### `content`

```yaml
content:
  format: markdown
  path: skill.md
  sections:
    - overview
    - procedure
    - output_format
    - examples
    - failure_modes
```

#### `execution` (optional)

For documentation-first skills:

```yaml
execution:
  mode: none
```

For executable skills:

```yaml
execution:
  mode: python_function
  module: skills.pdf_extract.main
  function: extract_text
```

Allowed `execution.mode` values:

```text
none
python_function
subprocess
http_local
mock
```

---

### 7.2 Example: Documentation-First Workflow Skill

```yaml
id: research.paper_claim_method_finding
name: Paper Claim-Method-Finding Workflow
version: 0.1.0

status: active
skill_type: workflow

category:
  primary: research
  secondary:
    - academic_paper_processing
    - analysis

description:
  short: A workflow for extracting claim, method, and findings from academic papers.
  long: >
    This skill guides an LLM to analyze academic papers by identifying
    the central claim, the proposed method, empirical findings, evidence,
    and limitations.

capabilities:
  - id: extract_claim
    description: Identify the central claim of an academic paper.
  - id: extract_method
    description: Identify the method proposed or used in the paper.
  - id: extract_findings
    description: Identify the empirical or theoretical findings.

interaction:
  mode: read_then_apply
  readable: true
  executable: false
  default_read_level: overview

content:
  format: markdown
  path: skill.md
  sections:
    - overview
    - procedure
    - output_format
    - examples
    - failure_modes

when_to_use:
  - The user asks to analyze an academic paper.
  - The task requires extracting claims, methods, or findings.
  - The user asks to summarize a research paper structurally.

when_not_to_use:
  - The input is not an academic paper.
  - The user only needs a brief summary without structure.

input_types:
  - paper_text
  - academic_text

output_types:
  - structured_text
  - markdown

examples:
  positive:
    - user_query: "Read this paper and summarize its claim, method, and findings."
      reason: "The task requires structured academic paper analysis."
    - user_query: "What is the main contribution of this paper?"
      reason: "Extracting the central claim is a core capability."
  negative:
    - user_query: "Translate this paper to French."
      reason: "Translation is not part of this workflow."

execution:
  mode: none

tags:
  - research
  - academic
  - paper
  - analysis
  - workflow
```

---

### 7.3 Example: Executable Tool Wrapper Skill

```yaml
id: pdf.extract_text
name: PDF Text Extractor
version: 0.1.0

status: active
skill_type: tool_wrapper

category:
  primary: document_processing
  secondary:
    - pdf
    - text_extraction
    - academic_paper_processing

description:
  short: Extract text from PDF files while preserving page order.
  long: >
    This skill reads a local PDF file and extracts page-level text.
    Use it when the user provides a PDF and the task requires reading,
    summarizing, quoting, or analyzing the textual content of the PDF.

capabilities:
  - id: read_pdf
    description: Read a local PDF file.
  - id: extract_page_text
    description: Extract text page by page.
  - id: preserve_page_order
    description: Preserve the original page order.

interaction:
  mode: execute_directly
  readable: true
  executable: true
  default_read_level: overview

content:
  format: markdown
  path: skill.md
  sections:
    - overview
    - schema
    - examples

when_to_use:
  - The input file is a PDF.
  - The user asks to summarize, analyze, quote, or search inside a PDF.
  - The task needs page-level text extraction.

when_not_to_use:
  - The PDF is image-only and OCR is required.
  - The input is not a PDF.
  - The user only asks a conceptual question about PDF files.

input_types:
  - pdf
  - file_path

output_types:
  - json
  - text

input_schema:
  type: object
  required:
    - file_path
  properties:
    file_path:
      type: string
      description: Local path to the PDF file.
    pages:
      type: array
      items:
        type: integer
      description: Optional list of page numbers to extract.

output_schema:
  type: object
  required:
    - pages
  properties:
    pages:
      type: array
      items:
        type: object
        properties:
          page_number:
            type: integer
          text:
            type: string

examples:
  positive:
    - user_query: "Summarize this PDF paper."
      reason: "The task requires reading textual content from a PDF."
    - user_query: "Extract the method section from this paper."
      reason: "The method section must first be obtained from PDF text."
  negative:
    - user_query: "Explain what a PDF file is."
      reason: "No local PDF processing is required."
    - user_query: "Read text from this screenshot."
      reason: "OCR is required instead of PDF text extraction."

dependencies:
  python:
    - pymupdf
  system: []

permissions:
  filesystem:
    read: true
    write: false
  network: false
  shell: false

risk:
  level: low
  notes: "Reads local files only."

execution:
  mode: python_function
  module: skills.pdf_extract.main
  function: extract_text

cost:
  expected_latency_ms: 500
  expected_token_cost: low
  expected_compute: cpu

tags:
  - pdf
  - document
  - text
  - academic
```

---

## 8. SQLite Tables

### 8.1 `skills`

```sql
CREATE TABLE skills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    status TEXT NOT NULL,
    skill_type TEXT NOT NULL,
    interaction_mode TEXT NOT NULL,
    readable INTEGER NOT NULL,
    executable INTEGER NOT NULL,
    default_read_level TEXT,
    category_primary TEXT,
    description_short TEXT,
    description_long TEXT,
    execution_mode TEXT,
    execution_module TEXT,
    execution_function TEXT,
    risk_level TEXT,
    expected_latency_ms INTEGER,
    expected_token_cost TEXT,
    expected_compute TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

---

### 8.2 `skill_categories`

```sql
CREATE TABLE skill_categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id TEXT NOT NULL,
    category TEXT NOT NULL,
    FOREIGN KEY(skill_id) REFERENCES skills(id)
);
```

---

### 8.3 `skill_capabilities`

```sql
CREATE TABLE skill_capabilities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id TEXT NOT NULL,
    capability_id TEXT NOT NULL,
    description TEXT,
    FOREIGN KEY(skill_id) REFERENCES skills(id)
);
```

---

### 8.4 `skill_usage_rules`

```sql
CREATE TABLE skill_usage_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id TEXT NOT NULL,
    rule_type TEXT NOT NULL,
    content TEXT NOT NULL,
    FOREIGN KEY(skill_id) REFERENCES skills(id)
);
```

`rule_type` should be one of:

```text
when_to_use
when_not_to_use
safety_note
execution_note
```

---

### 8.5 `skill_examples`

```sql
CREATE TABLE skill_examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id TEXT NOT NULL,
    example_type TEXT NOT NULL,
    user_query TEXT NOT NULL,
    reason TEXT,
    FOREIGN KEY(skill_id) REFERENCES skills(id)
);
```

`example_type` should be one of:

```text
positive
negative
```

---

### 8.6 `skill_schemas`

```sql
CREATE TABLE skill_schemas (
    skill_id TEXT PRIMARY KEY,
    input_schema_json TEXT,
    output_schema_json TEXT,
    FOREIGN KEY(skill_id) REFERENCES skills(id)
);
```

`input_schema_json` and `output_schema_json` are optional. Documentation-first skills may not have schemas.

---

### 8.7 `skill_dependencies`

```sql
CREATE TABLE skill_dependencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id TEXT NOT NULL,
    dependency_type TEXT NOT NULL,
    dependency_name TEXT NOT NULL,
    version_constraint TEXT,
    FOREIGN KEY(skill_id) REFERENCES skills(id)
);
```

`dependency_type` should be one of:

```text
python
system
model
service
environment
```

---

### 8.8 `skill_permissions`

```sql
CREATE TABLE skill_permissions (
    skill_id TEXT PRIMARY KEY,
    filesystem_read INTEGER NOT NULL,
    filesystem_write INTEGER NOT NULL,
    network INTEGER NOT NULL,
    shell INTEGER NOT NULL,
    FOREIGN KEY(skill_id) REFERENCES skills(id)
);
```

---

### 8.9 `skill_embeddings`

```sql
CREATE TABLE skill_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id TEXT NOT NULL,
    view_name TEXT NOT NULL,
    vector_index_name TEXT NOT NULL,
    vector_position INTEGER NOT NULL,
    source_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(skill_id) REFERENCES skills(id)
);
```

`view_name` should be one of:

```text
description
capability
examples
schema
usage_rules
content_section
```

---

### 8.10 `skill_documents`

```sql
CREATE TABLE skill_documents (
    skill_id TEXT PRIMARY KEY,
    content_format TEXT NOT NULL,
    content_path TEXT NOT NULL,
    full_text TEXT,
    token_count INTEGER,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(skill_id) REFERENCES skills(id)
);
```

---

### 8.11 `skill_sections`

```sql
CREATE TABLE skill_sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id TEXT NOT NULL,
    section_name TEXT NOT NULL,
    heading TEXT,
    content TEXT NOT NULL,
    token_count INTEGER,
    order_index INTEGER,
    FOREIGN KEY(skill_id) REFERENCES skills(id)
);
```

---

### 8.10 `skill_stats`

```sql
CREATE TABLE skill_stats (
    skill_id TEXT PRIMARY KEY,
    total_retrieved INTEGER DEFAULT 0,
    total_selected INTEGER DEFAULT 0,
    total_read INTEGER DEFAULT 0,
    total_invoked INTEGER DEFAULT 0,
    total_success INTEGER DEFAULT 0,
    total_failure INTEGER DEFAULT 0,
    avg_latency_ms REAL,
    avg_score REAL,
    updated_at TEXT,
    FOREIGN KEY(skill_id) REFERENCES skills(id)
);
```

---

### 8.11 `search_logs`

```sql
CREATE TABLE search_logs (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    user_query TEXT NOT NULL,
    rewritten_query TEXT,
    parsed_task_json TEXT,
    top_k INTEGER NOT NULL,
    retrieved_skill_ids_json TEXT NOT NULL,
    scores_json TEXT NOT NULL,
    latency_ms INTEGER,
    created_at TEXT NOT NULL
);
```

---

### 8.14 `skill_read_logs`

```sql
CREATE TABLE skill_read_logs (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    section TEXT,
    max_tokens INTEGER,
    returned_token_count INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY(skill_id) REFERENCES skills(id)
);
```

---

### 8.15 `skill_invocation_logs`

```sql
CREATE TABLE skill_invocation_logs (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    input_json TEXT NOT NULL,
    output_json TEXT,
    error_json TEXT,
    success INTEGER NOT NULL,
    latency_ms INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY(skill_id) REFERENCES skills(id)
);
```

---

### 8.16 `agent_trajectory_logs`

```sql
CREATE TABLE agent_trajectory_logs (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    step_type TEXT NOT NULL,
    content_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

`step_type` should be one of:

```text
llm_reasoning
skill_search
skill_selection
skill_read
skill_context_loaded
skill_application
skill_invocation
skill_result
final_answer
error
```

---

## 9. Internal Tool Schema: `skill_search`

The LLM should be given the following built-in framework tools:

```text
skill_search   — always available
skill_read     — always available
skill_invoke   — available only when agent.enable_skill_invoke = true in config
```

```json
{
  "name": "skill_search",
  "description": "Search the local skill library for skills that may help solve the current task or subtask. Skills may be instructional documents, workflows, recipes, references, prompt patterns, tool usage guides, executable tools, or hybrid skills.",
  "parameters": {
    "type": "object",
    "required": ["query"],
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural language description of the capability or procedural knowledge needed."
      },
      "task_context": {
        "type": "string",
        "description": "Relevant context from the current long-horizon task."
      },
      "required_capabilities": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "Specific capabilities that the agent believes are needed."
      },
      "input_types": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "Known input types, such as pdf, image, csv, xlsx, text, url, code, or database."
      },
      "output_types": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "Desired output types, such as markdown, json, chart, docx, pptx, text, table, or file."
      },
      "top_k": {
        "type": "integer",
        "description": "Maximum number of candidate skills to return.",
        "default": 5
      }
    }
  }
}
```

---

## 10. Internal Tool Schema: `skill_read`

```json
{
  "name": "skill_read",
  "description": "Read the content of a selected local skill document. Skills may be instructional documents, workflows, recipes, references, prompt patterns, tool usage guides, executable tools, or hybrid skills.",
  "parameters": {
    "type": "object",
    "required": ["skill_id"],
    "properties": {
      "skill_id": {
        "type": "string",
        "description": "The ID of the skill returned by skill_search."
      },
      "section": {
        "type": "string",
        "description": "Optional section to read, such as overview, procedure, examples, caveats, schema, execution, or full.",
        "default": "overview"
      },
      "max_tokens": {
        "type": "integer",
        "description": "Maximum number of tokens to return.",
        "default": 2000
      }
    }
  }
}
```

Output schema:

```json
{
  "skill_id": "research.paper_claim_method_finding",
  "name": "Paper Claim-Method-Finding Workflow",
  "skill_type": "workflow",
  "interaction_mode": "read_then_apply",
  "section": "procedure",
  "content": "...",
  "available_sections": [
    "overview",
    "procedure",
    "output_format",
    "examples",
    "failure_modes"
  ],
  "token_count": 1320,
  "execution_available": false
}
```

---

## 11. Internal Tool Output Schema: `skill_search`

The search tool should return compact skill cards that support both documentation-first and executable skills.

```json
{
  "query_id": "search_001",
  "candidates": [
    {
      "skill_id": "research.paper_claim_method_finding",
      "name": "Paper Claim-Method-Finding Workflow",
      "version": "0.1.0",
      "skill_type": "workflow",
      "interaction_mode": "read_then_apply",
      "execution_available": false,
      "description_short": "A workflow for extracting claim, method, and findings from academic papers.",
      "matched_capabilities": [
        "extract_claim",
        "extract_method",
        "extract_findings"
      ],
      "available_sections": [
        "overview",
        "procedure",
        "output_format",
        "examples"
      ],
      "read_recommendation": {
        "should_read": true,
        "recommended_section": "procedure",
        "reason": "The task requires following a paper-analysis workflow."
      },
      "when_to_use": [
        "The user asks to analyze an academic paper.",
        "The task requires extracting claims, methods, or findings."
      ],
      "when_not_to_use": [
        "The input is not an academic paper."
      ],
      "score": 0.92,
      "score_breakdown": {
        "dense_description": 0.88,
        "dense_capability": 0.91,
        "bm25": 0.71,
        "example": 0.82,
        "schema_match": 0.5,
        "risk_penalty": 0.0,
        "dependency_missing_penalty": 0.0,
        "permission_penalty": 0.0
      }
    },
    {
      "skill_id": "pdf.extract_text",
      "name": "PDF Text Extractor",
      "version": "0.1.0",
      "skill_type": "tool_wrapper",
      "interaction_mode": "execute_directly",
      "execution_available": true,
      "description_short": "Extract text from PDF files while preserving page order.",
      "matched_capabilities": [
        "read_pdf",
        "extract_page_text"
      ],
      "available_sections": [
        "overview",
        "schema",
        "examples"
      ],
      "input_schema": {
        "type": "object",
        "required": ["file_path"],
        "properties": {
          "file_path": {
            "type": "string"
          }
        }
      },
      "output_schema": {
        "type": "object",
        "properties": {
          "pages": {
            "type": "array"
          }
        }
      },
      "when_to_use": [
        "The input file is a PDF.",
        "The task requires reading textual content from a PDF."
      ],
      "when_not_to_use": [
        "The PDF is image-only and requires OCR."
      ],
      "score": 0.85,
      "score_breakdown": {
        "dense_description": 0.80,
        "dense_capability": 0.84,
        "bm25": 0.65,
        "example": 0.78,
        "schema_match": 1.0,
        "risk_penalty": 0.0,
        "dependency_missing_penalty": 0.0,
        "permission_penalty": 0.0
      }
    }
  ],
  "missing_capabilities": [],
  "search_latency_ms": 37
}
```

For executable skills, `input_schema` and `output_schema` are included when available. For documentation-first skills, these fields are omitted.

`missing_capabilities` is computed as: the set of `required_capabilities` from the request that have no semantic match (cosine similarity below a threshold, e.g. 0.5) against any capability description of any retrieved candidate skill.

```text
missing = [cap for cap in request.required_capabilities
           if max(cosine_sim(embed(cap), embed(skill_cap.description))
                  for skill in candidates for skill_cap in skill.capabilities) < threshold]
```

---

## 12. Internal Tool Schema: `skill_invoke` (Optional)

> **Important**: `skill_invoke` is only available when `agent.enable_skill_invoke = true` in the config. It should only be used for skills that declare `execution_available = true` or `execution.mode != none`. It should not be used for instructional, workflow, reference, prompt-pattern, or code-recipe skills unless they explicitly define an executable interface. The LLM should not assume a skill is executable merely because it was retrieved.

```json
{
  "name": "skill_invoke",
  "description": "Invoke a selected local skill with validated arguments. Only valid for skills with execution_available = true.",
  "parameters": {
    "type": "object",
    "required": ["skill_id", "arguments"],
    "properties": {
      "skill_id": {
        "type": "string",
        "description": "The ID of the skill to invoke."
      },
      "arguments": {
        "type": "object",
        "description": "Arguments matching the selected skill input schema."
      },
      "reason": {
        "type": "string",
        "description": "Why this skill is needed for the current task."
      }
    }
  }
}
```

## 13. Internal Tool Output Schema: `skill_invoke`

```json
{
  "skill_id": "pdf.extract_text",
  "success": true,
  "output": {
    "pages": [
      {
        "page_number": 1,
        "text": "..."
      }
    ]
  },
  "error": null,
  "latency_ms": 513
}
```

On failure:

```json
{
  "skill_id": "pdf.extract_text",
  "success": false,
  "output": null,
  "error": {
    "type": "ValidationError",
    "message": "Missing required field: file_path",
    "recoverable": true
  },
  "latency_ms": 5
}
```

---

## 14. Hybrid Search Design

The search engine should implement multi-view retrieval.

### 14.1 Search Inputs

The search engine receives:

```python
class SkillSearchRequest(BaseModel):
    query: str
    task_context: str | None = None
    required_capabilities: list[str] = []
    input_types: list[str] = []
    output_types: list[str] = []
    top_k: int = 5
```

---

### 14.2 Skill Text Views

For each skill, build multiple searchable views:

```text
description_view:
  name + short description + long description

capability_view:
  capability IDs + capability descriptions

example_view:
  positive examples + negative examples

schema_view:
  input schema + output schema + input/output types (optional, only if available)

usage_rule_view:
  when_to_use + when_not_to_use

content_section_view:
  skill document section headings + overview text (if available)
```

Each view should have a dense embedding representation for multi-view dense retrieval.

For BM25, build a single index per skill over the concatenation of all views, including content sections when available. This keeps the sparse retrieval simple while the dense retrieval handles view-level granularity.

---

### 14.3 Retrieval Pipeline

The retrieval pipeline has two stages: candidate generation via RRF, then re-scoring via the weighted formula.

**Stage 1: Candidate Generation**

```text
1. Normalize query.
2. Parse task context.
3. Generate expanded search query (concatenate query + required_capabilities + input_types).
4. Apply metadata filters (status = active, exclude unavailable dependencies).
5. Run BM25 retrieval over the single concatenated-text index → top recall_k results.
6. Run dense vector retrieval over each view (description, capability, example, schema, usage_rule, content_section) → top recall_k results per view.
7. Collect all rank lists (1 BM25 list + N dense view lists) and fuse into a single candidate pool using reciprocal rank fusion (§14.5).
8. Take the top recall_k candidates from the fused list.
```

**Stage 2: Re-Scoring**

```text
9.  For each candidate, compute the weighted re-score using the formula in §14.4.
10. Rank candidates by final score.
11. Return at most top_k compact skill cards with score >= minimum_score_threshold (default: 0.1).
```

---

### 14.4 Score Formula

Use this initial score formula:

```text
score(skill, request) =
    0.30 * dense_description_score
  + 0.25 * dense_capability_score
  + 0.15 * bm25_score
  + 0.15 * example_score
  + 0.15 * schema_match_score
  - 0.20 * risk_penalty
  - 0.30 * dependency_missing_penalty
  - 0.20 * permission_penalty
```

The positive weights sum to 1.0. Penalties are subtracted and can push the score below zero; candidates below `minimum_score_threshold` (default: 0.1) are discarded.

The weights should be configurable in `configs/default.yaml`. The weights should be improved upon experiment.

#### Component Definitions

All component scores must be normalized to [0, 1] before applying the weighted sum.

**`dense_description_score`**: Cosine similarity between the embedded query and the skill's description_view embedding. Normalized via `(cos_sim + 1) / 2` to map from [-1, 1] to [0, 1].

**`dense_capability_score`**: Cosine similarity between the embedded query and the skill's capability_view embedding. Same normalization as above.

**`bm25_score`**: BM25 score of the query against the skill's concatenated text. Normalize by dividing by the maximum BM25 score in the current result set (max-normalization). If only one candidate, set to 1.0.

**`example_score`**: Maximum cosine similarity between the embedded query and each of the skill's positive example `user_query` embeddings. If the skill has no positive examples, default to 0.0.

**`schema_match_score`**: Set overlap between the request's `input_types`/`output_types` and the skill's declared `input_types`/`output_types`. Computed as: `(|request_input ∩ skill_input| + |request_output ∩ skill_output|) / (|request_input| + |request_output|)`. If the request specifies no types, default to 0.5 (neutral).

**`risk_penalty`**: 0.0 if `risk.level == "low"`, 0.5 if `"medium"`, 1.0 if `"high"`.

**`dependency_missing_penalty`**: Fraction of the skill's declared Python/system dependencies that are not installed or available. 0.0 if all present, 1.0 if all missing.

**`permission_penalty`**: 1.0 if the skill requires permissions (filesystem_write, network, shell) that the current agent policy disallows. 0.0 otherwise.

---

### 14.5 Reciprocal Rank Fusion

RRF is used in **Stage 1 only** to merge the BM25 rank list and the per-view dense rank lists into a single candidate pool. It does **not** produce the final score — the weighted formula in §14.4 does that in Stage 2.

Implement RRF:

```python
def rrf_fusion(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores = {}
    for rank_list in rank_lists:
        for rank, skill_id in enumerate(rank_list, start=1):
            scores[skill_id] = scores.get(skill_id, 0.0) + 1.0 / (k + rank)
    return scores
```

The input `rank_lists` contains one list from BM25 retrieval and one list per dense view (description, capability, example, schema, usage_rule, content_section), for a total of 7 rank lists.

---

## 15. LLM Agent Loop

The LLM should receive only the built-in tools initially:

```text
skill_search   — always available
skill_read     — always available
skill_invoke   — available only when agent.enable_skill_invoke = true in config
```

It should not receive all local skill descriptions upfront.

### 15.1 Agent System Prompt

The system prompt should include:

```text
You are a long-horizon task-solving agent.

You have access to a local skill library, but you do not know all available skills upfront.

Skills are not necessarily executable tools.
A skill may be an instructional document, workflow, recipe, reference note,
prompt pattern, tool usage guide, executable tool wrapper, or hybrid skill.

When you need specialized capability or procedural knowledge:
1. call skill_search to find relevant local skills;
2. inspect the returned compact skill cards;
3. call skill_read to read the most relevant skill document or section;
4. apply the skill instructions in your reasoning;
5. only call skill_invoke if the skill explicitly declares execution_available = true.

Do not invent skill IDs.
Do not assume a skill is executable unless its metadata says so.
Prefer reading a skill before applying it.
If no retrieved skill is suitable, search again with a better capability query.
Select and read the minimal set of skills necessary for the task.
```

---

### 15.2 Runtime Loop

Implement:

```python
# Build tool list based on config
tools = [skill_search_spec, skill_read_spec]
if config.agent.enable_skill_invoke:
    tools.append(skill_invoke_spec)

while not done:
    response = llm.complete(messages, tools=tools)

    if response.tool_call == "skill_search":
        result = skill_search_engine.search(...)
        messages.append(tool_result(result))
        log_search(result)

    elif response.tool_call == "skill_read":
        result = skill_reader.read(...)
        skill_context = context_builder.build(result)
        messages.append(tool_result(result))
        messages.append(skill_context)
        log_skill_read(result)

    elif response.tool_call == "skill_invoke":
        result = skill_executor.invoke(...)
        messages.append(tool_result(result))
        log_skill_invocation(result)

    else:
        messages.append(response)
        if is_final(response):
            done = True
```

Note: when `agent.enable_skill_invoke = false`, `skill_invoke_spec` is omitted from the tool list entirely, so the LLM cannot call it.

---

## 16. Skill Executor Design (Optional)

The executor should:

1. Validate `skill_id`.
2. Load skill metadata.
3. Validate input arguments against `input_schema`.
4. Check permissions.
5. Check dependencies.
6. Execute the skill.
7. Validate output against `output_schema`.
8. Return structured result.
9. Log the invocation.

Execution modes:

```text
python_function
subprocess
http_local
mock
```

For MVP, implement only:

```text
python_function
mock
```

---

## 17. Safety and Reliability Rules

The framework should enforce the following:

1. The LLM cannot invoke arbitrary Python functions.
2. The LLM can only invoke registered skills, and only when `agent.enable_skill_invoke = true`.
3. Every skill must declare permissions (applicable to executable skills).
4. The executor must reject skill calls violating permissions.
5. Every skill input must be validated (for executable skills).
6. Every skill output should be validated when possible (for executable skills).
7. Errors should be returned as structured recoverable or unrecoverable errors.
8. The framework should log every discovery, reading, and optional invocation event.
9. Skill reading should respect max_tokens limits to avoid context overload.

---

## 18. MVP Implementation Milestones

### Milestone 1: Skill Schema and Loader

Implement:

```text
Pydantic SkillSpec schema
  including skill_type, interaction, content metadata, optional execution metadata
YAML skill loader
skill.md document loader
SQLite skill registry
  including skill_documents and skill_sections tables
basic validation
unit tests
```

Acceptance criteria:

```text
- Can load all skill.yaml files from data/skills.
- Can load skill.md documents and parse into sections.
- Invalid skill.yaml files produce clear validation errors.
- Skills are inserted into SQLite with skill_type, interaction_mode, and content metadata.
- Documentation-first skills with execution.mode = none load correctly.
- Executable skills with python_function metadata load correctly.
```

---

### Milestone 2: Index Builder

Implement:

```text
multi-view text generation
  including content_section_view from skill.md
BM25 index over all views including content sections
FAISS dense index per view
id mapping
index persistence
```

Acceptance criteria:

```text
- Can build indexes from local skill library.
- Can reload indexes from disk.
- Each skill has description, capability, example, schema (if available), usage-rule, and content-section views.
- Documentation-first skills without schemas are indexed correctly.
```

---

### Milestone 3: Hybrid Search Engine

Implement:

```text
SkillSearchRequest
BM25 retrieval
dense retrieval
RRF fusion
schema matching (optional, only when schema exists)
capability matching
risk/dependency filtering
top-k skill card output with skill_type, interaction_mode, execution_available, and read_recommendation
```

Acceptance criteria:

```text
- Given a query, returns ranked skill candidates.
- Output follows the updated skill_search response schema.
- Documentation-first and executable skills are both returned correctly.
- Search logs are stored in SQLite.
```

---

### Milestone 4: Skill Reader and Context Builder

Implement:

```text
skill_read tool interface
markdown skill document loader
section parser
token budget handling
context block construction
skill_read_logs
unit tests
```

Acceptance criteria:

```text
- Agent can read a selected skill document.
- Agent can read a selected section.
- Reader enforces max_tokens.
- Context Builder creates a usable skill context block.
- Read events are logged.
```

---

### Milestone 5: LLM Agent Loop

Implement:

```text
agent loop
tool specs for skill_search and skill_read
skill_search tool bridge
skill_read tool bridge
optional skill_invoke tool bridge (controlled by config flag)
trajectory logging
```

Acceptance criteria:

```text
- Agent can solve a task requiring skill reading and application.
- Agent can search, read, and apply skill instructions.
- Agent does not see all skill descriptions upfront.
- When agent.enable_skill_invoke = false, skill_invoke is not available.
```

---

### Milestone 6: Evaluation Pipeline

Implement:

```text
ToolRet-style retrieval evaluation
read-selection evaluation
skill-augmented task performance evaluation
synthetic skill benchmark
all-skill prompt baseline
dynamic retrieval + read baseline
metrics
result reports
```

Acceptance criteria:

```text
- Can run retrieval evaluation.
- Can run read-selection evaluation.
- Can compute Recall@k, MRR, nDCG@k, Correct Skill Read@k.
- Can compare BM25, dense, hybrid, and hybrid+rerank.
```

---

### Milestone 7: Optional Executable Skill Support

Implement:

```text
skill_invoke interface
input validation
permission checks
python_function execution
mock execution
output validation
invocation logs
```

Acceptance criteria:

```text
- LLM or test code can invoke a registered executable skill by ID.
- Invalid input is rejected.
- Unknown skill ID is rejected.
- Non-executable skills cannot be invoked.
- Execution result follows unified schema.
```

---

## 19. Benchmark and Validation Plan

Use a staged validation strategy.

---

## 19.1 Stage A: Retrieval-Only Evaluation

Primary benchmark:

```text
ToolRet-style tool retrieval benchmark
```

Purpose:

```text
Evaluate whether the local hybrid search engine can retrieve the correct skill candidates from a large skill library.
```

If directly adapting ToolRet:

```text
ToolRet tool corpus → local skill specs
ToolRet queries → skill_search requests
Gold tools → gold skills
```

Required adapter:

```python
class ToolRetAdapter:
    def convert_tool_to_skill(self, tool_doc: dict) -> SkillSpec:
        ...

    def convert_query_to_search_request(self, sample: dict) -> SkillSearchRequest:
        ...

    def get_gold_skill_ids(self, sample: dict) -> list[str]:
        ...
```

Metrics:

```text
Recall@1
Recall@3
Recall@5
Recall@10
MRR@10
nDCG@10
Precision@5
Average search latency
```

Baselines:

```text
BM25 only
Dense description only
Dense capability only
Dense multi-view
Hybrid BM25 + dense
Hybrid + schema matching
Hybrid + reranker
Random retrieval
```

Expected result:

```text
Hybrid retrieval should outperform BM25-only and dense-only retrieval on Recall@5 and MRR@10.
```

---

## 19.2 Stage B: Skill Read Selection Evaluation

Purpose:

```text
Evaluate whether the LLM chooses to read the correct skill documents after receiving search results.
```

Metrics:

```text
Correct Skill Read@1
Correct Skill Read@3
Skill Read Precision
Skill Read Recall
Unnecessary Read Rate
Missed Read Rate
Average Read Tokens
```

Baselines:

```text
No skill read
Random skill read
BM25 retrieved skill read
Dense retrieved skill read
Hybrid retrieved skill read
Oracle skill read
All skill documents loaded
```

Expected result:

```text
LLM with hybrid retrieval + skill_read should read the correct skill documents more accurately than random or BM25-only baselines.
```

---

## 19.3 Stage C: Skill-Augmented Task Performance

Purpose:

```text
Evaluate whether reading retrieved skill documents improves task performance.
```

Methods:

```text
No skill
All skills loaded
Retrieved compact skill cards only
Retrieved + skill_read
Oracle skill_read
```

Metrics:

```text
Task Success Rate
Output Quality Score
Instruction Following Score
Workflow Completeness
Average Prompt Tokens
Average Read Tokens
Average Latency
Oracle Read Gap
```

Define `Oracle Read Gap` as:

```text
TaskSuccess(oracle_skill_read) - TaskSuccess(retrieved_skill_read)
```

This helps distinguish retrieval failure from skill application failure.

Expected result:

```text
Retrieved + skill_read should outperform no-skill and compact-cards-only baselines on task success rate.
```

---

## 19.4 Stage D: Optional Executable Tool Evaluation

> Only use this stage when the benchmark defines callable tools and `agent.enable_skill_invoke = true`.

Purpose:

```text
Evaluate whether dynamic skill search helps complete tasks requiring executable tool invocation.
```

Metrics:

```text
Correct Skill Invocation Rate
Wrong Skill Invocation Rate
Argument Accuracy
Execution Success Rate
Task Success Rate
Average Prompt Tokens
Average Number of Search Calls
Average Number of Skill Read Calls
Average Number of Skill Invocation Calls
Average Latency
Failure Recovery Rate
```

---

## 19.5 Stage E: Context Pollution Scaling Experiment

Purpose:

```text
Directly test whether all-skill prompt loading degrades as the number of skills increases.
```

Setup:

```text
Create skill pools of different sizes:
  10 skills
  50 skills
  100 skills
  500 skills
  1000 skills
```

For each pool:

```text
Keep the same gold skills.
Add distractor skills with overlapping descriptions.
Run the same task set.
Compare all-skill prompt vs dynamic skill search.
```

Metrics:

```text
Skill Read Accuracy
Average Read Tokens
Total Context Tokens
Wrong Skill Read Rate
Unnecessary Skill Read Rate
Task Success Rate
Prompt Tokens
Wrong Skill Rate
Hallucinated Skill Rate
Latency
```

Expected result:

```text
All-skill prompt should degrade as skill pool size increases.
Dynamic skill search + read should be more stable.
```

---

## 20. Evaluation Output Format

Each evaluation run should produce:

```text
results.jsonl
summary.csv
summary.md
plots/
```

Example `results.jsonl` record:

```json
{
  "run_id": "eval_001",
  "task_id": "task_001",
  "method": "hybrid_read_top5",
  "user_query": "Read this paper and summarize its claim, method, and findings.",
  "gold_skills": [
    "research.paper_claim_method_finding"
  ],
  "retrieved_skills": [
    "research.paper_claim_method_finding",
    "pdf.extract_text",
    "text.summarize"
  ],
  "read_skills": [
    "research.paper_claim_method_finding"
  ],
  "applied_skills": [
    "research.paper_claim_method_finding"
  ],
  "invoked_skills": [],
  "success": true,
  "prompt_tokens": 2800,
  "read_tokens": 1320,
  "latency_ms": 4200,
  "num_search_calls": 1,
  "num_skill_read_calls": 1,
  "num_skill_invocation_calls": 0,
  "error": null
}
```
```

---

## 21. Metrics Implementation

Implement:

```python
def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """Fraction of gold skills found in the top-k retrieved results."""
    if not gold:
        return 1.0
    return len(set(retrieved[:k]) & set(gold)) / len(set(gold))


def precision_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    return len(set(retrieved[:k]) & set(gold)) / k


def mrr_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    gold_set = set(gold)
    for i, item in enumerate(retrieved[:k], start=1):
        if item in gold_set:
            return 1.0 / i
    return 0.0
```

Also implement:

```text
nDCG@k
Correct Skill Read@k
Skill Read Precision
Skill Read Recall
Over-selection Rate
Missing Skill Rate
Oracle Read Gap
Task Success Rate
Average Prompt Tokens
Average Read Tokens
Average Latency
```

---

## 22. Codex Implementation Instructions

Please implement this project in the following order:

1. Create project structure.
2. Define Pydantic schemas:
   - `SkillSpec` (with skill_type, interaction, content, optional execution)
   - `SkillSearchRequest`
   - `SkillSearchResponse`
   - `SkillReadRequest`
   - `SkillReadResponse`
   - `SkillInvocationRequest` (optional)
   - `SkillInvocationResponse` (optional)
   - log schemas
3. Implement YAML skill loader.
4. Implement skill.md document loader and section parser.
5. Implement SQLite schema and registry (including skill_documents, skill_sections).
6. Implement multi-view text generation (including content_section_view).
7. Implement BM25 index.
8. Implement FAISS vector index.
9. Implement hybrid search with RRF.
10. Implement score breakdown.
11. Implement `skill_search` tool interface.
12. Implement `skill_read` tool interface.
13. Implement skill reader and context builder.
14. Implement mock skills for testing.
15. Implement agent loop with skill_search and skill_read.
16. Implement evaluation metrics.
17. Implement ToolRet-style adapter interface.
18. Implement synthetic benchmark generator.
19. Add tests.
20. Add CLI commands.
21. Add documentation.
22. (Optional) Implement `skill_invoke` tool interface.
23. (Optional) Implement basic Python function skill executor.

---

## 23. CLI Commands

Provide these CLI commands:

```text
skill-agent validate-skills --skill-dir data/skills

skill-agent build-index --skill-dir data/skills --index-dir data/indexes

skill-agent search "extract text from a pdf" --top-k 5

skill-agent read research.paper_claim_method_finding --section procedure --max-tokens 2000

skill-agent invoke pdf.extract_text --args '{"file_path": "sample.pdf"}'  # only when enable_skill_invoke = true

skill-agent run-task --task-file data/eval/synthetic_tasks/task_001.json

skill-agent eval-retrieval --dataset data/eval/toolret --method hybrid

skill-agent eval-read-selection --dataset data/eval/synthetic_tasks --method hybrid

skill-agent eval-skill-augmented --dataset data/eval/synthetic_tasks --method hybrid-read

skill-agent eval-agent --dataset data/eval/synthetic_tasks --method hybrid

skill-agent report --result-file data/eval/results/results.jsonl
```

---

## 24. Configuration File

Example `configs/default.yaml`:

```yaml
database:
  url: sqlite:///data/skills.db

skill_library:
  path: data/skills

indexes:
  path: data/indexes
  dense:
    enabled: true
    backend: faiss
    embedding_model: BAAI/bge-small-en-v1.5
  sparse:
    enabled: true
    backend: bm25

retrieval:
  top_k: 5
  recall_k: 50
  rrf_k: 60
  minimum_score_threshold: 0.1
  weights:
    dense_description: 0.30
    dense_capability: 0.25
    bm25: 0.15
    example: 0.15
    schema_match: 0.15
    risk_penalty: 0.20
    dependency_missing_penalty: 0.30
    permission_penalty: 0.20

skill_reader:
  default_section: overview
  default_max_tokens: 2000

agent:
  max_steps: 20
  max_search_calls: 5
  max_skill_read_calls: 10
  max_skill_invoke_calls: 10
  enable_skill_invoke: false

logging:
  path: logs/runs
  save_trajectory: true
```

---

## 25. Unit Tests

Minimum tests:

```text
test_skill_schema.py
  - valid skill passes
  - missing required fields fail
  - invalid interaction mode fails

test_skill_document_schema.py
  - workflow skill with execution.mode = none passes
  - executable skill with python_function metadata passes
  - invalid interaction mode fails

test_indexing.py
  - skill views are generated (including content_section_view)
  - BM25 index builds
  - FAISS index builds
  - indexes reload correctly

test_hybrid_search.py
  - search returns expected skill
  - RRF fusion works
  - unavailable skill is penalized
  - schema mismatch lowers score
  - documentation-first skills are returned correctly

test_skill_read.py
  - read full skill document
  - read specific section
  - missing section returns clear error
  - max_tokens is enforced

test_context_builder.py
  - builds valid skill context block
  - includes metadata and section content
  - avoids loading unrelated sections

test_skill_invocation.py (optional, for executable skills)
  - valid skill call succeeds
  - invalid skill ID fails
  - invalid arguments fail
  - non-executable skill invocation is rejected
  - output validation works

test_eval_metrics.py
  - Recall@k
  - Precision@k
  - MRR@k
  - nDCG@k
  - Correct Skill Read@k
```

---

## 26. Success Criteria for MVP

The MVP is successful if:

```text
1. Skills can be defined locally using YAML.
2. Skills can be documentation-first, executable, or hybrid.
3. Skill metadata and document sections can be loaded into SQLite.
4. BM25 and dense indexes can be built over metadata and skill document sections.
5. The agent initially sees only `skill_search` and `skill_read`; `skill_invoke` is optional (controlled by config).
6. The agent can search for skills during reasoning.
7. The search engine returns relevant compact skill cards.
8. The agent can read selected skill documents or sections.
9. The Skill Context Builder can load relevant skill content into the conversation.
10. Retrieval and read-selection evaluation can produce Recall@k, MRR, and Correct Skill Read@k.
11. Dynamic skill search + read can be compared against all-skill prompt loading.
12. Results are logged in a reproducible format.
```

---

## 27. Initial Local Skills for MVP

Create at least the following skills (mix of documentation-first and executable):

```text
Documentation-first / workflow skills:
  research.paper_claim_method_finding
  writing.academic_abstract
  analysis.data_exploration_workflow

Executable / tool wrapper skills:
  pdf.extract_text
  csv.read
  dataframe.describe
  chart.generate
  text.summarize
  markdown.format
  json.write
  file.read_text
  file.write_text
  image.ocr_mock
```

For fast development, some skills can be mock skills.

Example mock executable skill:

```python
def summarize_text(text: str, max_words: int = 200) -> dict:
    words = text.split()
    return {
        "summary": " ".join(words[:max_words])
    }
```

---

## 28. Important Design Constraints

Do not implement the system as a simple classifier from query to skill ID.

Do not assume every skill is executable.

Do not force documentation-first skills into function-call schemas.

Skill reading is the primary interaction mode. Skill invocation is optional and only for skills with explicit executable interfaces.

The framework should support procedural knowledge, workflows, references, recipes, and tool wrappers under one unified skill registry.

The search engine must return:

```text
candidate skills
matched capabilities
score breakdown
skill_type and interaction_mode
available_sections and read_recommendation
input/output schema (only if available)
usage constraints
```

The LLM should make the final selection and reading decisions based on retrieved candidates.

The framework should support future extension to:

```text
skill graph planning
multi-skill composition
learned reranking
skill usage feedback
skill marketplace
MCP tool import
```

---

## 29. Future Extensions

After MVP:

```text
1. Add LLM query rewriting before retrieval.
2. Add cross-encoder reranker.
3. Add skill graph planning.
4. Add automatic skill composition.
5. Add feedback-based ranking using historical success.
6. Add MCP server importer.
7. Add sandboxed subprocess skills.
8. Add UI for inspecting skill search and read results.
9. Add experiment dashboard.
10. Add long-horizon benchmark based on real local files.
11. Add skill document summarization for context compression.
12. Add adaptive section selection based on task complexity.
```

---

## 30. Final Expected Outcome

The final framework should demonstrate that:

```text
Dynamic local skill search and skill reading can reduce context pollution,
scale better than all-skill prompt loading,
and allow long-horizon LLM agents to discover, inspect, and apply
local procedural knowledge on demand.
```

Executable skill invocation is supported as an optional extension, but the core contribution is:

```text
Capability-aware skill discovery and context-efficient skill loading
for long-horizon LLM agents.
```