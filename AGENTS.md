# AGENTS.md

Guidance for agents working in this repository.

## Project Identity

This repository is a local skill search and SRA-Bench experimentation workspace.
The main project implements SkillBrowser: a SkillSpec-based skill searcher with
BM25, dense, and hybrid retrieval. The upstream benchmark lives in the
`benchmarks/SR-Agents` submodule and should be treated as the source of truth
for SRA-Bench prompts, providers, inference engines, and evaluators.

## Important Tech Stack

- Main project runtime: `uv`, Python project in `pyproject.toml`, source under
  `src/`.
- Main configuration setting in `config.toml`.

Always set UTF-8 explicitly for benchmark runs:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = (Resolve-Path src).Path
$env:UV_CACHE_DIR = (Resolve-Path .uv-cache).Path
```

## SRA-Bench Experiment Conventions

The standard comparison uses three methods per selected dataset:

1. `skillbrowser_hybrid_top5_direct`
   - SkillBrowser hybrid retrieval over the full processed skill corpus from
     `config.toml` `[sra].skill_dirs`.
   - Retrieval mode: `hybrid`.
   - Top-k: `5`.
   - Inference provider: SR-Agents `topk`, `k=5`.
   - Engine: `direct`, unless SR-Agents normally requires another engine.

2. `oracle_direct`
   - Native SR-Agents `oracle` provider.
   - Same model and dataset-native prompt.
   - Engine: `direct`, unless SR-Agents normally requires another engine.

3. `sragents_bm25_top5_direct`
   - Native SR-Agents BM25 retrieval.
   - Top-k: `5`.
   - Inference provider: SR-Agents `topk`, `k=5`.
   - Engine: `direct`, unless SR-Agents normally requires another engine.

Use `direct` for:

- `theoremqa`
- `logicbench`
- `champ`
- `medcalcbench`

Be cautious with:

- `toolqa`: SR-Agents normally uses `react`; it can execute model-selected
  ToolQA actions, including `PythonInterpreter`.
- `bigcodebench`: evaluation executes model-generated Python code. Treat this
  as untrusted code execution and get explicit approval before running.

## Full-Run Concurrency

For full runs requested in this workspace, use 5 concurrent workers for
SR-Agents inference/evaluation unless the user says otherwise:

```powershell
--workers 5
--eval-workers 5
```

For staged SkillBrowser experiments, prefer:

1. `scripts/sra_bench.py retrieve`
2. `scripts/sra_bench.py infer --provider topk --provider-k 5 --workers 5`
3. `scripts/sra_bench.py evaluate --eval-workers 5`

This preserves SkillBrowser retrieval while letting SR-Agents inference use the
requested worker count.

## Output Organization

Raw benchmark outputs live under:

- `data/eval/sra/results/retrieval/`
- `data/eval/sra/results/inference/`
- `data/eval/sra/results/eval/`
- `data/eval/sra/results/e2e/`

For organized model-specific result packages, place copies under:

- `data/eval/sra/results/retrieval/<model-name>/`
- `data/eval/sra/results/inference/<model-name>/`
- `data/eval/sra/results/eval/<model-name>/`

Names of the result files: 

- Eval: `<subset>-<method>.json`
- Inference: `<subset>-<method>.jsonl`
- Retrieval: `<subset>-<retrieval_method>.json`
- Retrieval checkpoints:
  `<subset>-<retrieval_method>.checkpoint.jsonl`

Current organized method names:

- `skillbrowser_hybrid_top5_direct`
- `oracle_direct`
- `sragents_bm25_top5_direct`
- Retrieval-only names:
  - `skillbrowser_hybrid_top5`
  - `sragents_bm25_top5`

## Result Reporting

When reporting SRA-Bench results, include:

- Dataset/subset.
- Method/prompt type.
- Number of instances.
- Retrieval metrics when applicable: at least Recall@1, Recall@5, nDCG@1,
  nDCG@5.
- Accuracy, correct, total.
- Runtime or notable runtime events.
- Output paths.
- Failures, skipped runs, resumed runs, or killed/stuck jobs.
