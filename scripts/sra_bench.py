"""Run SRA-Bench with this project's Hybrid skill searcher.

This script keeps SR-Agents as a git submodule while producing retrieval files
that its native infer/evaluate stages can consume.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.sra import (  # noqa: E402
    SRA_CORPUS_PATH,
    SRA_INSTANCES_DIR,
    SRA_RESULTS_DIR,
    SRA_SUBMODULE_DIR,
    ensure_sra_corpus,
    load_sra_corpus,
    run_sra_retrieval,
    sra_corpus_to_specs,
)
from benchmarks.sra_agent import (  # noqa: E402
    run_sra_agent_inference,
    write_sra_agent_summary,
)
from benchmarks.sra_preprocess import (  # noqa: E402
    DEFAULT_SRA_PREPROCESS_CHECKPOINT,
    DEFAULT_SRA_PREPROCESS_MODEL,
    DEFAULT_SRA_SKILL_DIR,
    run_sra_preprocess,
)
from cli import add_retrieval_arguments, build_searcher  # noqa: E402
from config import load_app_config  # noqa: E402
from loader import load_skills  # noqa: E402
from llm import MockLLMClient, OpenAICompatibleLLMClient  # noqa: E402


SRA_METADATA_DENSE_VIEW_NAMES = {"description", "capability", "usage", "examples", "schema"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sra-bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Prepare the SR-Agents submodule data")
    prepare.add_argument("--corpus", default=str(SRA_CORPUS_PATH))

    preprocess = sub.add_parser("preprocess", help="Preprocess SRA corpus into local SkillSpec files")
    preprocess.add_argument("--corpus", default=str(SRA_CORPUS_PATH))
    preprocess.add_argument("--output-skill-dir", default=str(DEFAULT_SRA_SKILL_DIR))
    preprocess.add_argument("--checkpoint", default=str(DEFAULT_SRA_PREPROCESS_CHECKPOINT))
    preprocess.add_argument("--model", default=DEFAULT_SRA_PREPROCESS_MODEL)
    preprocess.add_argument("--llm", choices=["mock", "openai-compatible"], default="openai-compatible")
    preprocess.add_argument("--config", default="config.toml")
    preprocess.add_argument("--api-base", help="OpenAI-compatible endpoint override")
    preprocess.add_argument("--temperature", type=float, default=0.0)
    preprocess.add_argument("--max-tokens", type=int, default=1200)
    preprocess.add_argument("--limit", type=int)
    preprocess.add_argument("--resume", action="store_true")
    preprocess.add_argument("--force", action="store_true")

    retrieve = sub.add_parser("retrieve", help="Run SkillBrowser Hybrid retrieval on SRA-Bench")
    add_common_retrieval_args(retrieve)

    infer = sub.add_parser("infer", help="Run SR-Agents inference using a Hybrid retrieval file")
    add_common_infer_args(infer)

    infer_agent = sub.add_parser("infer-agent", help="Run the local SRA general-purpose skill_search agent")
    add_common_agent_infer_args(infer_agent)

    evaluate = sub.add_parser("evaluate", help="Score an SR-Agents inference JSONL file")
    add_common_eval_args(evaluate)

    summarize_agent = sub.add_parser("summarize-agent", help="Merge task accuracy with SRA agent skill metrics")
    add_common_agent_summary_args(summarize_agent)

    run = sub.add_parser("run", help="Run prepare, retrieve, infer, and evaluate")
    add_common_retrieval_args(run)
    add_common_infer_args(run, include_dataset=False, include_retrieval_output=False)
    add_common_eval_args(run, include_dataset=False, include_input=False)

    run_agent = sub.add_parser("run-agent", help="Run local SRA agent inference, evaluation, and summary")
    add_common_agent_infer_args(run_agent)
    add_common_eval_args(run_agent, include_dataset=False, include_input=False)
    return parser


def add_common_retrieval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="theoremqa", help="SRA-Bench dataset name")
    parser.add_argument("--corpus", default=str(SRA_CORPUS_PATH), help="Path to SRA corpus.json")
    parser.add_argument("--sra-skill-dir", help="Optional preprocessed SRA SkillSpec directory")
    parser.add_argument("--instances", help="Path to SRA instances JSON")
    parser.add_argument("--retrieval-output", help="Path for SR-Agents-compatible retrieval JSON")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--limit", type=int, help="Optional query limit for smoke tests")
    parser.add_argument("--config", default="config.toml")
    add_retrieval_arguments(parser, include_config=False)


def add_common_infer_args(
    parser: argparse.ArgumentParser,
    *,
    include_dataset: bool = True,
    include_retrieval_output: bool = True,
) -> None:
    if include_dataset:
        parser.add_argument("--dataset", default="theoremqa")
    if include_retrieval_output:
        parser.add_argument("--retrieval-output", help="Path to retrieval JSON from the retrieve stage")
    parser.add_argument("--inference-output", help="Path for SR-Agents inference JSONL")
    parser.add_argument("--provider-k", type=int, default=1, help="How many retrieved skills to expose")
    parser.add_argument("--engine", default="direct", help="SR-Agents inference engine")
    parser.add_argument("--model", required=True, help="Model identifier passed to SR-Agents")
    parser.add_argument("--api-base", help="OpenAI-compatible endpoint")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--force", action="store_true")


def add_common_eval_args(
    parser: argparse.ArgumentParser,
    *,
    include_dataset: bool = True,
    include_input: bool = True,
) -> None:
    if include_dataset:
        parser.add_argument("--dataset", default="theoremqa")
    if include_input:
        parser.add_argument("--inference-output", required=True, help="Path to inference JSONL")
    parser.add_argument("--eval-output", help="Path for SR-Agents eval JSON")
    parser.add_argument("--eval-workers", type=int, default=8)
    parser.add_argument("--eval-force", action="store_true")


def add_common_agent_infer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="theoremqa", help="SRA-Bench dataset name")
    parser.add_argument("--corpus", default=str(SRA_CORPUS_PATH), help="Path to SRA corpus.json")
    parser.add_argument("--sra-skill-dir", help="Optional preprocessed SRA SkillSpec directory")
    parser.add_argument("--instances", help="Path to SRA instances JSON")
    parser.add_argument("--inference-output", help="Path for local agent inference JSONL")
    parser.add_argument("--agent-top-k", type=int, default=5, help="Number of search candidates returned to the agent")
    parser.add_argument("--max-rounds", type=int, default=6, help="Maximum LLM/tool rounds per instance")
    parser.add_argument("--limit", type=int, help="Optional instance limit for smoke tests")
    parser.add_argument("--inject-full-search-skill", action="store_true")
    parser.add_argument("--llm", choices=["mock", "openai-compatible"], default="openai-compatible")
    parser.add_argument("--model", required=True, help="Model identifier recorded in inference output")
    parser.add_argument("--api-base", help="OpenAI-compatible endpoint override")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=1, help="Reserved for future parallel local agent runs")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config", default="config.toml")
    add_retrieval_arguments(parser, include_config=False)


def add_common_agent_summary_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="theoremqa")
    parser.add_argument("--instances", help="Path to SRA instances JSON")
    parser.add_argument("--inference-output", required=True, help="Path to local agent inference JSONL")
    parser.add_argument("--eval-output", help="Optional SR-Agents eval JSON to merge")
    parser.add_argument("--summary-output", help="Path for merged local agent summary JSON")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        path = ensure_sra_corpus(args.corpus)
        print(json.dumps({"ok": True, "corpus": str(path)}, indent=2))
        return 0
    if args.command == "preprocess":
        preprocess(args)
        return 0
    if args.command == "retrieve":
        retrieve(args)
        return 0
    if args.command == "infer":
        infer(args)
        return 0
    if args.command == "infer-agent":
        infer_agent(args)
        return 0
    if args.command == "evaluate":
        evaluate(args)
        return 0
    if args.command == "summarize-agent":
        summarize_agent(args)
        return 0
    if args.command == "run":
        ensure_sra_corpus(args.corpus)
        retrieve(args)
        infer(args)
        evaluate(args)
        return 0
    if args.command == "run-agent":
        ensure_sra_corpus(args.corpus)
        infer_agent(args)
        evaluate(args)
        summarize_agent(args)
        return 0
    raise ValueError(f"Unhandled command: {args.command}")


def retrieve(args: argparse.Namespace) -> Path:
    corpus = load_sra_corpus(args.corpus)
    skills = load_sra_search_specs(args, corpus)
    searcher = build_sra_searcher(skills, args)
    instances = Path(args.instances) if args.instances else default_instances(args.dataset)
    output = Path(args.retrieval_output) if args.retrieval_output else default_retrieval_output(args.dataset)
    result = run_sra_retrieval(
        searcher=searcher,
        corpus_size=len(skills),
        instances_path=instances,
        output_path=output,
        top_k=args.top_k,
        limit=args.limit,
        sra_repo=ROOT / SRA_SUBMODULE_DIR,
    )
    print(json.dumps({"retrieval_output": str(output), "metrics": result["metrics"]}, indent=2))
    args.retrieval_output = str(output)
    return output


def infer(args: argparse.Namespace) -> Path:
    retrieval = Path(args.retrieval_output) if args.retrieval_output else default_retrieval_output(args.dataset)
    instances = Path(getattr(args, "instances", "") or default_instances(args.dataset))
    output = Path(args.inference_output) if args.inference_output else default_inference_output(args.dataset, args.model, args.provider_k)
    command = [
        "uv",
        "run",
        "--project",
        str(ROOT / SRA_SUBMODULE_DIR),
        "sragents",
        "infer",
        "--instances",
        str(instances),
        "--output",
        str(output),
        "--model",
        args.model,
        "--provider",
        "topk",
        "--provider-arg",
        f"source={retrieval}",
        "--provider-arg",
        f"k={args.provider_k}",
        "--provider-arg",
        f"corpus_path={Path(args.corpus) if hasattr(args, 'corpus') else ROOT / SRA_CORPUS_PATH}",
        "--engine",
        args.engine,
        "--workers",
        str(args.workers),
        "--temperature",
        str(args.temperature),
        "--max-tokens",
        str(args.max_tokens),
        "--label",
        f"skillbrowser_hybrid_top{args.provider_k}_{args.engine}",
    ]
    if args.api_base:
        command.extend(["--api-base", args.api_base])
    if args.force:
        command.append("--force")
    run_command(command)
    args.inference_output = str(output)
    return output


def infer_agent(args: argparse.Namespace) -> Path:
    corpus = load_sra_corpus(args.corpus)
    skills = load_sra_search_specs(args, corpus)
    searcher = build_sra_searcher(skills, args)
    instances = Path(args.instances) if args.instances else default_instances(args.dataset)
    output = (
        Path(args.inference_output)
        if args.inference_output
        else default_agent_inference_output(args.dataset, args.model, args.agent_top_k)
    )
    llm = build_agent_llm(args)
    result = run_sra_agent_inference(
        searcher=searcher,
        corpus=corpus,
        instances_path=instances,
        output_path=output,
        llm=llm,
        model_name=args.model,
        top_k=args.agent_top_k,
        max_rounds=args.max_rounds,
        limit=args.limit,
        inject_full_search_skill=args.inject_full_search_skill,
        force=args.force,
        sra_repo=ROOT / SRA_SUBMODULE_DIR,
    )
    print(json.dumps(result, indent=2))
    args.inference_output = str(output)
    return output


def preprocess(args: argparse.Namespace) -> Path:
    llm = build_preprocess_llm(args)
    result = run_sra_preprocess(
        corpus_path=args.corpus,
        output_skill_dir=args.output_skill_dir,
        checkpoint_path=args.checkpoint,
        llm=llm,
        model_name=args.model,
        limit=args.limit,
        resume=args.resume,
        force=args.force,
    )
    print(json.dumps(result, indent=2))
    return Path(args.output_skill_dir)


def load_sra_search_specs(args: argparse.Namespace, corpus: list[dict]):
    skill_dir = getattr(args, "sra_skill_dir", None)
    if skill_dir:
        return load_skills(Path(skill_dir))
    return sra_corpus_to_specs(corpus)


def build_sra_searcher(skills, args: argparse.Namespace):
    dense_view_names = SRA_METADATA_DENSE_VIEW_NAMES if getattr(args, "sra_skill_dir", None) else None
    return build_searcher(skills, args, dense_view_names=dense_view_names)


def evaluate(args: argparse.Namespace) -> Path:
    inference = Path(args.inference_output)
    instances = Path(getattr(args, "instances", "") or default_instances(args.dataset))
    output = Path(args.eval_output) if args.eval_output else default_eval_output(args.dataset, inference)
    command = [
        "uv",
        "run",
        "--project",
        str(ROOT / SRA_SUBMODULE_DIR),
        "sragents",
        "evaluate",
        "--input",
        str(inference),
        "--instances",
        str(instances),
        "--output",
        str(output),
        "--workers",
        str(args.eval_workers),
    ]
    if args.eval_force:
        command.append("--force")
    run_command(command)
    args.eval_output = str(output)
    return output


def summarize_agent(args: argparse.Namespace) -> Path:
    inference = Path(args.inference_output)
    instances = Path(getattr(args, "instances", "") or default_instances(args.dataset))
    eval_output = getattr(args, "eval_output", None)
    eval_path = Path(eval_output) if eval_output else None
    output = (
        Path(args.summary_output)
        if getattr(args, "summary_output", None)
        else default_agent_summary_output(args.dataset, inference)
    )
    payload = write_sra_agent_summary(
        inference_path=inference,
        instances_path=instances,
        output_path=output,
        eval_path=eval_path,
    )
    print(json.dumps({"summary_output": str(output), **payload.get("agent_metrics", {})}, indent=2))
    return output


def build_agent_llm(args: argparse.Namespace):
    if args.llm == "mock":
        return MockLLMClient()
    config = load_app_config(args.config)
    if config.llm is None:
        raise ValueError("Config file must include an [llm] section for openai-compatible SRA agent mode")
    return OpenAICompatibleLLMClient(
        base_url=args.api_base or config.llm.base_url,
        api_key=config.llm.api_key,
        model=args.model or config.llm.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=config.llm.timeout,
    )


def build_preprocess_llm(args: argparse.Namespace):
    if args.llm == "mock":
        return MockLLMClient(
            [
                json.dumps(
                    {
                        "short_description": "Use the named SRA skill for matching benchmark tasks.",
                        "long_description": (
                            "This generated smoke metadata describes the skill using its title, "
                            "description, and source content. Use it when a query asks for the same "
                            "method, formula, tool, or reasoning pattern."
                        ),
                        "capabilities": [
                            {
                                "id": "apply_sra_skill",
                                "description": "Apply the SR-Agents skill to a matching benchmark task.",
                            }
                        ],
                        "when_to_use": ["The task asks for the same method or tool described by this skill."],
                        "positive_examples": [
                            {
                                "user_query": "Find the matching SR-Agents skill for this task.",
                                "reason": "Smoke metadata for preprocessing tests.",
                            }
                        ],
                        "tags": ["sra-bench", "smoke"],
                    }
                )
            ]
        )
    config = load_app_config(args.config)
    if config.llm is None:
        raise ValueError("Config file must include an [llm] section for openai-compatible SRA preprocessing")
    return OpenAICompatibleLLMClient(
        base_url=args.api_base or config.llm.base_url,
        api_key=config.llm.api_key,
        model=args.model or config.llm.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=config.llm.timeout,
    )


def run_command(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def default_instances(dataset: str) -> Path:
    return ROOT / SRA_INSTANCES_DIR / f"{dataset}.json"


def default_retrieval_output(dataset: str) -> Path:
    return ROOT / SRA_RESULTS_DIR / "retrieval" / f"{dataset}-skillbrowser-hybrid.json"


def default_inference_output(dataset: str, model: str, provider_k: int) -> Path:
    model_name = Path(model).name.replace(":", "_")
    return ROOT / SRA_RESULTS_DIR / "inference" / f"{dataset}-{model_name}-hybrid_top{provider_k}.jsonl"


def default_agent_inference_output(dataset: str, model: str, agent_top_k: int) -> Path:
    model_name = Path(model).name.replace(":", "_")
    return ROOT / SRA_RESULTS_DIR / "inference" / f"{dataset}-{model_name}-general_agent_top{agent_top_k}.jsonl"


def default_eval_output(dataset: str, inference: Path) -> Path:
    return ROOT / SRA_RESULTS_DIR / "eval" / f"{dataset}-{inference.stem}.json"


def default_agent_summary_output(dataset: str, inference: Path) -> Path:
    return ROOT / SRA_RESULTS_DIR / "eval" / f"{dataset}-{inference.stem}-agent-summary.json"


if __name__ == "__main__":
    raise SystemExit(main())
