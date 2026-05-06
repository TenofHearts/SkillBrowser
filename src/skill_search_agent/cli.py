from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ValidationError

from .agent import SkillAgent
from .config import load_app_config, load_app_config_if_exists
from .loader import SkillLoadError, load_skills
from .evaluation import evaluate_retrieval, load_retrieval_dataset
from .gatewaybench import (
    compare_gatewaybench_selectors,
    evaluate_gatewaybench_lite,
    load_gatewaybench_lite_dataset,
    make_gatewaybench_hybrid_selector,
)
from .llm import MockLLMClient, OpenAICompatibleLLMClient
from .reader import SkillReader
from .registry import default_db_path, dump_json_summary, rebuild_registry
from .schema import AgentRunRequest, SkillReadRequest, SkillSearchRequest
from .search import SkillSearcher
from .selectors import BaselineLLMToolSelector


def print_json(model: BaseModel) -> None:
    print(model.json(indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skill-agent")
    parser.add_argument("--skill-dir", default="data/skills", help="Directory containing */skill.yaml")
    command_parent = argparse.ArgumentParser(add_help=False)
    command_parent.add_argument(
        "--skill-dir",
        dest="skill_dir",
        default=argparse.SUPPRESS,
        help="Directory containing */skill.yaml",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate-skills", parents=[command_parent], help="Validate local skill specs")

    build_index = sub.add_parser("build-index", parents=[command_parent], help="Build the local SQLite skill registry")
    build_index.add_argument("--index-dir", default="data/indexes")
    build_index.add_argument("--db-path")

    search = sub.add_parser("search", parents=[command_parent], help="Search local skills")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)

    run_agent = sub.add_parser("run-agent", parents=[command_parent], help="Run the LLM skill-selection agent")
    run_agent.add_argument("task")
    run_agent.add_argument("--top-k", type=int)
    run_agent.add_argument("--max-steps", type=int)
    run_agent.add_argument("--read-max-tokens", type=int)
    run_agent.add_argument("--llm", choices=["mock", "openai-compatible"])
    run_agent.add_argument("--config", default="config.toml", help="TOML config file for openai-compatible LLM mode")

    read = sub.add_parser("read", parents=[command_parent], help="Read a skill document or section")
    read.add_argument("skill_id")
    read.add_argument("--section")
    read.add_argument("--max-tokens", type=int, default=2000)

    eval_retrieval = sub.add_parser(
        "eval-retrieval", parents=[command_parent], help="Evaluate skill retrieval on a JSONL dataset"
    )
    eval_retrieval.add_argument("--dataset", required=True)
    eval_retrieval.add_argument("--top-k", type=int, default=5)

    eval_gatewaybench = sub.add_parser(
        "eval-gatewaybench-lite",
        help="Evaluate skill retrieval on a local GatewayBench JSONL export",
    )
    eval_gatewaybench.add_argument("--dataset", required=True)
    eval_gatewaybench.add_argument("--top-k", type=int, default=5)
    eval_gatewaybench.add_argument("--limit", type=int)

    eval_gatewaybench_compare = sub.add_parser(
        "eval-gatewaybench-compare",
        help="Compare GatewayBench selectors on a local JSONL export",
    )
    eval_gatewaybench_compare.add_argument("--dataset")
    eval_gatewaybench_compare.add_argument("--top-k", type=int)
    eval_gatewaybench_compare.add_argument("--limit", type=int)
    eval_gatewaybench_compare.add_argument("--workers", type=int)
    eval_gatewaybench_compare.add_argument(
        "--selector",
        action="append",
        choices=["hybrid", "llm-baseline"],
        default=None,
        help="Selector to evaluate. Repeat to compare multiple selectors.",
    )
    eval_gatewaybench_compare.add_argument("--llm", choices=["mock", "openai-compatible"])
    eval_gatewaybench_compare.add_argument(
        "--config",
        default="config.toml",
        help="TOML config file for openai-compatible LLM mode",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "eval-gatewaybench-lite":
            examples = load_gatewaybench_lite_dataset(args.dataset, limit=args.limit)
            print_json(evaluate_gatewaybench_lite(examples, args.top_k))
            return 0
        if args.command == "eval-gatewaybench-compare":
            config = load_app_config_if_exists(args.config)
            dataset = args.dataset or config.gatewaybench_compare.dataset
            if not dataset:
                raise ValueError("GatewayBench dataset must be set by --dataset or gatewaybench_compare.dataset")
            top_k = args.top_k if args.top_k is not None else config.gatewaybench_compare.top_k
            limit = args.limit if args.limit is not None else config.gatewaybench_compare.limit
            workers = args.workers if args.workers is not None else config.gatewaybench_compare.workers
            llm_mode = args.llm or config.gatewaybench_compare.llm
            examples = load_gatewaybench_lite_dataset(dataset, limit=limit)
            llm = build_llm(llm_mode, args.config)
            requested_selectors = args.selector or config.gatewaybench_compare.selectors
            factories = {}
            if "hybrid" in requested_selectors:
                factories["hybrid"] = make_gatewaybench_hybrid_selector
            if "llm-baseline" in requested_selectors:
                factories["llm-baseline"] = lambda _example: BaselineLLMToolSelector(llm)
            print_json(compare_gatewaybench_selectors(examples, factories, top_k, workers=workers, progress=True))
            return 0

        skills = load_skills(Path(args.skill_dir))
        if args.command == "validate-skills":
            print(json.dumps({"ok": True, "skill_count": len(skills), "skill_ids": [s.id for s in skills]}, indent=2))
            return 0
        if args.command == "build-index":
            db_path = Path(args.db_path) if args.db_path else default_db_path(args.index_dir)
            summary = rebuild_registry(skills, db_path)
            print(dump_json_summary({"ok": True, "db_path": str(db_path), **summary}))
            return 0
        if args.command == "search":
            print_json(SkillSearcher(skills).search(SkillSearchRequest(query=args.query, top_k=args.top_k)))
            return 0
        if args.command == "run-agent":
            config = load_app_config_if_exists(args.config)
            llm_mode = args.llm or config.agent.llm
            llm = build_llm(llm_mode, args.config)
            result = SkillAgent(SkillSearcher(skills), SkillReader(skills), llm).run(
                AgentRunRequest(
                    task=args.task,
                    top_k=args.top_k if args.top_k is not None else config.agent.top_k,
                    max_steps=args.max_steps if args.max_steps is not None else config.agent.max_steps,
                    read_max_tokens=(
                        args.read_max_tokens if args.read_max_tokens is not None else config.agent.read_max_tokens
                    ),
                )
            )
            print_json(result)
            return 0
        if args.command == "read":
            print_json(
                SkillReader(skills).read(
                    SkillReadRequest(skill_id=args.skill_id, section=args.section, max_tokens=args.max_tokens)
                )
            )
            return 0
        if args.command == "eval-retrieval":
            examples = load_retrieval_dataset(args.dataset)
            print_json(evaluate_retrieval(SkillSearcher(skills), examples, args.top_k))
            return 0
    except (SkillLoadError, ValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.error(f"Unhandled command: {args.command}")
    return 2


def build_llm(mode: str, config_path: str = "config.toml"):
    if mode == "mock":
        return MockLLMClient()
    if mode == "openai-compatible":
        config = load_app_config(config_path)
        if config.llm is None:
            raise ValueError("Config file must include an [llm] section for openai-compatible LLM mode")
        return OpenAICompatibleLLMClient(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            timeout=config.llm.timeout,
        )
    raise ValueError(f"Unsupported LLM mode: {mode}")


if __name__ == "__main__":
    raise SystemExit(main())
