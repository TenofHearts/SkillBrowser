"""Command-line interface for validating, indexing, searching, reading, and evaluating skills."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ValidationError

from agent import SkillAgent
from benchmarks.retrieval_runner import run_eval_retrieval
from benchmarks.toolret_runner import run_build_toolret_candidates, run_eval_toolret
from config import load_app_config, load_app_config_if_exists
from core.search import SkillSearcher
from loader import SkillLoadError, load_skills
from llm import MockLLMClient, OpenAICompatibleLLMClient
from reader import SkillReader
from registry import default_db_path, dump_json_summary, rebuild_registry
from schema import AgentRunRequest, SkillReadRequest, SkillSearchRequest


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

    eval_toolret = sub.add_parser(
        "eval-toolret",
        help="Evaluate retrieval-only ToolRet queries against a ToolRet tool corpus",
    )
    eval_toolret.add_argument("--queries", help="ToolRet query JSONL/JSON/parquet export")
    eval_toolret.add_argument("--tools", help="ToolRet tool JSONL/JSON/parquet export")
    eval_toolret.add_argument(
        "--first-stage-candidates",
        help="Optional RankGPT first-stage candidate JSON/JSONL, e.g. NV-Embed-v1 candidates from ToolRet",
    )
    eval_toolret.add_argument("--subset", help="Optional ToolRet subset filter, e.g. apibank")
    eval_toolret.add_argument("--category", choices=["all", "web", "code", "customized"])
    eval_toolret.add_argument("--limit", type=int)
    eval_toolret.add_argument("--top-k", type=int)
    instruction_group = eval_toolret.add_mutually_exclusive_group()
    instruction_group.add_argument("--use-instruction", dest="use_instruction", action="store_true")
    instruction_group.add_argument("--no-instruction", dest="use_instruction", action="store_false")
    eval_toolret.set_defaults(use_instruction=None)
    eval_toolret.add_argument(
        "--baseline",
        choices=["hybrid", "rankgpt", "llm-rerank", "toolret-rankgpt", "paper-rankgpt", "compare"],
    )
    eval_toolret.add_argument("--candidate-pool-size", type=int)
    eval_toolret.add_argument("--rankgpt-window-size", type=int)
    eval_toolret.add_argument("--rankgpt-step-size", type=int)
    eval_toolret.add_argument("--workers", type=int)
    eval_toolret.add_argument("--llm", choices=["mock", "openai-compatible"])
    eval_toolret.add_argument("--output", help="Optional path to write JSON result")
    eval_toolret.add_argument(
        "--config",
        default="config.toml",
        help="TOML config file for defaults and openai-compatible LLM mode",
    )

    build_toolret_candidates = sub.add_parser(
        "build-toolret-candidates",
        help="Build ToolRet first-stage candidates with NV-Embed-v1 over SkillSpec documents",
    )
    build_toolret_candidates.add_argument("--queries", required=True, help="ToolRet query JSONL/JSON/parquet export")
    build_toolret_candidates.add_argument("--tools", required=True, help="ToolRet tool JSONL/JSON/parquet export")
    build_toolret_candidates.add_argument("--output", required=True, help="Path to write candidate JSONL")
    build_toolret_candidates.add_argument("--subset", help="Optional ToolRet subset filter, e.g. apibank")
    build_toolret_candidates.add_argument("--category", choices=["all", "web", "code", "customized"], default="all")
    build_toolret_candidates.add_argument("--limit", type=int)
    build_toolret_candidates.add_argument("--top-k", type=int, default=100)
    build_candidates_instruction_group = build_toolret_candidates.add_mutually_exclusive_group()
    build_candidates_instruction_group.add_argument("--use-instruction", dest="use_instruction", action="store_true")
    build_candidates_instruction_group.add_argument("--no-instruction", dest="use_instruction", action="store_false")
    build_toolret_candidates.set_defaults(use_instruction=True)
    build_toolret_candidates.add_argument("--model", default="nvidia/NV-Embed-v1")
    build_toolret_candidates.add_argument("--batch-size", type=int, default=8)
    build_toolret_candidates.add_argument("--max-length", type=int, default=4096)
    build_toolret_candidates.add_argument("--device", help="Optional torch device, e.g. cuda:0 or cpu")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "build-toolret-candidates":
            print_json(run_build_toolret_candidates(args))
            return 0

        if args.command == "eval-toolret":
            print_json(run_eval_toolret(args, build_llm))
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
            print_json(run_eval_retrieval(args, skills))
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
