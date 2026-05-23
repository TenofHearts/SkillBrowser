"""Command-line interface for validating, indexing, searching, reading, and evaluating skills."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ValidationError

from agent import SkillAgent
from benchmarks.retrieval_runner import run_eval_retrieval
from config import load_app_config, load_app_config_if_exists
from core.embeddings import build_embedder
from core.search import SearchWeights, SkillSearcher
from loader import SkillLoadError, load_skills
from llm import MockLLMClient, OpenAICompatibleLLMClient
from reader import SkillReader
from registry import default_db_path, dump_json_summary, persist_dense_embeddings, rebuild_registry
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
    add_retrieval_arguments(build_index)

    search = sub.add_parser("search", parents=[command_parent], help="Search local skills")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    add_retrieval_arguments(search)

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
    add_retrieval_arguments(eval_retrieval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        skills = load_skills(Path(args.skill_dir))
        if args.command == "validate-skills":
            print(json.dumps({"ok": True, "skill_count": len(skills), "skill_ids": [s.id for s in skills]}, indent=2))
            return 0
        if args.command == "build-index":
            db_path = Path(args.db_path) if args.db_path else default_db_path(args.index_dir)
            summary = rebuild_registry(skills, db_path)
            if should_persist_embeddings(args):
                embedder = build_configured_embedder(args, backend=resolved_embedding_backend(args))
                summary.update(persist_dense_embeddings(skills, db_path, args.index_dir, embedder))
            print(dump_json_summary({"ok": True, "db_path": str(db_path), **summary}))
            return 0
        if args.command == "search":
            print_json(build_searcher(skills, args).search(SkillSearchRequest(query=args.query), top_k=args.top_k))
            return 0
        if args.command == "run-agent":
            config = load_app_config_if_exists(args.config)
            llm_mode = args.llm or config.agent.llm
            llm = build_llm(llm_mode, args.config)
            result = SkillAgent(build_searcher(skills, args), SkillReader(skills), llm).run(
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
            print_json(run_eval_retrieval(args, skills, build_searcher(skills, args)))
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


def add_retrieval_arguments(parser: argparse.ArgumentParser, *, include_config: bool = True) -> None:
    parser.add_argument(
        "--retrieval-mode",
        choices=["hybrid", "bm25", "dense"],
        help="Search mode. Defaults to config embedding settings, or bm25 when embeddings are disabled.",
    )
    parser.add_argument("--embedding-backend", choices=["none", "fake", "hf-transformers"])
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-batch-size", type=int)
    parser.add_argument("--embedding-max-length", type=int)
    parser.add_argument("--embedding-device")
    parser.add_argument("--embedding-cache-dir")
    parser.add_argument("--weight-lexical", type=float)
    parser.add_argument("--weight-sparse-view", type=float)
    parser.add_argument("--weight-dense", type=float)
    parser.add_argument("--weight-rrf", type=float)
    parser.add_argument("--weight-capability", type=float)
    parser.add_argument("--weight-usage", type=float)
    parser.add_argument("--weight-input-type", type=float)
    parser.add_argument("--weight-output-type", type=float)
    parser.add_argument("--weight-penalty", type=float)
    parser.add_argument(
        "--minimum-score-threshold",
        type=float,
        help="Exclude skills whose final search score is less than or equal to this threshold.",
    )
    if include_config:
        parser.add_argument("--config", default="config.toml", help="TOML config file for retrieval defaults")


def build_searcher(
    skills,
    args: argparse.Namespace,
    *,
    dense_view_names: set[str] | None = None,
) -> SkillSearcher:
    config = load_app_config_if_exists(getattr(args, "config", "config.toml"))
    embedding_config = config.embedding
    mode = getattr(args, "retrieval_mode", None)
    backend = resolved_embedding_backend(args)
    dense_enabled = embedding_config.enabled and backend != "none"

    if mode == "bm25":
        dense_enabled = False
        backend = "none"
        bm25_enabled = True
        sparse_view_enabled = False
    elif mode == "dense":
        dense_enabled = True
        if backend == "none":
            backend = "hf-transformers"
        bm25_enabled = False
        sparse_view_enabled = False
    elif mode == "hybrid":
        dense_enabled = backend != "none"
        bm25_enabled = True
        sparse_view_enabled = True
    else:
        bm25_enabled = True
        sparse_view_enabled = True

    embedder = build_configured_embedder(args, backend=backend)
    return SkillSearcher(
        skills,
        embedder=embedder,
        dense_enabled=dense_enabled,
        bm25_enabled=bm25_enabled,
        sparse_view_enabled=sparse_view_enabled,
        dense_view_names=dense_view_names,
        weights=build_search_weights(args),
        minimum_score_threshold=build_minimum_score_threshold(args),
        dense_cache_dir=getattr(args, "embedding_cache_dir", None) or embedding_config.cache_dir,
    )


def build_configured_embedder(args: argparse.Namespace, backend: str | None = None):
    config = load_app_config_if_exists(getattr(args, "config", "config.toml"))
    embedding_config = config.embedding
    return build_embedder(
        backend or getattr(args, "embedding_backend", None) or embedding_config.backend,
        model_name=getattr(args, "embedding_model", None) or embedding_config.model,
        batch_size=getattr(args, "embedding_batch_size", None) or embedding_config.batch_size,
        max_length=getattr(args, "embedding_max_length", None) or embedding_config.max_length,
        device=getattr(args, "embedding_device", None) or embedding_config.device,
    )


def resolved_embedding_backend(args: argparse.Namespace) -> str:
    config = load_app_config_if_exists(getattr(args, "config", "config.toml"))
    backend = getattr(args, "embedding_backend", None) or config.embedding.backend
    if getattr(args, "retrieval_mode", None) == "dense" and backend == "none":
        return "hf-transformers"
    return backend


def should_persist_embeddings(args: argparse.Namespace) -> bool:
    config = load_app_config_if_exists(getattr(args, "config", "config.toml"))
    backend = resolved_embedding_backend(args)
    mode = getattr(args, "retrieval_mode", None)
    if mode == "dense":
        return True
    if mode == "hybrid":
        return backend != "none"
    return config.embedding.enabled and backend != "none"


def build_search_weights(args: argparse.Namespace) -> SearchWeights:
    config = load_app_config_if_exists(getattr(args, "config", "config.toml"))
    search_config = config.search
    defaults = SearchWeights()
    return SearchWeights(
        lexical=_configured_weight(args, search_config, "weight_lexical", defaults.lexical),
        sparse_view=(
            _configured_weight(args, search_config, "weight_sparse_view", defaults.sparse_view)
        ),
        dense=_configured_weight(args, search_config, "weight_dense", defaults.dense),
        rrf=_configured_weight(args, search_config, "weight_rrf", defaults.rrf),
        capability=(
            _configured_weight(args, search_config, "weight_capability", defaults.capability)
        ),
        usage=_configured_weight(args, search_config, "weight_usage", defaults.usage),
        input_type=(
            _configured_weight(args, search_config, "weight_input_type", defaults.input_type)
        ),
        output_type=(
            _configured_weight(args, search_config, "weight_output_type", defaults.output_type)
        ),
        penalty=_configured_weight(args, search_config, "weight_penalty", defaults.penalty),
    )


def build_minimum_score_threshold(args: argparse.Namespace) -> float:
    config = load_app_config_if_exists(getattr(args, "config", "config.toml"))
    cli_value = getattr(args, "minimum_score_threshold", None)
    if cli_value is not None:
        return cli_value
    return config.search.minimum_score_threshold


def _configured_weight(args: argparse.Namespace, search_config, name: str, default: float) -> float:
    cli_value = getattr(args, name, None)
    if cli_value is not None:
        return cli_value
    config_value = getattr(search_config, name)
    if config_value is not None:
        return config_value
    return default


if __name__ == "__main__":
    raise SystemExit(main())
