from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ValidationError

from .loader import SkillLoadError, load_skills
from .evaluation import evaluate_retrieval, load_retrieval_dataset
from .reader import SkillReader
from .registry import default_db_path, dump_json_summary, rebuild_registry
from .schema import SkillReadRequest, SkillSearchRequest
from .search import SkillSearcher


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

    read = sub.add_parser("read", parents=[command_parent], help="Read a skill document or section")
    read.add_argument("skill_id")
    read.add_argument("--section")
    read.add_argument("--max-tokens", type=int, default=2000)

    eval_retrieval = sub.add_parser(
        "eval-retrieval", parents=[command_parent], help="Evaluate skill retrieval on a JSONL dataset"
    )
    eval_retrieval.add_argument("--dataset", required=True)
    eval_retrieval.add_argument("--top-k", type=int, default=5)

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
            print(dump_json_summary({"ok": True, "db_path": str(db_path), **summary}))
            return 0
        if args.command == "search":
            print_json(SkillSearcher(skills).search(SkillSearchRequest(query=args.query, top_k=args.top_k)))
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


if __name__ == "__main__":
    raise SystemExit(main())
