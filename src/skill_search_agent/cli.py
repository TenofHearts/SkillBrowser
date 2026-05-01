from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ValidationError

from .loader import SkillLoadError, load_skills
from .reader import SkillReader
from .schema import SkillReadRequest, SkillSearchRequest
from .search import SkillSearcher


def print_json(model: BaseModel) -> None:
    print(model.json(indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skill-agent")
    parser.add_argument("--skill-dir", default="data/skills", help="Directory containing */skill.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate-skills", help="Validate local skill specs")

    search = sub.add_parser("search", help="Search local skills")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)

    read = sub.add_parser("read", help="Read a skill document or section")
    read.add_argument("skill_id")
    read.add_argument("--section")
    read.add_argument("--max-tokens", type=int, default=2000)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        skills = load_skills(Path(args.skill_dir))
        if args.command == "validate-skills":
            print(json.dumps({"ok": True, "skill_count": len(skills), "skill_ids": [s.id for s in skills]}, indent=2))
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
    except (SkillLoadError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
