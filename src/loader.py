from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

from schema import SkillSpec


class SkillLoadError(ValueError):
    pass


def find_skill_files(skill_dir: Path) -> Iterable[Path]:
    yield from sorted(skill_dir.glob("*/skill.yaml"))


def load_skill(path: Path) -> SkillSpec:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        spec = SkillSpec.parse_obj(raw)
    except Exception as exc:  # pydantic and yaml errors both need path context
        raise SkillLoadError(f"Failed to load {path}: {exc}") from exc
    spec.root_dir = path.parent
    return spec


def load_skills(skill_dir: str | Path) -> list[SkillSpec]:
    root = Path(skill_dir)
    if not root.exists():
        raise SkillLoadError(f"Skill directory does not exist: {root}")
    skills = [load_skill(path) for path in find_skill_files(root)]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for skill in skills:
        if skill.id in seen:
            duplicates.add(skill.id)
        seen.add(skill.id)
    if duplicates:
        raise SkillLoadError(f"Duplicate skill id(s): {', '.join(sorted(duplicates))}")
    return skills


def load_skill_document(skill: SkillSpec) -> str:
    if skill.root_dir is None:
        raise SkillLoadError(f"Skill {skill.id} has no root directory")
    content_path = Path(skill.content.path)
    if content_path.is_absolute():
        raise SkillLoadError(f"Skill document path must be relative for {skill.id}: {content_path}")
    root = skill.root_dir.resolve()
    path = (root / content_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SkillLoadError(f"Skill document path escapes skill directory for {skill.id}: {content_path}") from exc
    if not path.exists():
        raise SkillLoadError(f"Skill document does not exist for {skill.id}: {path}")
    return path.read_text(encoding="utf-8")
