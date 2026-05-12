"""Build searchable text views from skill metadata and markdown content."""

from __future__ import annotations

from dataclasses import dataclass

from loader import load_skill_document
from schema import SkillSpec
from .sections import parse_markdown_sections


@dataclass(frozen=True)
class SkillView:
    skill_id: str
    view_name: str
    text: str


def build_skill_views(skill: SkillSpec) -> list[SkillView]:
    views = [
        SkillView(
            skill.id,
            "description",
            "\n".join(
                [
                    skill.name,
                    skill.description.short,
                    skill.description.long or "",
                    skill.category.primary,
                    " ".join(skill.category.secondary),
                    " ".join(skill.tags),
                ]
            ),
        ),
        SkillView(
            skill.id,
            "capability",
            "\n".join(f"{capability.id}: {capability.description}" for capability in skill.capabilities),
        ),
        SkillView(skill.id, "usage", "\n".join(skill.when_to_use)),
        SkillView(
            skill.id,
            "examples",
            "\n".join(example.user_query for example in skill.examples.positive),
        ),
    ]

    if skill.input_schema or skill.output_schema:
        views.append(
            SkillView(
                skill.id,
                "schema",
                "\n".join(
                    [
                        " ".join(skill.input_types),
                        " ".join(skill.output_types),
                        str(skill.input_schema or ""),
                        str(skill.output_schema or ""),
                    ]
                ),
            )
        )

    try:
        document = load_skill_document(skill)
        for section in parse_markdown_sections(document):
            if section.key in {"failure_modes", "when_not_to_use", "contraindications"}:
                continue
            views.append(SkillView(skill.id, f"content_section:{section.key}", section.content))
    except Exception:
        pass

    return [view for view in views if view.text.strip()]


def build_skill_search_text(skill: SkillSpec) -> str:
    return "\n".join(view.text for view in build_skill_views(skill))
