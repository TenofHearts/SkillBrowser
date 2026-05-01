from __future__ import annotations

from .loader import SkillLoadError, load_skill_document
from .schema import SkillReadRequest, SkillReadResponse, SkillSpec
from .sections import normalize_section_name, parse_markdown_sections, token_count, truncate_tokens


class SkillReader:
    def __init__(self, skills: list[SkillSpec]):
        self.skills = {skill.id: skill for skill in skills}

    def read(self, request: SkillReadRequest) -> SkillReadResponse:
        skill = self.skills.get(request.skill_id)
        if skill is None:
            raise SkillLoadError(f"Unknown skill id: {request.skill_id}")

        full_text = load_skill_document(skill)
        sections = parse_markdown_sections(full_text)
        available = [section.key for section in sections]

        selected_section = request.section
        if selected_section:
            key = normalize_section_name(selected_section)
            matching = [section for section in sections if section.key == key]
            if not matching:
                raise SkillLoadError(
                    f"Section '{request.section}' not found for {skill.id}. Available: {', '.join(available)}"
                )
            content = matching[0].content
        else:
            content = full_text

        content, truncated = truncate_tokens(content, request.max_tokens)
        return SkillReadResponse(
            skill_id=skill.id,
            name=skill.name,
            section=selected_section,
            content=content,
            token_count=token_count(content),
            truncated=truncated,
            available_sections=available,
        )
