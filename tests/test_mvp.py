from __future__ import annotations

from pathlib import Path

import pytest

from skill_search_agent.loader import SkillLoadError, load_skills
from skill_search_agent.reader import SkillReader
from skill_search_agent.schema import SkillReadRequest, SkillSearchRequest
from skill_search_agent.search import SkillSearcher


SKILL_DIR = Path(__file__).parent / "fixtures" / "skills"


def test_load_skills() -> None:
    skills = load_skills(SKILL_DIR)
    assert {skill.id for skill in skills} == {"research.paper_claim_method_finding", "pdf.extract_text"}
    assert skills[0].interaction.readable


def test_search_returns_relevant_skill() -> None:
    skills = load_skills(SKILL_DIR)
    response = SkillSearcher(skills).search(SkillSearchRequest(query="extract text from a PDF", top_k=1))
    assert response.results[0].id == "pdf.extract_text"
    assert response.results[0].execution_available is True
    assert response.results[0].read_recommendation in response.results[0].available_sections


def test_search_penalizes_when_not_to_use_matches() -> None:
    skills = load_skills(SKILL_DIR)
    response = SkillSearcher(skills).search(SkillSearchRequest(query="OCR screenshot image", top_k=5))
    assert all(card.id != "pdf.extract_text" for card in response.results)


def test_read_specific_section() -> None:
    skills = load_skills(SKILL_DIR)
    response = SkillReader(skills).read(
        SkillReadRequest(skill_id="research.paper_claim_method_finding", section="procedure", max_tokens=20)
    )
    assert response.section == "procedure"
    assert response.token_count <= 20
    assert "claim" in response.content.lower()


def test_missing_section_is_clear() -> None:
    skills = load_skills(SKILL_DIR)
    with pytest.raises(SkillLoadError, match="Available"):
        SkillReader(skills).read(SkillReadRequest(skill_id="pdf.extract_text", section="missing"))


def test_duplicate_skill_ids_fail() -> None:
    with pytest.raises(SkillLoadError, match="Duplicate skill id"):
        load_skills(Path(__file__).parent / "fixtures" / "duplicate_skills")


def test_document_path_must_stay_inside_skill_root() -> None:
    skills = load_skills(Path(__file__).parent / "fixtures" / "bad_path_skill")
    with pytest.raises(SkillLoadError, match="escapes skill directory"):
        SkillReader(skills).read(SkillReadRequest(skill_id="bad.path"))
