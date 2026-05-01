from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from skill_search_agent.loader import SkillLoadError, load_skills
from skill_search_agent.evaluation import evaluate_retrieval, load_retrieval_dataset
from skill_search_agent.reader import SkillReader
from skill_search_agent.registry import rebuild_registry, registry_summary
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


def test_registry_persists_skills_sections_and_views() -> None:
    pytest.importorskip("_sqlite3")
    skills = load_skills(SKILL_DIR)
    db_dir = Path(__file__).parent / ".tmp"
    db_dir.mkdir(exist_ok=True)
    db_path = db_dir / f"skills-{uuid4().hex}.db"

    try:
        result = rebuild_registry(skills, db_path)
        summary = registry_summary(db_path)
    finally:
        db_path.unlink(missing_ok=True)

    assert result["skill_count"] == 2
    assert summary["skills"] == 2
    assert summary["skill_documents"] == 2
    assert summary["skill_sections"] >= 8
    assert summary["skill_views"] >= 10


def test_retrieval_evaluation_reports_metrics() -> None:
    skills = load_skills(SKILL_DIR)
    examples = load_retrieval_dataset(Path(__file__).parent / "fixtures" / "retrieval_eval.jsonl")

    result = evaluate_retrieval(SkillSearcher(skills), examples, top_k=1)

    assert result.query_count == 2
    assert result.recall_at_k == 1.0
    assert result.mrr == 1.0
    assert result.misses == []
