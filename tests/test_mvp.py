from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from skill_search_agent.loader import SkillLoadError, load_skills
from skill_search_agent.cli import main
from skill_search_agent.evaluation import RetrievalExample, evaluate_retrieval, load_retrieval_dataset, score_retrieval_result
from skill_search_agent.gatewaybench import (
    gateway_example_to_skills,
    load_gatewaybench_lite_dataset,
)
from skill_search_agent.reader import SkillReader
from skill_search_agent.registry import rebuild_registry, registry_summary
from skill_search_agent.schema import SkillReadRequest, SkillSearchRequest
from skill_search_agent.search import SkillSearcher
from skill_search_agent.views import build_skill_search_text


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
    assert result.precision_at_k == 1.0
    assert result.f1_at_k == 1.0
    assert result.misses == []


def test_legacy_retrieval_examples_normalize_to_expected_skill_ids() -> None:
    examples = load_retrieval_dataset(Path(__file__).parent / "fixtures" / "retrieval_eval.jsonl")

    assert examples[0].expected_skill_ids == ["pdf.extract_text"]


def test_multi_answer_retrieval_metrics_score_sets() -> None:
    example = RetrievalExample(
        query="summarize a PDF paper claim method findings",
        expected_skill_ids=["pdf.extract_text", "research.paper_claim_method_finding"],
    )

    stats = score_retrieval_result(example, ["research.paper_claim_method_finding", "pdf.extract_text"], top_k=2)

    assert stats["hit"] is True
    assert stats["recall"] == 1.0
    assert stats["precision"] == 1.0
    assert stats["f1"] == 1.0


def test_precision_at_k_counts_missing_results_as_false_positives() -> None:
    example = RetrievalExample(query="extract text from a PDF", expected_skill_ids=["pdf.extract_text"])

    stats = score_retrieval_result(example, ["pdf.extract_text"], top_k=5)

    assert stats["precision"] == 0.2


def test_abstention_examples_penalize_non_empty_results() -> None:
    example = RetrievalExample(query="read a scanned screenshot with OCR", allow_no_result=True)

    empty_stats = score_retrieval_result(example, [], top_k=1)
    non_empty_stats = score_retrieval_result(example, ["csv.read"], top_k=1)

    assert empty_stats["hit"] is True
    assert non_empty_stats["hit"] is False
    assert non_empty_stats["recall"] == 0.0


def test_irrelevant_returned_skills_are_counted() -> None:
    example = RetrievalExample(
        query="load customer data",
        expected_skill_ids=["csv.read"],
        relevance_by_id={"csv.read": "required", "pdf.extract_text": "irrelevant"},
    )

    stats = score_retrieval_result(example, ["pdf.extract_text", "csv.read"], top_k=2)

    assert stats["irrelevant_returned"] == 1


def test_gatewaybench_lite_filters_and_hides_relevance_labels() -> None:
    examples = load_gatewaybench_lite_dataset(Path(__file__).parent / "fixtures" / "gatewaybench_lite.jsonl")

    assert [example.id for example in examples] == ["gb-1", "gb-2"]

    searchable_text = "\n".join(build_skill_search_text(skill) for skill in gateway_example_to_skills(examples[0]))

    assert "required" not in searchable_text
    assert "irrelevant" not in searchable_text


def test_local_hard_retrieval_eval_cli_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "eval-retrieval",
            "--skill-dir",
            "data/skills",
            "--dataset",
            "data/eval/local_hard_retrieval.jsonl",
            "--top-k",
            "3",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"query_count": 10' in captured.out
    assert '"irrelevant_selection_rate"' in captured.out


def test_gatewaybench_lite_eval_cli_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "eval-gatewaybench-lite",
            "--dataset",
            str(Path(__file__).parent / "fixtures" / "gatewaybench_lite.jsonl"),
            "--top-k",
            "2",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"query_count": 2' in captured.out


def test_cli_accepts_skill_dir_before_command(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--skill-dir", str(SKILL_DIR), "validate-skills"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"skill_count": 2' in captured.out


def test_cli_accepts_skill_dir_after_command(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["validate-skills", "--skill-dir", str(SKILL_DIR)])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"skill_count": 2' in captured.out
