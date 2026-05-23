from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from benchmarks.retrieval import (
    RetrievalExample,
    evaluate_retrieval,
    load_retrieval_dataset,
    score_retrieval_result,
)
from cli import build_searcher, main
from core.embeddings import FakeSemanticEmbedder
from core.search import SkillSearcher, rrf_fusion
from loader import SkillLoadError, load_skills
from reader import SkillReader
from registry import persist_dense_embeddings, rebuild_registry, registry_summary
from schema import SkillReadRequest, SkillSearchRequest


SKILL_DIR = Path(__file__).parent / "fixtures" / "skills"


def test_load_skills() -> None:
    skills = load_skills(SKILL_DIR)
    assert {skill.id for skill in skills} == {"research.paper_claim_method_finding", "pdf.extract_text"}
    assert skills[0].interaction.readable


def test_search_returns_relevant_skill() -> None:
    skills = load_skills(SKILL_DIR)
    response = SkillSearcher(skills).search(SkillSearchRequest(query="extract text from a PDF"), top_k=1)
    assert response.results[0].id == "pdf.extract_text"
    assert response.results[0].execution_available is True
    assert response.results[0].read_recommendation in response.results[0].available_sections


def test_search_filters_results_at_minimum_score_threshold() -> None:
    skills = load_skills(SKILL_DIR)
    baseline = SkillSearcher(skills).search(SkillSearchRequest(query="extract text from a PDF"), top_k=1)
    threshold = baseline.results[0].score

    response = SkillSearcher(skills, minimum_score_threshold=threshold).search(
        SkillSearchRequest(query="extract text from a PDF"),
        top_k=1,
    )

    assert response.results == []
    assert response.abstained is True
    assert response.abstention_reason == "no_candidate_above_threshold"


def test_search_penalizes_when_not_to_use_matches() -> None:
    skills = load_skills(SKILL_DIR)
    response = SkillSearcher(skills).search(SkillSearchRequest(query="OCR screenshot image"), top_k=5)
    assert all(card.id != "pdf.extract_text" for card in response.results)


def test_rrf_fuses_independent_rank_lists() -> None:
    fused = rrf_fusion(
        [
            ["first", "second"],
            ["second", "first"],
        ],
        k=10,
    )

    assert fused["first"] == pytest.approx((1 / 11) + (1 / 12))
    assert fused["second"] == pytest.approx((1 / 12) + (1 / 11))


def test_multi_view_retrieval_uses_example_and_vector_rank_lists() -> None:
    skills = load_skills(SKILL_DIR)
    response = SkillSearcher(skills).search(SkillSearchRequest(query="main contribution"), top_k=1)

    assert response.results[0].id == "research.paper_claim_method_finding"
    assert response.results[0].score_breakdown.sparse_view > 0
    assert response.results[0].score_breakdown.vector == 0
    assert response.results[0].score_breakdown.rrf > 0


def test_dense_retrieval_handles_semantic_query_without_keyword_overlap() -> None:
    skills = load_skills(SKILL_DIR)
    request = SkillSearchRequest(query="portable")

    bm25_response = SkillSearcher(
        skills,
        bm25_enabled=True,
        sparse_view_enabled=False,
    ).search(request)
    dense_response = SkillSearcher(
        skills,
        embedder=FakeSemanticEmbedder(),
        dense_enabled=True,
        bm25_enabled=False,
        sparse_view_enabled=False,
    ).search(request)

    assert bm25_response.results == []
    assert dense_response.results[0].id == "pdf.extract_text"
    assert dense_response.results[0].score_breakdown.dense > 0
    assert dense_response.results[0].score_breakdown.vector == dense_response.results[0].score_breakdown.dense
    assert dense_response.results[0].score_breakdown.lexical == 0
    assert dense_response.results[0].score_breakdown.sparse_view == 0


def test_skill_search_request_rejects_top_k_in_structured_intent() -> None:
    with pytest.raises(ValidationError, match="top_k"):
        SkillSearchRequest(query="extract text from a PDF", top_k=1)


def test_dense_first_weights_dominate_semantic_matches() -> None:
    skills = load_skills(SKILL_DIR)
    response = SkillSearcher(
        skills,
        embedder=FakeSemanticEmbedder(),
        dense_enabled=True,
        bm25_enabled=True,
        sparse_view_enabled=True,
    ).search(SkillSearchRequest(query="portable"), top_k=1)

    assert response.results[0].id == "pdf.extract_text"
    assert response.results[0].score > 1.0
    assert response.results[0].score_breakdown.dense > 0
    assert response.results[0].score_breakdown.lexical == 0


def test_capability_match_flows_through_dense_capability_view() -> None:
    skills = load_skills(SKILL_DIR)
    response = SkillSearcher(
        skills,
        embedder=FakeSemanticEmbedder(),
        dense_enabled=True,
        bm25_enabled=False,
        sparse_view_enabled=False,
    ).search(
        SkillSearchRequest(query="handle this", required_capabilities=["extract method"]),
        top_k=1,
    )

    assert response.results[0].id == "research.paper_claim_method_finding"
    assert response.results[0].score_breakdown.dense_capability > 0


def test_dense_capability_view_prevents_premature_required_capability_abstention() -> None:
    skills = load_skills(SKILL_DIR)
    response = SkillSearcher(
        skills,
        embedder=FakeSemanticEmbedder(),
        dense_enabled=True,
        bm25_enabled=False,
        sparse_view_enabled=False,
    ).search(
        SkillSearchRequest(query="handle this", required_capabilities=["contribution evidence"]),
        top_k=1,
    )

    assert response.abstained is False
    assert response.results[0].id == "research.paper_claim_method_finding"


def test_search_hard_abstains_when_required_capability_is_missing() -> None:
    skills = load_skills(SKILL_DIR)
    response = SkillSearcher(skills).search(
        SkillSearchRequest(query="handle this", required_capabilities=["send email newsletter"]),
        top_k=5,
    )

    assert response.abstained is True
    assert response.abstention_reason == "required_capability_miss"
    assert response.results == []


def test_usage_examples_flow_through_dense_usage_view() -> None:
    skills = load_skills(SKILL_DIR)
    response = SkillSearcher(
        skills,
        embedder=FakeSemanticEmbedder(),
        dense_enabled=True,
        bm25_enabled=False,
        sparse_view_enabled=False,
    ).search(
        SkillSearchRequest(
            query="handle this",
            positive_signals=["What is the main contribution of this paper?"],
        ),
        top_k=1,
    )

    assert response.results[0].id == "research.paper_claim_method_finding"
    assert response.results[0].score_breakdown.dense_usage > 0


def test_negative_signals_suppress_strong_dense_matches() -> None:
    skills = load_skills(SKILL_DIR)
    response = SkillSearcher(
        skills,
        embedder=FakeSemanticEmbedder(),
        dense_enabled=True,
        bm25_enabled=True,
        sparse_view_enabled=True,
    ).search(
        SkillSearchRequest(
            query="extract text from PDF",
            negative_signals=["OCR screenshot image"],
        ),
        top_k=5,
    )

    assert all(card.id != "pdf.extract_text" for card in response.results)


def test_search_top_k_is_configured_outside_structured_request() -> None:
    skills = load_skills(SKILL_DIR)
    response = SkillSearcher(skills).search(SkillSearchRequest(query="paper PDF analysis"), top_k=1)

    assert len(response.results) == 1


def test_build_searcher_uses_cli_minimum_score_threshold_over_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[search]
minimum_score_threshold = 99.0
""",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config=str(config_path),
        retrieval_mode="bm25",
        embedding_backend=None,
        embedding_model=None,
        embedding_batch_size=None,
        embedding_max_length=None,
        embedding_device=None,
        embedding_cache_dir=None,
        minimum_score_threshold=1.15,
    )

    searcher = build_searcher(load_skills(SKILL_DIR), args)

    assert searcher.minimum_score_threshold == 1.15


def test_hybrid_combines_dense_and_sparse_rank_lists() -> None:
    skills = load_skills(SKILL_DIR)
    response = SkillSearcher(
        skills,
        embedder=FakeSemanticEmbedder(),
        dense_enabled=True,
    ).search(SkillSearchRequest(query="portable document"), top_k=1)

    assert response.results[0].id == "pdf.extract_text"
    assert response.results[0].score_breakdown.dense > 0
    assert response.results[0].score_breakdown.sparse_view > 0
    assert response.results[0].score_breakdown.rrf > 0


def test_request_capability_and_type_hints_influence_search() -> None:
    skills = load_skills(SKILL_DIR)
    searcher = SkillSearcher(skills)

    paper_response = searcher.search(
        SkillSearchRequest(
            query="handle this",
            required_capabilities=["extract_method"],
            input_types=["paper_text"],
            output_types=["structured_text"],
        ),
        top_k=1,
    )
    pdf_response = searcher.search(
        SkillSearchRequest(
            query="handle this",
            required_capabilities=["read_pdf"],
            input_types=["pdf"],
            output_types=["json"],
        ),
        top_k=1,
    )

    assert paper_response.results[0].id == "research.paper_claim_method_finding"
    assert paper_response.results[0].score_breakdown.capability > 0
    assert paper_response.results[0].score_breakdown.input_type > 0
    assert paper_response.results[0].score_breakdown.output_type > 0
    assert pdf_response.results[0].id == "pdf.extract_text"


def test_contraindication_remains_stronger_than_hybrid_positive_matches() -> None:
    skills = load_skills(SKILL_DIR)
    response = SkillSearcher(skills).search(
        SkillSearchRequest(query="extract text from screenshot OCR image", input_types=["image"]),
        top_k=5,
    )

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


def test_registry_persists_dense_embedding_metadata_and_files() -> None:
    pytest.importorskip("_sqlite3")
    skills = load_skills(SKILL_DIR)
    db_dir = Path(__file__).parent / ".tmp"
    db_dir.mkdir(exist_ok=True)
    run_id = uuid4().hex
    db_path = db_dir / f"skills-{run_id}.db"
    index_dir = db_dir / f"indexes-{run_id}"

    try:
        rebuild_registry(skills, db_path)
        result = persist_dense_embeddings(skills, db_path, index_dir, FakeSemanticEmbedder())
        summary = registry_summary(db_path)
    finally:
        db_path.unlink(missing_ok=True)
        (index_dir / "dense_views.jsonl").unlink(missing_ok=True)
        (index_dir / "dense_views_id_map.json").unlink(missing_ok=True)
        try:
            index_dir.rmdir()
        except OSError:
            pass

    assert result["embedding_count"] >= 10
    assert summary["skill_embeddings"] == result["embedding_count"]


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


def test_retrieval_examples_use_expected_skill_ids() -> None:
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
