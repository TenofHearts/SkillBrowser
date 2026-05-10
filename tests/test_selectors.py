from __future__ import annotations

from pathlib import Path

from core.search import SkillSearcher
from core.selectors import (
    BaselineLLMToolSelector,
    HybridSearchToolSelector,
    parse_selection_json,
    parse_skill_search_decision,
)
from loader import load_skills
from llm import MockLLMClient
from schema import CandidateTool, ToolSelectionRequest


SKILL_DIR = Path(__file__).parent / "fixtures" / "skills"


def test_hybrid_selector_returns_retrieved_skill_ids() -> None:
    skills = load_skills(SKILL_DIR)
    llm = MockLLMClient(
        [
            '{"action": "skill_search"}',
            '{"ranked_tool_ids": ["pdf.extract_text"]}',
        ]
    )
    selector = HybridSearchToolSelector(SkillSearcher(skills), llm)

    result = selector.select(
        ToolSelectionRequest(
            prompt="extract text from a PDF",
            candidates=[],
            top_k=2,
        )
    )

    assert result.ranked_tool_ids == ["pdf.extract_text"]
    assert len(llm.calls) == 2
    selection_prompt = llm.calls[1][1]["content"]
    assert "retrieval_rank" in selection_prompt


def test_llm_selector_parses_json() -> None:
    candidates = [
        CandidateTool(id="pdf.extract_text", name="PDF Extract Text", description="Extract text from PDFs."),
        CandidateTool(id="csv.read", name="CSV Read", description="Read CSV files."),
    ]
    llm = MockLLMClient(['{"ranked_tool_ids": ["pdf.extract_text"]}'])
    selector = BaselineLLMToolSelector(llm)

    result = selector.select(
        ToolSelectionRequest(
            prompt="extract text from a PDF",
            candidates=candidates,
            top_k=2,
        )
    )

    assert result.ranked_tool_ids == ["pdf.extract_text"]


def test_llm_selector_reports_malformed_output() -> None:
    ranked, error = parse_selection_json("not json", {"pdf.extract_text"})

    assert ranked == []
    assert error is not None


def test_llm_selector_parses_fenced_json_output() -> None:
    ranked, error = parse_selection_json(
        '```json\n{"ranked_tool_ids": ["pdf.extract_text"]}\n```',
        {"pdf.extract_text"},
    )

    assert ranked == ["pdf.extract_text"]
    assert error is None


def test_hybrid_selector_parses_skill_search_decision() -> None:
    query, error = parse_skill_search_decision('{"action": "skill_search"}', default_query="payments and invoices")

    assert query == "payments and invoices"
    assert error is None
