from __future__ import annotations

from pathlib import Path

from agent import SkillAgent
from cli import main
from core.search import SkillSearcher
from llm import MockLLMClient
from loader import load_skills
from reader import SkillReader
from schema import AgentRunRequest


SKILL_DIR = Path(__file__).parent / "fixtures" / "skills"


def test_agent_searches_reads_and_returns_final_answer() -> None:
    skills = load_skills(SKILL_DIR)
    llm = MockLLMClient(
        [
            '{"action": "skill_search", "query": "extract text from a PDF"}',
            '{"selected_ids": ["pdf.extract_text"], "reason": "PDF text extraction is needed."}',
            '{"final_answer": "Use the PDF extraction skill before downstream analysis."}',
        ]
    )

    result = SkillAgent(SkillSearcher(skills), SkillReader(skills), llm).run(
        AgentRunRequest(task="extract text from a PDF", top_k=2)
    )

    assert result.selected_skill_ids == ["pdf.extract_text"]
    assert result.read_skill_ids == ["pdf.extract_text"]
    assert result.final_answer == "Use the PDF extraction skill before downstream analysis."
    assert [step.action for step in result.steps] == [
        "llm_decide_tools",
        "skill_search",
        "llm_choose_skill",
        "skill_read",
        "llm_final_answer",
    ]


def test_agent_records_parse_errors_and_falls_back_to_top_search_result() -> None:
    skills = load_skills(SKILL_DIR)
    llm = MockLLMClient(
        [
            '{"action": "skill_search", "query": "extract text from a PDF"}',
            "not json",
            '{"final_answer": "Fallback still produced an answer."}',
        ]
    )

    result = SkillAgent(SkillSearcher(skills), SkillReader(skills), llm).run(
        AgentRunRequest(task="extract text from a PDF", top_k=1)
    )

    assert result.selected_skill_ids == ["pdf.extract_text"]
    assert result.parse_errors
    assert result.steps[2].error is not None
    assert result.final_answer == "Fallback still produced an answer."


def test_run_agent_cli_smoke(capsys) -> None:
    exit_code = main(
        [
            "run-agent",
            "--skill-dir",
            str(SKILL_DIR),
            "extract text from a PDF",
            "--top-k",
            "1",
            "--llm",
            "mock",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"task": "extract text from a PDF"' in captured.out
    assert '"skill_search"' in captured.out
