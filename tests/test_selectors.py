from __future__ import annotations

from pathlib import Path

from skill_search_agent.gatewaybench import (
    gateway_example_to_candidate_tools,
    gateway_example_to_skills,
    load_gatewaybench_lite_dataset,
)
from skill_search_agent.llm import MockLLMClient
from skill_search_agent.schema import ToolSelectionRequest
from skill_search_agent.search import SkillSearcher
from skill_search_agent.selectors import BaselineLLMToolSelector, HybridSearchToolSelector, parse_selection_json


GATEWAY_FIXTURE = Path(__file__).parent / "fixtures" / "gatewaybench_lite.jsonl"


def test_hybrid_selector_returns_gatewaybench_skill_ids() -> None:
    example = load_gatewaybench_lite_dataset(GATEWAY_FIXTURE, limit=1)[0]
    selector = HybridSearchToolSelector(SkillSearcher(gateway_example_to_skills(example)))

    result = selector.select(
        ToolSelectionRequest(
            prompt=example.user_prompt,
            candidates=gateway_example_to_candidate_tools(example),
            top_k=2,
        )
    )

    assert result.ranked_tool_ids
    assert all(tool_id.startswith("gateway.") for tool_id in result.ranked_tool_ids)


def test_llm_selector_parses_json_and_hides_relevance_labels() -> None:
    example = load_gatewaybench_lite_dataset(GATEWAY_FIXTURE, limit=1)[0]
    llm = MockLLMClient(['{"ranked_tool_ids": ["gateway.query_payments", "gateway.get_invoice"]}'])
    selector = BaselineLLMToolSelector(llm)

    result = selector.select(
        ToolSelectionRequest(
            prompt=example.user_prompt,
            candidates=gateway_example_to_candidate_tools(example),
            top_k=2,
        )
    )
    prompt_text = "\n".join(message["content"] for message in llm.calls[0])

    assert result.ranked_tool_ids == ["gateway.query_payments", "gateway.get_invoice"]
    assert "required" not in prompt_text
    assert "irrelevant" not in prompt_text
    assert "ideal_tool_subset" not in prompt_text


def test_llm_selector_reports_malformed_output() -> None:
    ranked, error = parse_selection_json("not json", {"gateway.query_payments"})

    assert ranked == []
    assert error is not None


def test_llm_selector_parses_fenced_json_output() -> None:
    ranked, error = parse_selection_json(
        '```json\n{"ranked_tool_ids": ["gateway.query_payments"]}\n```',
        {"gateway.query_payments"},
    )

    assert ranked == ["gateway.query_payments"]
    assert error is None
