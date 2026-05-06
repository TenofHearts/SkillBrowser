from __future__ import annotations

from pathlib import Path

from skill_search_agent.cli import main
from skill_search_agent.gatewaybench import (
    compare_gatewaybench_selectors,
    load_gatewaybench_lite_dataset,
    make_gatewaybench_hybrid_selector,
)
from skill_search_agent.llm import MockLLMClient
from skill_search_agent.selectors import BaselineLLMToolSelector


GATEWAY_FIXTURE = Path(__file__).parent / "fixtures" / "gatewaybench_lite.jsonl"


def test_gatewaybench_compare_reports_metrics_for_each_selector() -> None:
    examples = load_gatewaybench_lite_dataset(GATEWAY_FIXTURE)
    llm = MockLLMClient(
        [
            '{"ranked_tool_ids": ["gateway.query_payments", "gateway.get_invoice"]}',
            '{"ranked_tool_ids": ["gateway.get_employee", "gateway.get_benefits"]}',
        ]
    )

    result = compare_gatewaybench_selectors(
        examples,
        {
            "hybrid": make_gatewaybench_hybrid_selector,
            "llm-baseline": lambda _example: BaselineLLMToolSelector(llm),
        },
        top_k=2,
    )

    assert result.query_count == 2
    assert set(result.results) == {"hybrid", "llm-baseline"}
    assert result.results["llm-baseline"].parse_failure_count == 0
    assert result.results["llm-baseline"].recall_at_k == 1.0


def test_gatewaybench_compare_cli_smoke(capsys) -> None:
    exit_code = main(
        [
            "eval-gatewaybench-compare",
            "--dataset",
            str(GATEWAY_FIXTURE),
            "--top-k",
            "2",
            "--limit",
            "2",
            "--selector",
            "hybrid",
            "--llm",
            "mock",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"hybrid"' in captured.out
    assert '"query_count": 2' in captured.out
