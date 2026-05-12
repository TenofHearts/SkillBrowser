from __future__ import annotations

from pathlib import Path

from benchmarks.toolret import (
    TextEmbedder,
    build_toolret_first_stage_candidates,
    compare_toolret_hybrid_with_paper_llm,
    evaluate_toolret_llm_rerank,
    evaluate_toolret_paper_llm_baseline,
    evaluate_toolret_retrieval,
    get_toolret_gold_skill_ids,
    load_toolret_first_stage_candidates,
    load_toolret_queries,
    load_toolret_tools,
    score_toolret_ranking,
    toolret_query_to_search_request,
)
from cli import main
from core.search import SkillSearcher
from llm import MockLLMClient


TOOLRET_TOOLS = Path(__file__).parent / "fixtures" / "toolret_tools.jsonl"
TOOLRET_QUERIES = Path(__file__).parent / "fixtures" / "toolret_queries.jsonl"
TOOLRET_CANDIDATES = Path(__file__).parent / "fixtures" / "toolret_candidates.jsonl"


class FakeEmbedder(TextEmbedder):
    model_name = "fake-embedder"

    def embed_queries(self, examples, *, use_instruction: bool):
        return [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
        ][: len(examples)]

    def embed_passages(self, passages):
        return [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.3],
            [0.2, 0.8],
        ][: len(passages)]


def test_toolret_adapter_loads_jsonl_and_preserves_ids() -> None:
    tools = load_toolret_tools(TOOLRET_TOOLS)
    queries = load_toolret_queries(TOOLRET_QUERIES)

    assert [tool.id for tool in tools][:3] == [
        "toolret.web_tool_1",
        "toolret.code_tool_1",
        "toolret.custom_tool_1",
    ]
    assert get_toolret_gold_skill_ids(queries[0]) == ["toolret.web_tool_1"]


def test_toolret_query_request_uses_instruction_as_context() -> None:
    example = load_toolret_queries(TOOLRET_QUERIES, category="code", limit=1)[0]

    with_instruction = toolret_query_to_search_request(example, top_k=10, use_instruction=True)
    without_instruction = toolret_query_to_search_request(example, top_k=10, use_instruction=False)

    assert with_instruction.query == example.query
    assert "population growth" in with_instruction.task_context
    assert without_instruction.task_context is None


def test_toolret_metrics_support_multilabel_rankings() -> None:
    stats = score_toolret_ranking(
        ["toolret.a", "toolret.b"],
        ["toolret.x", "toolret.a", "toolret.y", "toolret.b"],
    )

    assert stats["recall@1"] == 0.0
    assert stats["recall@3"] == 0.5
    assert stats["recall@10"] == 1.0
    assert stats["precision@10"] == 0.2
    assert stats["mrr@10"] == 0.5
    assert stats["completeness@10"] == 1.0
    assert 0.0 < stats["ndcg@10"] < 1.0


def test_toolret_retrieval_eval_smoke() -> None:
    tools = load_toolret_tools(TOOLRET_TOOLS)
    examples = load_toolret_queries(TOOLRET_QUERIES)

    result = evaluate_toolret_retrieval(SkillSearcher(tools), examples, top_k=10, use_instruction=True)

    assert result.query_count == 3
    assert result.recall_at["10"] == 1.0
    assert set(result.by_category) == {"web", "code", "customized"}


def test_toolret_rankgpt_uses_first_stage_candidates() -> None:
    tools = load_toolret_tools(TOOLRET_TOOLS)
    examples = load_toolret_queries(TOOLRET_QUERIES, limit=1)
    first_stage = load_toolret_first_stage_candidates(TOOLRET_CANDIDATES)
    llm = MockLLMClient(['{"ranking": [2, 1]}'])

    result = evaluate_toolret_llm_rerank(
        SkillSearcher(tools),
        tools,
        examples,
        llm,
        top_k=2,
        use_instruction=True,
        candidate_pool_size=2,
        first_stage_candidates=first_stage,
        window_size=2,
        step_size=1,
    )

    assert result.query_count == 1
    assert result.recall_at["1"] == 1.0
    assert result.parse_failure_count == 0
    assert llm.calls
    assert "Instruct:" in llm.calls[0][1]["content"]


def test_toolret_paper_llm_baseline_uses_provided_candidates() -> None:
    tools = load_toolret_tools(TOOLRET_TOOLS)
    examples = load_toolret_queries(TOOLRET_QUERIES, limit=1)
    first_stage = load_toolret_first_stage_candidates(TOOLRET_CANDIDATES)
    llm = MockLLMClient(['{"ranking": [2, 1]}'])

    result = evaluate_toolret_paper_llm_baseline(
        tools,
        examples,
        llm,
        top_k=2,
        use_instruction=True,
        candidate_pool_size=2,
        first_stage_candidates=first_stage,
        window_size=2,
        step_size=1,
    )

    assert result.query_count == 1
    assert result.recall_at["1"] == 1.0
    assert result.parse_failure_count == 0
    assert llm.calls


def test_toolret_compare_hybrid_with_paper_llm_reports_delta() -> None:
    tools = load_toolret_tools(TOOLRET_TOOLS)
    examples = load_toolret_queries(TOOLRET_QUERIES, limit=1)
    first_stage = load_toolret_first_stage_candidates(TOOLRET_CANDIDATES)
    llm = MockLLMClient(['{"ranking": [2, 1]}'])

    result = compare_toolret_hybrid_with_paper_llm(
        SkillSearcher(tools),
        tools,
        examples,
        llm,
        top_k=2,
        use_instruction=True,
        candidate_pool_size=2,
        first_stage_candidates=first_stage,
        window_size=2,
        step_size=1,
    )

    assert result.query_count == 1
    assert result.hybrid.query_count == 1
    assert result.llm.query_count == 1
    assert result.document_representation == "SkillSpec-derived ToolRet candidate documents"
    assert "ndcg@10" in result.delta_llm_minus_hybrid


def test_toolret_build_first_stage_candidates_writes_ranked_jsonl(tmp_path: Path) -> None:
    tools = load_toolret_tools(TOOLRET_TOOLS)
    examples = load_toolret_queries(TOOLRET_QUERIES)
    output = tmp_path / "candidates.jsonl"

    result = build_toolret_first_stage_candidates(
        tools,
        examples,
        FakeEmbedder(),
        top_k=2,
        use_instruction=True,
        output_path=output,
    )
    candidates = load_toolret_first_stage_candidates(output)

    assert result.query_count == 3
    assert result.tool_count == 4
    assert result.model == "fake-embedder"
    assert candidates["apibank_query_0"] == ["toolret.web_tool_1", "toolret.custom_tool_1"]
    assert candidates["apigen_query_5"][0] == "toolret.code_tool_1"


def test_toolret_cli_smoke(capsys) -> None:
    exit_code = main(
        [
            "eval-toolret",
            "--tools",
            str(TOOLRET_TOOLS),
            "--queries",
            str(TOOLRET_QUERIES),
            "--top-k",
            "10",
            "--limit",
            "3",
            "--baseline",
            "hybrid",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"query_count": 3' in captured.out
    assert '"recall_at"' in captured.out


def test_toolret_rankgpt_cli_smoke(capsys) -> None:
    exit_code = main(
        [
            "eval-toolret",
            "--tools",
            str(TOOLRET_TOOLS),
            "--queries",
            str(TOOLRET_QUERIES),
            "--first-stage-candidates",
            str(TOOLRET_CANDIDATES),
            "--top-k",
            "2",
            "--limit",
            "1",
            "--baseline",
            "rankgpt",
            "--llm",
            "mock",
            "--candidate-pool-size",
            "2",
            "--rankgpt-window-size",
            "2",
            "--rankgpt-step-size",
            "1",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"query_count": 1' in captured.out
    assert '"input_tokens"' in captured.out


def test_toolret_paper_rankgpt_cli_requires_first_stage_candidates(capsys) -> None:
    exit_code = main(
        [
            "eval-toolret",
            "--tools",
            str(TOOLRET_TOOLS),
            "--queries",
            str(TOOLRET_QUERIES),
            "--top-k",
            "2",
            "--limit",
            "1",
            "--baseline",
            "toolret-rankgpt",
            "--llm",
            "mock",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "requires --first-stage-candidates" in captured.err


def test_toolret_paper_rankgpt_cli_smoke(capsys) -> None:
    exit_code = main(
        [
            "eval-toolret",
            "--tools",
            str(TOOLRET_TOOLS),
            "--queries",
            str(TOOLRET_QUERIES),
            "--first-stage-candidates",
            str(TOOLRET_CANDIDATES),
            "--top-k",
            "2",
            "--limit",
            "1",
            "--baseline",
            "toolret-rankgpt",
            "--llm",
            "mock",
            "--candidate-pool-size",
            "2",
            "--rankgpt-window-size",
            "2",
            "--rankgpt-step-size",
            "1",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"query_count": 1' in captured.out
    assert '"parse_failure_count"' in captured.out


def test_toolret_compare_cli_smoke(capsys) -> None:
    exit_code = main(
        [
            "eval-toolret",
            "--tools",
            str(TOOLRET_TOOLS),
            "--queries",
            str(TOOLRET_QUERIES),
            "--first-stage-candidates",
            str(TOOLRET_CANDIDATES),
            "--top-k",
            "2",
            "--limit",
            "1",
            "--baseline",
            "compare",
            "--llm",
            "mock",
            "--candidate-pool-size",
            "2",
            "--rankgpt-window-size",
            "2",
            "--rankgpt-step-size",
            "1",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"hybrid"' in captured.out
    assert '"llm"' in captured.out
    assert '"delta_llm_minus_hybrid"' in captured.out
