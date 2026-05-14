from __future__ import annotations

import json
import sys
from pathlib import Path

from benchmarks.sra import sra_corpus_to_specs
from benchmarks.sra_agent import (
    FULL_SEARCH_SKILL_DESCRIPTION,
    SRAGeneralPurposeAgent,
    compute_sra_agent_skill_metrics,
    run_sra_agent_inference,
)
from core.search import SkillSearcher
from llm import MockLLMClient


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from sra_bench import main as sra_bench_main  # noqa: E402


def _corpus() -> list[dict]:
    return [
        {
            "skill_id": "theoremqa_001",
            "name": "Apply Bayes rule",
            "description": "Use Bayes rule for conditional probability.",
            "content": "Use P(A|B) = P(B|A)P(A)/P(B).",
        },
        {
            "skill_id": "logicbench_002",
            "name": "Check syllogisms",
            "description": "Reason about symbolic logic.",
            "content": "Translate each statement into symbolic form.",
        },
    ]


def _instance() -> dict:
    return {
        "instance_id": "q1",
        "dataset": "theoremqa",
        "question": "Use Bayes rule to solve this tiny problem.",
        "skill_annotations": ["theoremqa_001"],
        "eval_data": {"answer": "0.5"},
    }


def _agent(llm: MockLLMClient, *, inject_full_search_skill: bool = False) -> SRAGeneralPurposeAgent:
    corpus = _corpus()
    return SRAGeneralPurposeAgent(
        searcher=SkillSearcher(sra_corpus_to_specs(corpus)),
        corpus=corpus,
        llm=llm,
        model_name="mock-model",
        top_k=2,
        max_rounds=4,
        inject_full_search_skill=inject_full_search_skill,
    )


def test_sra_agent_solves_without_search() -> None:
    llm = MockLLMClient(["Therefore, the answer is 0.5"])

    record = _agent(llm).run_instance(_instance())

    assert record.raw_output == "Therefore, the answer is 0.5"
    assert record.skill_ids_used == []
    assert record.meta["search_call_count"] == 0
    assert record.meta["recalled_gold"] is False


def test_sra_agent_searches_loads_and_records_recall() -> None:
    llm = MockLLMClient(
        [
            '{"tool":"skill_search","operation":"search","retrieval_intent":{"query":"Bayes rule"}}',
            '{"tool":"skill_search","operation":"load_skill","skill_id":"theoremqa_001","section":"overview"}',
            '{"final_answer":"Therefore, the answer is 0.5"}',
        ]
    )

    record = _agent(llm).run_instance(_instance())

    assert record.raw_output == "Therefore, the answer is 0.5"
    assert record.skill_ids_used == ["theoremqa_001"]
    assert record.meta["search_call_count"] == 1
    assert record.meta["load_call_count"] == 1
    assert record.meta["recalled_gold"] is True
    assert "skill_search result" in (record.transcript or "")


def test_sra_agent_records_malformed_json_without_crashing() -> None:
    llm = MockLLMClient(['{"tool":"skill_search","operation":"search",}'])

    record = _agent(llm).run_instance(_instance())

    assert record.error is None
    assert record.raw_output
    assert record.meta["parse_errors"]


def test_sra_agent_supports_full_search_skill_prompt_injection() -> None:
    llm = MockLLMClient(["Therefore, the answer is 0.5"])

    _agent(llm, inject_full_search_skill=True).run_instance(_instance())

    assert llm.calls
    assert FULL_SEARCH_SKILL_DESCRIPTION in llm.calls[0][0]["content"]
    assert '"operation":"load_skill"' in llm.calls[0][0]["content"]


def test_sra_agent_inference_jsonl_schema_and_metrics(tmp_path: Path) -> None:
    output = tmp_path / "inference.jsonl"
    llm = MockLLMClient(
        [
            '{"tool":"skill_search","operation":"search","retrieval_intent":{"query":"Bayes rule"}}',
            '{"tool":"skill_search","operation":"load_skill","skill_id":"theoremqa_001"}',
            "Therefore, the answer is 0.5",
        ]
    )

    result = run_sra_agent_inference(
        searcher=SkillSearcher(sra_corpus_to_specs(_corpus())),
        corpus=_corpus(),
        instances_path=_write_instances(tmp_path, [_instance()]),
        output_path=output,
        llm=llm,
        model_name="mock-model",
        top_k=2,
        max_rounds=4,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["instance_id"] == "q1"
    assert rows[0]["raw_output"] == "Therefore, the answer is 0.5"
    assert rows[0]["skill_ids_used"] == ["theoremqa_001"]
    assert result["metrics"]["skill_recall_when_searched"] == 1.0


def test_compute_sra_agent_skill_metrics_only_counts_recall_when_searched() -> None:
    records = [
        {"instance_id": "q1", "skill_ids_used": ["theoremqa_001"], "meta": {"search_call_count": 1}},
        {"instance_id": "q2", "raw_output": "direct", "meta": {"search_call_count": 0}},
    ]
    instances = [
        _instance(),
        {
            "instance_id": "q2",
            "dataset": "theoremqa",
            "question": "Direct question",
            "skill_annotations": ["theoremqa_999"],
        },
    ]

    metrics = compute_sra_agent_skill_metrics(records, instances)

    assert metrics["search_call_rate"] == 0.5
    assert metrics["skill_recall_when_searched"] == 1.0
    assert metrics["skill_recall_at_loaded"] == 1.0


def test_sra_bench_infer_agent_cli_smoke(tmp_path: Path, capsys) -> None:
    corpus = tmp_path / "corpus.json"
    instances = _write_instances(tmp_path, [_instance()])
    output = tmp_path / "agent.jsonl"
    corpus.write_text(json.dumps(_corpus()), encoding="utf-8")

    exit_code = sra_bench_main(
        [
            "infer-agent",
            "--corpus",
            str(corpus),
            "--instances",
            str(instances),
            "--inference-output",
            str(output),
            "--llm",
            "mock",
            "--model",
            "mock-model",
            "--max-rounds",
            "1",
            "--limit",
            "1",
            "--retrieval-mode",
            "bm25",
            "--embedding-backend",
            "none",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output.exists()
    assert '"inference_output"' in captured.out


def _write_instances(tmp_path: Path, instances: list[dict]) -> Path:
    path = tmp_path / "instances.json"
    path.write_text(json.dumps(instances), encoding="utf-8")
    return path
