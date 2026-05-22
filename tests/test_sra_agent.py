from __future__ import annotations

import json
import sys
from pathlib import Path

import benchmarks.sra_agent as sra_agent_module
from benchmarks.sra import sra_corpus_to_specs
from benchmarks.sra_agent import (
    SEARCH_SKILL_DESCRIPTION,
    SEARCH_DECISION_SYSTEM_PROMPT,
    SRAGeneralPurposeAgent,
    SRASearchDecisionAgent,
    build_sra_prompt,
    compute_sra_agent_skill_metrics,
    parse_sra_search_decision,
    run_sra_agent_inference,
    run_sra_search_decision_inference,
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


def _agent(llm: MockLLMClient) -> SRAGeneralPurposeAgent:
    corpus = _corpus()
    return SRAGeneralPurposeAgent(
        searcher=SkillSearcher(sra_corpus_to_specs(corpus)),
        corpus=corpus,
        llm=llm,
        model_name="mock-model",
        top_k=2,
        max_rounds=4,
    )


class _FakeSolveResult:
    def __init__(self, raw_output: str, skill_ids_used: list[str] | None = None):
        self.raw_output = raw_output
        self.transcript = "fake solve transcript"
        self.skill_ids_used = skill_ids_used or []
        self.meta = {"fake": True}


class _FakeSolveEngine:
    def __init__(self):
        self.calls: list[tuple[dict, list[dict], object, str]] = []

    def run(self, instance, skills, client, model, **kwargs):
        self.calls.append((instance, skills, client, model))
        return _FakeSolveResult(
            "Therefore, the answer is 0.5",
            [skill["skill_id"] for skill in skills],
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


def test_sra_agent_repairs_malformed_json_without_crashing() -> None:
    llm = MockLLMClient(['{"tool":"skill_search","operation":"search",}'])

    record = _agent(llm).run_instance(_instance())

    assert record.error is None
    assert record.raw_output
    assert record.meta["parse_errors"] == []
    assert record.meta["search_call_count"] == 1


def test_sra_agent_ignores_model_supplied_type_gates_for_search() -> None:
    llm = MockLLMClient(
        [
            (
                '{"tool":"skill_search","operation":"search",'
                '"retrieval_intent":{"query":"Bayes rule",'
                '"input_types":["equation"],"output_types":["solution"]}}'
            ),
            '{"final_answer":"Therefore, the answer is 0.5"}',
        ]
    )

    record = _agent(llm).run_instance(_instance())

    assert record.error is None
    assert record.meta["search_call_count"] == 1
    assert '"candidates": [' in (record.transcript or "")
    assert '"skill_id": "theoremqa_001"' in (record.transcript or "")


def test_sra_agent_prompt_encourages_search_before_final_answer() -> None:
    llm = MockLLMClient(["Therefore, the answer is 0.5"])

    _agent(llm).run_instance(_instance())

    system_prompt = llm.calls[0][0]["content"]
    assert SEARCH_SKILL_DESCRIPTION in system_prompt
    assert "Treat skill_search as your default first move" in system_prompt
    assert "Directly answering without searching should be rare" in system_prompt
    assert "prefer calling operation=load_skill" in system_prompt


def test_sra_agent_uses_dataset_native_sra_reasoning_prompts() -> None:
    theorem_system, theorem_user = build_sra_prompt(_instance())
    logic_system, logic_user = build_sra_prompt(
        {
            "instance_id": "logic1",
            "dataset": "logicbench",
            "question": "Does P entail Q?",
            "skill_annotations": ["logicbench_001"],
        }
    )
    champ_system, champ_user = build_sra_prompt(
        {
            "instance_id": "champ1",
            "dataset": "champ",
            "question": "Compute 2 + 2.",
            "skill_annotations": ["champ_001"],
        }
    )

    assert "science teacher" in theorem_system
    assert theorem_user.startswith("Problem:")
    assert logic_system == ""
    assert logic_user == "Does P entail Q?"
    assert "expert on mathematics" in champ_system
    assert "ANSWER: <your answer>" in champ_user


def test_search_decision_parser_accepts_wrapped_search_and_plain_skip() -> None:
    search, search_error = parse_sra_search_decision(
        'This needs Bayes rule.\n<tool>\n{"operation":"search","retrieval_intent":{"query":"Bayes"}}\n</tool>'
    )
    skip, skip_error = parse_sra_search_decision("This is ordinary reasoning, so no search is needed.")

    assert search_error is None
    assert search and search["tool"] == "skill_search"
    assert skip_error is None
    assert skip is None


def test_search_decision_parser_repairs_json_tool_call(monkeypatch) -> None:
    monkeypatch.setattr(
        sra_agent_module,
        "repair_json",
        lambda _text: '{"operation":"search","retrieval_intent":{"query":"Bayes"}}',
    )

    search, search_error = parse_sra_search_decision(
        '<tool>{"operation":"search","retrieval_intent":{"query":"Bayes",}}</tool>'
    )

    assert search_error is None
    assert search and search["operation"] == "search"
    assert search["retrieval_intent"]["query"] == "Bayes"


def test_search_decision_prompt_uses_retrieval_only_schema() -> None:
    assert '"tool":"skill_search"' not in SEARCH_DECISION_SYSTEM_PROMPT
    assert '"desired_capabilities"' not in SEARCH_DECISION_SYSTEM_PROMPT
    assert '"negative_signals"' not in SEARCH_DECISION_SYSTEM_PROMPT
    assert '"constraints"' not in SEARCH_DECISION_SYSTEM_PROMPT
    assert '"query"' in SEARCH_DECISION_SYSTEM_PROMPT
    assert '"task_context"' in SEARCH_DECISION_SYSTEM_PROMPT
    assert '"required_capabilities"' in SEARCH_DECISION_SYSTEM_PROMPT
    assert '"positive_signals"' in SEARCH_DECISION_SYSTEM_PROMPT


def test_search_decision_agent_searches_then_delegates_to_sra_engine() -> None:
    llm = MockLLMClient(
        [
            (
                "This calls for Bayes rule.\n"
                '<tool>{"tool":"skill_search","operation":"search",'
                '"retrieval_intent":{"query":"Bayes rule"}}</tool>'
            )
        ]
    )
    engine = _FakeSolveEngine()
    agent = SRASearchDecisionAgent(
        searcher=SkillSearcher(sra_corpus_to_specs(_corpus())),
        corpus=_corpus(),
        decision_llm=llm,
        solve_engine=engine,
        solve_client=object(),
        model_name="mock-model",
        top_k=2,
        solve_engine_name="direct",
    )

    record = agent.run_instance(_instance())

    assert record.raw_output == "Therefore, the answer is 0.5"
    assert record.meta["search_call_count"] == 1
    assert record.meta["route"] == "search"
    assert record.meta["retrieved_skill_ids"][0] == "theoremqa_001"
    assert record.skill_ids_used[0] == "theoremqa_001"
    assert engine.calls[0][1][0]["skill_id"] == "theoremqa_001"
    assert SEARCH_DECISION_SYSTEM_PROMPT in llm.calls[0][0]["content"]
    assert "first reason briefly" in llm.calls[0][0]["content"]
    assert "<tool>" in llm.calls[0][0]["content"]
    assert "SRA-Bench dataset context for retrieval decision:" in llm.calls[0][0]["content"]
    assert "TheoremQA tasks often involve science" in llm.calls[0][0]["content"]
    assert "Therefore, the answer is" not in llm.calls[0][0]["content"]
    assert llm.calls[0][1]["content"].startswith("Problem:")


def test_search_decision_agent_can_skip_search_and_still_solve() -> None:
    llm = MockLLMClient(["This is simple enough to solve without a skill search."])
    engine = _FakeSolveEngine()
    agent = SRASearchDecisionAgent(
        searcher=SkillSearcher(sra_corpus_to_specs(_corpus())),
        corpus=_corpus(),
        decision_llm=llm,
        solve_engine=engine,
        solve_client=None,
        model_name="mock-model",
        top_k=2,
    )

    record = agent.run_instance(_instance())

    assert record.raw_output == "Therefore, the answer is 0.5"
    assert record.meta["search_call_count"] == 0
    assert record.meta["route"] == "skip"
    assert record.skill_ids_used == []
    assert engine.calls[0][1] == []


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


def test_sra_search_decision_inference_jsonl_schema(tmp_path: Path) -> None:
    output = tmp_path / "decision.jsonl"
    llm = MockLLMClient(
        ['<tool>{"tool":"skill_search","operation":"search","retrieval_intent":{"query":"Bayes rule"}}</tool>']
    )

    result = run_sra_search_decision_inference(
        searcher=SkillSearcher(sra_corpus_to_specs(_corpus())),
        corpus=_corpus(),
        instances_path=_write_instances(tmp_path, [_instance()]),
        output_path=output,
        decision_llm=llm,
        solve_engine=_FakeSolveEngine(),
        solve_client=None,
        model_name="mock-model",
        top_k=2,
        solve_engine_name="direct",
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["instance_id"] == "q1"
    assert rows[0]["raw_output"] == "Therefore, the answer is 0.5"
    assert rows[0]["meta"]["solve_engine"] == "direct"
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
            "--config",
            str(tmp_path / "missing-config.toml"),
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


def test_sra_bench_infer_decision_agent_cli_smoke(tmp_path: Path, capsys) -> None:
    corpus = tmp_path / "corpus.json"
    instances = _write_instances(tmp_path, [_instance()])
    output = tmp_path / "decision-agent.jsonl"
    corpus.write_text(json.dumps(_corpus()), encoding="utf-8")

    exit_code = sra_bench_main(
        [
            "infer-decision-agent",
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
            "--limit",
            "1",
            "--config",
            str(tmp_path / "missing-config.toml"),
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
