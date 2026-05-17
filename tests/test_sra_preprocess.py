from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from benchmarks.sra_preprocess import (
    SraSkillMetadata,
    deterministic_sra_metadata,
    load_sra_preprocess_checkpoint,
    parse_sra_metadata,
    run_sra_preprocess,
    sra_skill_to_enriched_spec,
    write_sra_skill_files,
)
from loader import load_skills
from llm import MockLLMClient


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import sra_bench  # noqa: E402
from sra_bench import main as sra_bench_main  # noqa: E402


def test_parse_sra_metadata_accepts_strict_json() -> None:
    metadata = parse_sra_metadata(json.dumps(_metadata_payload()))

    assert metadata.short_description.startswith("Apply Bayes")
    assert metadata.capabilities[0].id == "apply_bayes_rule"


def test_parse_sra_metadata_rejects_non_json_wrappers() -> None:
    with pytest.raises(ValueError, match="strict JSON"):
        parse_sra_metadata('```json\n{"short_description": "wrapped"}\n```')


def test_enriched_spec_validates_and_maps_tool_parameters() -> None:
    spec = sra_skill_to_enriched_spec(_tool_skill(), SraSkillMetadata.parse_obj(_metadata_payload()))

    assert spec.id == "medcalcbench_000"
    assert spec.description.long
    assert spec.capabilities[0].id == "apply_bayes_rule"
    assert spec.input_schema is not None
    assert spec.input_schema["properties"]["age"]["type"] == "integer"
    assert spec.input_schema["properties"]["female"]["type"] == "integer"
    assert spec.output_types == []
    assert spec.when_not_to_use == []
    assert spec.examples.negative == []


def test_write_sra_skill_files_can_be_loaded(tmp_path: Path) -> None:
    write_sra_skill_files(_basic_skill(), SraSkillMetadata.parse_obj(_metadata_payload()), tmp_path)

    skills = load_skills(tmp_path)
    skill_md = (tmp_path / "theoremqa_001" / "skill.md").read_text(encoding="utf-8")

    assert skills[0].id == "theoremqa_001"
    assert skills[0].interaction.default_read_level == "overview"
    assert "Retrieval Profile" not in skill_md
    assert skill_md == "Use P(A|B) = P(B|A)P(A)/P(B)."


def test_run_sra_preprocess_resumes_checkpoint(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    output_dir = tmp_path / "skills"
    checkpoint = tmp_path / "checkpoint.jsonl"
    corpus.write_text(json.dumps([_basic_skill()]), encoding="utf-8")
    checkpoint.write_text(
        json.dumps(
            {
                "skill_id": "theoremqa_001",
                "model": "deepseek-v4-pro",
                "metadata": _metadata_payload(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    llm = MockLLMClient(['{"this": "should not be used"}'])

    result = run_sra_preprocess(
        corpus_path=corpus,
        output_skill_dir=output_dir,
        checkpoint_path=checkpoint,
        llm=llm,
        model_name="deepseek-v4-pro",
        resume=True,
    )

    assert result["processed"] == 1
    assert result["reused"] == 1
    assert llm.calls == []
    assert load_sra_preprocess_checkpoint(checkpoint)["theoremqa_001"].short_description


def test_deterministic_web_metadata_is_stable_and_loadable(tmp_path: Path) -> None:
    first = deterministic_sra_metadata(_web_skill())
    second = deterministic_sra_metadata(_web_skill())

    assert first.dict() == second.dict()
    assert first.short_description
    assert "web" in first.tags

    write_sra_skill_files(_web_skill(), first, tmp_path)
    skills = load_skills(tmp_path)

    assert skills[0].id == "web_00001"
    assert skills[0].category.primary == "web"
    assert skills[0].execution.mode == "none"


def test_run_sra_preprocess_deterministic_web_does_not_call_llm(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    output_dir = tmp_path / "skills"
    checkpoint = tmp_path / "checkpoint.jsonl"
    corpus.write_text(json.dumps([_basic_skill(), _web_skill(), _web_skill("web_00002")]), encoding="utf-8")
    llm = MockLLMClient(['{"this": "should not be used"}'])

    result = run_sra_preprocess(
        corpus_path=corpus,
        output_skill_dir=output_dir,
        checkpoint_path=checkpoint,
        llm=llm,
        model_name="deterministic",
        dataset="web",
        mode="deterministic",
        limit=2,
        resume=True,
    )

    assert result["processed"] == 2
    assert result["failed"] == 0
    assert result["dataset"] == "web"
    assert llm.calls == []
    records = [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()]
    assert {record["skill_id"] for record in records} == {"web_00001", "web_00002"}
    assert all(record["mode"] == "deterministic" for record in records)
    assert len(load_skills(output_dir)) == 2


def test_sra_bench_preprocess_deterministic_web_defaults(tmp_path: Path, monkeypatch) -> None:
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps([_web_skill(), _web_skill("web_00002")]), encoding="utf-8")
    monkeypatch.setattr(sra_bench, "ROOT", tmp_path)

    exit_code = sra_bench_main(
        [
            "preprocess",
            "--corpus",
            str(corpus),
            "--dataset",
            "web",
            "--mode",
            "deterministic",
            "--limit",
            "2",
        ]
    )

    assert exit_code == 0
    assert len(load_skills(tmp_path / "data/eval/sra/web")) == 2
    assert (tmp_path / "data/eval/sra/preprocess/web.jsonl").exists()


def test_sra_bench_retrieve_uses_preprocessed_skill_dir(tmp_path: Path, capsys) -> None:
    corpus = tmp_path / "corpus.json"
    instances = tmp_path / "instances.json"
    output = tmp_path / "retrieval.json"
    skill_dir = tmp_path / "skills"
    corpus.write_text(json.dumps([_basic_skill(), _distractor_skill()]), encoding="utf-8")
    instances.write_text(
        json.dumps(
            [
                {
                    "instance_id": "q1",
                    "dataset": "theoremqa",
                    "question": "Use Bayes theorem for conditional probability.",
                    "skill_annotations": ["theoremqa_001"],
                }
            ]
        ),
        encoding="utf-8",
    )
    metadata = _metadata_payload()
    metadata["positive_examples"][0]["user_query"] = "Use Bayes theorem for conditional probability."
    write_sra_skill_files(_basic_skill(), SraSkillMetadata.parse_obj(metadata), skill_dir)
    write_sra_skill_files(_distractor_skill(), SraSkillMetadata.parse_obj(_distractor_metadata()), skill_dir)

    exit_code = sra_bench_main(
        [
            "retrieve",
            "--corpus",
            str(corpus),
            "--instances",
            str(instances),
            "--retrieval-output",
            str(output),
            "--sra-skill-dir",
            str(skill_dir),
            "--top-k",
            "1",
            "--retrieval-mode",
            "bm25",
            "--embedding-backend",
            "none",
        ]
    )

    captured = capsys.readouterr()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert '"retrieval_output"' in captured.out
    assert result["results"][0]["retrieved"][0]["skill_id"] == "theoremqa_001"


def test_sra_bench_preprocessed_search_uses_metadata_only_dense_views(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_build_searcher(skills, args, *, dense_view_names=None):
        captured["dense_view_names"] = dense_view_names
        return object()

    monkeypatch.setattr(sra_bench, "build_searcher", fake_build_searcher)
    args = type("Args", (), {"sra_skill_dir": str(tmp_path)})()

    sra_bench.build_sra_searcher([], args)

    assert captured["dense_view_names"] == sra_bench.SRA_METADATA_DENSE_VIEW_NAMES
    assert all(not name.startswith("content_section:") for name in captured["dense_view_names"])


def test_sra_bench_retrieve_uses_configured_skill_dirs(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    web_dir = tmp_path / "web"
    config = tmp_path / "config.toml"
    write_sra_skill_files(_basic_skill(), SraSkillMetadata.parse_obj(_metadata_payload()), gold_dir)
    write_sra_skill_files(_web_skill(), deterministic_sra_metadata(_web_skill()), web_dir)
    config.write_text(
        f"""
[sra]
skill_dirs = ["{gold_dir.as_posix()}", "{web_dir.as_posix()}"]
""",
        encoding="utf-8",
    )
    args = type("Args", (), {"sra_skill_dir": None, "config": str(config)})()

    skills = sra_bench.load_sra_search_specs(args, [])

    assert [skill.id for skill in skills] == ["theoremqa_001", "web_00001"]


def test_sra_bench_cli_skill_dir_overrides_configured_skill_dirs(tmp_path: Path) -> None:
    cli_dir = tmp_path / "cli"
    web_dir = tmp_path / "web"
    config = tmp_path / "config.toml"
    write_sra_skill_files(_basic_skill(), SraSkillMetadata.parse_obj(_metadata_payload()), cli_dir)
    write_sra_skill_files(_web_skill(), deterministic_sra_metadata(_web_skill()), web_dir)
    config.write_text(
        f"""
[sra]
skill_dirs = ["{web_dir.as_posix()}"]
""",
        encoding="utf-8",
    )
    args = type("Args", (), {"sra_skill_dir": str(cli_dir), "config": str(config)})()

    skills = sra_bench.load_sra_search_specs(args, [])

    assert [skill.id for skill in skills] == ["theoremqa_001"]


def test_sra_bench_configured_skill_dirs_reject_duplicate_ids(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    write_sra_skill_files(_basic_skill(), SraSkillMetadata.parse_obj(_metadata_payload()), first_dir)
    write_sra_skill_files(_basic_skill(), SraSkillMetadata.parse_obj(_metadata_payload()), second_dir)

    with pytest.raises(Exception, match="Duplicate skill id"):
        sra_bench.load_sra_skill_dirs([str(first_dir), str(second_dir)])


def _metadata_payload() -> dict:
    return {
        "short_description": "Apply Bayes rule for conditional probability questions.",
        "long_description": (
            "Use Bayes theorem to compute posterior or conditional probabilities. "
            "Recognition cues include P(A|B), prior probability, likelihood, and evidence."
        ),
        "capabilities": [
            {
                "id": "apply_bayes_rule",
                "description": "Compute conditional probabilities with Bayes theorem.",
            }
        ],
        "when_to_use": ["The task asks for posterior probability from conditional evidence."],
        "positive_examples": [
            {
                "user_query": "Find P(A|B) using Bayes theorem.",
                "reason": "The query asks for Bayes rule.",
            }
        ],
        "tags": ["bayes", "conditional probability", "theoremqa"],
    }


def _distractor_metadata() -> dict:
    return {
        "short_description": "Check symbolic syllogism conclusions.",
        "long_description": "Use formal logic rules to evaluate syllogism validity.",
        "capabilities": [{"id": "check_syllogism", "description": "Evaluate symbolic logic conclusions."}],
        "when_to_use": ["The task asks whether a logical conclusion follows."],
        "positive_examples": [{"user_query": "Does this syllogism follow?", "reason": "Logic task."}],
        "tags": ["logicbench", "syllogism"],
    }


def _basic_skill() -> dict:
    return {
        "skill_id": "theoremqa_001",
        "name": "Apply Bayes rule",
        "description": "Use Bayes rule for conditional probability.",
        "content": "Use P(A|B) = P(B|A)P(A)/P(B).",
    }


def _distractor_skill() -> dict:
    return {
        "skill_id": "logicbench_002",
        "name": "Check syllogisms",
        "description": "Reason about symbolic logic.",
        "content": "Translate each statement into symbolic form.",
    }


def _tool_skill() -> dict:
    skill = _basic_skill()
    skill["skill_id"] = "medcalcbench_000"
    skill["tools"] = [
        {
            "name": "compute_score",
            "description": "Compute a medical score.",
            "parameters": {"age": "int", "female": "int"},
        }
    ]
    return skill


def _web_skill(skill_id: str = "web_00001") -> dict:
    return {
        "skill_id": skill_id,
        "name": "React performance guidelines",
        "description": "Use this skill to optimize React and Next.js performance patterns.",
        "content": (
            "# React Performance Guidelines\n\n"
            "## When to Use\n\n"
            "- Reviewing React components for slow rendering.\n"
            "- Optimizing Next.js data fetching and bundle size.\n\n"
            "Use memoization, direct imports, and parallel data fetching where appropriate."
        ),
    }
