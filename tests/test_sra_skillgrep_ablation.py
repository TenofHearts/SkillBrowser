from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import sra_skillgrep_ablation as ablation  # noqa: E402
from benchmarks.sra_preprocess import SraSkillMetadata, write_sra_skill_files  # noqa: E402


def test_best_gold_hit_uses_best_ranked_gold() -> None:
    retrieved = [
        {"rank": 1, "skill_id": "x", "score": 3.0},
        {"rank": 2, "skill_id": "b", "score": 2.0},
        {"rank": 3, "skill_id": "a", "score": 1.0},
    ]

    best = ablation.best_gold_hit(["a", "b"], retrieved)

    assert best is not None
    assert best["skill_id"] == "b"
    assert best["rank"] == 2
    assert best["score"] == 2.0


def test_compute_ablation_metrics_handles_missed_gold() -> None:
    records = [
        {
            "instance_id": "q1",
            "gold_skill_ids": ["gold"],
            "retrieved": [{"rank": 1, "skill_id": "other", "score": 1.0}],
            "gold_hit_at_5": False,
            "best_gold_rank": None,
            "best_gold_score": None,
        }
    ]

    metrics = ablation.compute_ablation_metrics(records, top_k=5)

    assert metrics["GoldHit@5"] == 0.0
    assert metrics["GoldMRR@5"] == 0.0
    assert metrics["GoldMeanRank@5"] is None
    assert metrics["GoldMeanScore@5"] is None


def test_ablation_variants_disable_expected_components() -> None:
    args = argparse.Namespace(
        config="missing.toml",
        weight_lexical=None,
        weight_sparse_view=None,
        weight_dense=None,
        weight_rrf=None,
        weight_capability=None,
        weight_usage=None,
        weight_input_type=None,
        weight_output_type=None,
        weight_penalty=None,
    )

    variants = {variant.name: variant for variant in ablation.build_ablation_variants(args)}

    assert variants["bm25_only"].bm25_enabled is True
    assert variants["bm25_only"].dense_enabled is False
    assert variants["bm25_only"].sparse_view_enabled is False
    assert variants["bm25_only"].weights.dense == 0.0
    assert variants["dense_only"].bm25_enabled is False
    assert variants["dense_only"].dense_enabled is True
    assert variants["sparse_view_only"].sparse_view_enabled is True
    assert variants["sparse_view_only"].weights.lexical == 0.0
    assert variants["no_rrf_score"].weights.rrf == 0.0
    assert variants["no_penalty"].weights.penalty == 0.0


def test_skillgrep_ablation_smoke_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    corpus = tmp_path / "corpus.json"
    instances_dir = tmp_path / "instances"
    skill_dir = tmp_path / "skills"
    output_dir = tmp_path / "out"
    instances_dir.mkdir()
    corpus.write_text(json.dumps([_basic_skill(), _distractor_skill()]), encoding="utf-8")
    instances_dir.joinpath("theoremqa.json").write_text(
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
    monkeypatch.setattr(ablation, "build_sra_query", lambda instance, sra_repo: str(instance["question"]))

    result = ablation.main(
        [
            "--datasets",
            "theoremqa",
            "--variants",
            "bm25_only",
            "--corpus",
            str(corpus),
            "--sra-skill-dir",
            str(skill_dir),
            "--instances-dir",
            str(instances_dir),
            "--output-dir",
            str(output_dir),
            "--embedding-backend",
            "fake",
            "--minimum-score-threshold",
            "0.0",
            "--limit",
            "1",
        ]
    )

    assert result == 0
    summary = json.loads((output_dir / "skillgrep_component_ablation_summary.json").read_text(encoding="utf-8"))
    details = (output_dir / "skillgrep_component_ablation_details.jsonl").read_text(encoding="utf-8").splitlines()
    assert summary["metadata"]["variants"] == ["bm25_only"]
    assert summary["summary_rows"]
    assert details
    assert (output_dir / "skillgrep_component_ablation_summary.csv").exists()
    assert (output_dir / "skillgrep_component_ablation_summary.md").exists()
    assert (output_dir / "skillgrep_component_ablation_summary.tex").exists()


def _metadata_payload() -> dict:
    return {
        "short_description": "Apply Bayes rule for conditional probability questions.",
        "long_description": "Use Bayes theorem to compute conditional probabilities.",
        "capabilities": [{"id": "apply_bayes_rule", "description": "Compute Bayes theorem probabilities."}],
        "when_to_use": ["The task asks for posterior probability from conditional evidence."],
        "positive_examples": [{"user_query": "Find P(A|B) using Bayes theorem.", "reason": "Bayes rule."}],
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
