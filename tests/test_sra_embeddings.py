from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from benchmarks.sra_preprocess import SraSkillMetadata, write_sra_skill_files
from core.search import dense_cache_path, read_dense_cache


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import sra_build_embeddings  # noqa: E402


def test_sra_build_embeddings_writes_cache_checkpoint_and_log(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills"
    cache_dir = tmp_path / "cache"
    checkpoint = tmp_path / "checkpoint.jsonl"
    log = tmp_path / "log.jsonl"
    config = tmp_path / "config.toml"
    write_sra_skill_files(_skill(), SraSkillMetadata.parse_obj(_metadata()), skill_dir)
    config.write_text(
        f"""
[embedding]
enabled = true
backend = "fake"
model = "fake-model"
batch_size = 2
max_length = 128
cache_dir = "{cache_dir.as_posix()}"

[sra]
skill_dirs = ["{skill_dir.as_posix()}"]
""",
        encoding="utf-8",
    )

    exit_code = sra_build_embeddings.main_with_args(
        [
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--log",
            str(log),
            "--view-name",
            "description",
        ]
    )

    assert exit_code == 0
    checkpoint_records = [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()]
    log_records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert checkpoint_records[-1]["status"] == "built"
    assert any(record["event"] == "embed_done" for record in log_records)
    cache_path = Path(checkpoint_records[-1]["cache_path"])
    assert read_dense_cache(cache_path) is not None


def test_sra_agent_helper_requires_existing_embedding_cache(tmp_path: Path) -> None:
    helper = _load_agent_helper()
    skill_dir = tmp_path / "skills"
    cache_dir = tmp_path / "cache"
    config = tmp_path / "config.toml"
    write_sra_skill_files(_skill(), SraSkillMetadata.parse_obj(_metadata()), skill_dir)
    config.write_text(
        f"""
[embedding]
enabled = true
backend = "fake"
model = "fake-model"
cache_dir = "{cache_dir.as_posix()}"

[sra]
skill_dirs = ["{skill_dir.as_posix()}"]
""",
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
            {
                "config": str(config),
                "sra_skill_dir": None,
                "embedding_backend": None,
            "embedding_cache_dir": None,
            "embedding_model": None,
            "retrieval_mode": "hybrid",
        },
    )()
    skills = helper._load_configured_or_cli_skills(args)

    with pytest.raises(ValueError, match="Missing dense embedding cache"):
        helper._require_embedding_cache(args, skills)


def test_sra_agent_helper_accepts_existing_embedding_cache(tmp_path: Path) -> None:
    helper = _load_agent_helper()
    skill_dir = tmp_path / "skills"
    cache_dir = tmp_path / "cache"
    config = tmp_path / "config.toml"
    write_sra_skill_files(_skill(), SraSkillMetadata.parse_obj(_metadata()), skill_dir)
    config.write_text(
        f"""
[embedding]
enabled = true
backend = "fake"
model = "fake-model"
cache_dir = "{cache_dir.as_posix()}"

[sra]
skill_dirs = ["{skill_dir.as_posix()}"]
""",
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
            {
                "config": str(config),
                "sra_skill_dir": None,
                "embedding_backend": None,
            "embedding_cache_dir": None,
            "embedding_model": None,
            "retrieval_mode": "hybrid",
        },
    )()
    skills = helper._load_configured_or_cli_skills(args)
    for view_name in helper.SRA_METADATA_DENSE_VIEW_NAMES:
        text = helper._metadata_view_text(skills[0], view_name)
        if not text.strip():
            continue
        path = dense_cache_path(cache_dir, "fake-semantic", view_name, [skills[0].id], [text])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"position": 0, "vector": [1.0, 0.0]}) + "\n", encoding="utf-8")

    helper._require_embedding_cache(args, skills)


def _load_agent_helper():
    path = ROOT / ".tmp" / "sra_run_agent_theoremqa_concurrent.py"
    spec = importlib.util.spec_from_file_location("sra_run_agent_theoremqa_concurrent_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata() -> dict:
    return {
        "short_description": "Apply Bayes rule for conditional probability questions.",
        "long_description": "Use Bayes theorem to compute posterior probabilities.",
        "capabilities": [{"id": "apply_bayes_rule", "description": "Compute conditional probabilities."}],
        "when_to_use": ["The task asks for posterior probability."],
        "positive_examples": [{"user_query": "Find P(A|B).", "reason": "Bayes rule."}],
        "tags": ["bayes"],
    }


def _skill() -> dict:
    return {
        "skill_id": "theoremqa_001",
        "name": "Apply Bayes rule",
        "description": "Use Bayes rule for conditional probability.",
        "content": "Use P(A|B) = P(B|A)P(A)/P(B).",
    }
