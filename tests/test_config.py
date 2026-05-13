from __future__ import annotations

from pathlib import Path

from config import load_app_config


def test_load_app_config_from_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[llm]
base_url = "https://example.test/v1"
api_key = "secret"
model = "Qwen/Qwen3.5-397B-A17B"
temperature = 0.2
max_tokens = 2048
timeout = 30.0

[agent]
llm = "mock"
top_k = 3
max_steps = 4
read_max_tokens = 1000

[toolret]
queries = "tests/fixtures/toolret_queries.jsonl"
tools = "tests/fixtures/toolret_tools.jsonl"
first_stage_candidates = "tests/fixtures/toolret_candidates.jsonl"
subset = "apibank"
category = "web"
limit = 30
top_k = 10
use_instruction = true
baseline = "rankgpt"
llm = "mock"
first_stage_model = "BAAI/bge-small-en-v1.5"
first_stage_backend = "hf-transformers"
embed_batch_size = 4
embed_max_length = 256
embed_device = "cpu"
candidate_pool_size = 100
rankgpt_window_size = 20
rankgpt_step_size = 10
workers = 4
output = "toolret-result.json"
""",
        encoding="utf-8",
    )

    config = load_app_config(config_path)

    assert config.llm.base_url == "https://example.test/v1"
    assert config.llm.api_key == "secret"
    assert config.llm.model == "Qwen/Qwen3.5-397B-A17B"
    assert config.llm.temperature == 0.2
    assert config.llm.max_tokens == 2048
    assert config.llm.timeout == 30.0
    assert config.agent.top_k == 3
    assert config.toolret.queries == "tests/fixtures/toolret_queries.jsonl"
    assert config.toolret.tools == "tests/fixtures/toolret_tools.jsonl"
    assert config.toolret.first_stage_candidates == "tests/fixtures/toolret_candidates.jsonl"
    assert config.toolret.category == "web"
    assert config.toolret.baseline == "rankgpt"
    assert config.toolret.first_stage_model == "BAAI/bge-small-en-v1.5"
    assert config.toolret.first_stage_backend == "hf-transformers"
    assert config.toolret.embed_batch_size == 4
    assert config.toolret.embed_max_length == 256
    assert config.toolret.embed_device == "cpu"
    assert config.toolret.candidate_pool_size == 100
    assert config.toolret.rankgpt_window_size == 20
    assert config.toolret.rankgpt_step_size == 10
    assert config.toolret.workers == 4
