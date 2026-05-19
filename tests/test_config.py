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

[embedding]
enabled = true
backend = "fake"
model = "BAAI/bge-small-en-v1.5"
batch_size = 2
max_length = 128
device = "cpu"
cache_dir = "data/eval/sra/embedding_cache"

[search]
weight_lexical = 1.1
weight_sparse_view = 0.2
weight_dense = 0.3
weight_rrf = 0.4
weight_capability = 0.5
weight_usage = 0.6
weight_input_type = 0.7
weight_output_type = 0.8
weight_penalty = 0.9

[sra]
skill_dirs = ["data/eval/sra/theoremQA", "data/eval/sra/web"]
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
    assert config.embedding.enabled is True
    assert config.embedding.backend == "fake"
    assert config.embedding.model == "BAAI/bge-small-en-v1.5"
    assert config.embedding.batch_size == 2
    assert config.embedding.max_length == 128
    assert config.embedding.device == "cpu"
    assert config.embedding.cache_dir == "data/eval/sra/embedding_cache"
    assert config.search.weight_lexical == 1.1
    assert config.search.weight_dense == 0.3
    assert config.search.weight_penalty == 0.9
    assert config.sra.skill_dirs == ["data/eval/sra/theoremQA", "data/eval/sra/web"]


def test_load_app_config_without_sra_uses_empty_skill_dirs(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")

    config = load_app_config(config_path)

    assert config.sra.skill_dirs == []
