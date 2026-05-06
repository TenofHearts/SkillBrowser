from __future__ import annotations

from pathlib import Path

from skill_search_agent.config import load_app_config


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

[gatewaybench_compare]
dataset = "tests/fixtures/gatewaybench_lite.jsonl"
limit = 50
top_k = 5
selectors = ["hybrid", "llm-baseline"]
llm = "mock"
workers = 10
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
    assert config.gatewaybench_compare.dataset == "tests/fixtures/gatewaybench_lite.jsonl"
    assert config.gatewaybench_compare.selectors == ["hybrid", "llm-baseline"]
    assert config.gatewaybench_compare.workers == 10
