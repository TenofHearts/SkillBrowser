"""Application configuration models and TOML loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python < 3.11 without tomllib
    tomllib = None  # type: ignore[assignment]

from pydantic import BaseModel

from llm import DEFAULT_OPENAI_COMPATIBLE_MODEL


class LLMConfig(BaseModel):
    base_url: str
    api_key: str
    model: str = DEFAULT_OPENAI_COMPATIBLE_MODEL
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = 120.0


class AgentConfig(BaseModel):
    llm: str = "mock"
    top_k: int = 5
    max_steps: int = 5
    read_max_tokens: int = 2000


class ToolRetConfig(BaseModel):
    queries: Optional[str] = None
    tools: Optional[str] = None
    first_stage_candidates: Optional[str] = None
    subset: Optional[str] = None
    category: str = "all"
    limit: int = 30
    top_k: int = 10
    use_instruction: bool = True
    baseline: str = "hybrid"
    llm: str = "mock"
    candidate_pool_size: int = 100
    rankgpt_window_size: int = 20
    rankgpt_step_size: int = 10
    workers: int = 1
    output: Optional[str] = None


class AppConfig(BaseModel):
    llm: Optional[LLMConfig] = None
    agent: AgentConfig = AgentConfig()
    toolret: ToolRetConfig = ToolRetConfig()


def load_app_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")
    data = _load_toml(config_path)
    return AppConfig.parse_obj(data)


def load_app_config_if_exists(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()
    return load_app_config(config_path)


def _load_toml(path: Path) -> dict[str, Any]:
    if tomllib is not None:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    return _load_minimal_toml(path.read_text(encoding="utf-8"))


def _load_minimal_toml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current: dict[str, Any] = data
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            if not section:
                raise ValueError(f"Invalid empty TOML section at line {line_number}")
            current = data.setdefault(section, {})
            if not isinstance(current, dict):
                raise ValueError(f"Invalid TOML section at line {line_number}: {section}")
            continue
        if "=" not in line:
            raise ValueError(f"Invalid TOML assignment at line {line_number}: {raw_line}")
        key, value = [part.strip() for part in line.split("=", 1)]
        current[key] = _parse_minimal_toml_value(value, line_number)
    return data


def _parse_minimal_toml_value(value: str, line_number: int) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_minimal_toml_value(part.strip(), line_number) for part in inner.split(",")]
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported TOML value at line {line_number}: {value}") from exc
