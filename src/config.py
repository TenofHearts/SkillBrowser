"""Application configuration models and TOML loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python < 3.11 without tomllib
    tomllib = None  # type: ignore[assignment]

from pydantic import BaseModel, Field

from core.embeddings import DEFAULT_EMBEDDING_MODEL
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


class EmbeddingConfig(BaseModel):
    enabled: bool = False
    backend: str = "none"
    model: str = DEFAULT_EMBEDDING_MODEL
    batch_size: int = 8
    max_length: int = 512
    device: Optional[str] = None
    cache_dir: Optional[str] = None


class SearchConfig(BaseModel):
    weight_lexical: Optional[float] = None
    weight_sparse_view: Optional[float] = None
    weight_dense: Optional[float] = None
    weight_rrf: Optional[float] = None
    weight_capability: Optional[float] = None
    weight_usage: Optional[float] = None
    weight_input_type: Optional[float] = None
    weight_output_type: Optional[float] = None
    weight_penalty: Optional[float] = None


class SraConfig(BaseModel):
    skill_dirs: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    llm: Optional[LLMConfig] = None
    agent: AgentConfig = AgentConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    search: SearchConfig = SearchConfig()
    sra: SraConfig = SraConfig()


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
    logical_lines = _minimal_toml_logical_lines(text)
    for line_number, raw_line in logical_lines:
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


def _minimal_toml_logical_lines(text: str) -> list[tuple[int, str]]:
    lines = []
    pending = None
    pending_line = 0
    bracket_depth = 0
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.split("#", 1)[0].strip()
        if pending is None:
            pending = raw_line
            pending_line = line_number
            bracket_depth = stripped.count("[") - stripped.count("]")
        else:
            pending += " " + raw_line.strip()
            bracket_depth += stripped.count("[") - stripped.count("]")
        if bracket_depth <= 0:
            lines.append((pending_line, pending))
            pending = None
            bracket_depth = 0
    if pending is not None:
        lines.append((pending_line, pending))
    return lines


def _parse_minimal_toml_value(value: str, line_number: int) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_minimal_toml_value(part.strip(), line_number) for part in inner.split(",") if part.strip()]
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
