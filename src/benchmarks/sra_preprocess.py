"""Preprocess SR-Agents corpus skills into local SkillSpec files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, validator

from llm import LLMClient
from schema import SkillSpec

from .sra import SRA_CORPUS_PATH, _dataset_from_skill_id, _slugify, load_sra_corpus


DEFAULT_SRA_SKILL_DIR = Path("data/skills/sra")
DEFAULT_SRA_PREPROCESS_CHECKPOINT = Path("data/eval/sra/preprocess/deepseek-v4-pro.jsonl")
DEFAULT_SRA_PREPROCESS_MODEL = "deepseek-v4-pro"
CONTENT_EXCERPT_CHARS = 6000


SYSTEM_PROMPT = (
    "You generate lean retrieval metadata for a local SkillSpec database. "
    "Return strict JSON only. Do not solve the task. Do not invent benchmark answers. "
    "Use only the provided SR-Agents skill name, description, content, tools, and dataset. "
    "Focus on natural-language phrases that help dense retrieval match user questions to this skill."
)


class SraCapabilityMetadata(BaseModel):
    id: str
    description: str

    class Config:
        extra = "forbid"

    @validator("id")
    def validate_capability_id(cls, value: str) -> str:
        cleaned = _slugify(value)
        if not cleaned:
            raise ValueError("capability id cannot be empty")
        return cleaned

    @validator("description")
    def validate_capability_description(cls, value: str) -> str:
        cleaned = _compact_text(value, 600)
        if not cleaned:
            raise ValueError("capability description cannot be empty")
        return cleaned


class SraPositiveExampleMetadata(BaseModel):
    user_query: str
    reason: Optional[str] = None

    class Config:
        extra = "forbid"

    @validator("user_query")
    def validate_user_query(cls, value: str) -> str:
        cleaned = _compact_text(value, 500)
        if not cleaned:
            raise ValueError("positive example user_query cannot be empty")
        return cleaned

    @validator("reason")
    def validate_reason(cls, value: Optional[str]) -> Optional[str]:
        return _compact_text(value, 300) if value else None


class SraSkillMetadata(BaseModel):
    short_description: str
    long_description: str
    capabilities: list[SraCapabilityMetadata] = Field(min_items=1, max_items=4)
    when_to_use: list[str] = Field(min_items=1, max_items=4)
    positive_examples: list[SraPositiveExampleMetadata] = Field(min_items=1, max_items=3)
    tags: list[str] = Field(default_factory=list, max_items=12)

    class Config:
        extra = "forbid"

    @validator("short_description")
    def validate_short_description(cls, value: str) -> str:
        cleaned = _compact_text(value, 220)
        if not cleaned:
            raise ValueError("short_description cannot be empty")
        return cleaned

    @validator("long_description")
    def validate_long_description(cls, value: str) -> str:
        cleaned = _compact_text(value, 1200)
        if not cleaned:
            raise ValueError("long_description cannot be empty")
        return cleaned

    @validator("when_to_use", each_item=True)
    def validate_when_to_use(cls, value: str) -> str:
        cleaned = _compact_text(value, 300)
        if not cleaned:
            raise ValueError("when_to_use item cannot be empty")
        return cleaned

    @validator("tags", pre=True)
    def normalize_tags(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("tags must be a list")
        tags = []
        for tag in value:
            cleaned = _compact_text(tag, 60).lower()
            if cleaned and cleaned not in tags:
                tags.append(cleaned)
        return tags[:12]


def parse_sra_metadata(raw: str) -> SraSkillMetadata:
    """Parse strict JSON metadata returned by the preprocessing LLM."""

    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"SRA metadata response must be strict JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("SRA metadata response must be a JSON object")
    try:
        return SraSkillMetadata.parse_obj(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid SRA metadata JSON: {exc}") from exc


def sra_metadata_messages(skill: dict[str, Any]) -> list[dict[str, str]]:
    skill_id = _skill_id(skill)
    dataset = _dataset_from_skill_id(skill_id)
    tools = skill.get("tools")
    tools_json = json.dumps(tools or [], ensure_ascii=False, indent=2)
    content_excerpt = str(skill.get("content") or "")[:CONTENT_EXCERPT_CHARS]
    user_prompt = f"""Create minimal SkillSpec retrieval metadata for this SR-Agents skill.

Dataset: {dataset}
Skill ID: {skill_id}
Name: {skill.get("name") or skill_id}
Description:
{skill.get("description") or ""}

Tools, if any:
{tools_json}

Content excerpt:
{content_excerpt}

Return exactly:
{{
  "short_description": "one sentence, <= 220 chars",
  "long_description": "2-3 sentences with aliases, recognition cues, formulas, libraries, or task wording",
  "capabilities": [
    {{"id": "snake_case_id", "description": "natural-language capability"}}
  ],
  "when_to_use": ["2-4 short task situations"],
  "positive_examples": [
    {{"user_query": "realistic search query", "reason": "short match reason"}}
  ],
  "tags": ["lowercase aliases and topic words"]
}}

Rules:
- 2-4 capabilities only.
- 2-3 positive examples only.
- Tags must be compact; no long phrases.
- Include formal names and common aliases when present.
- For code skills, include module/library names and common coding task wording.
- For logic skills, include rule names and plain-English premise patterns.
- For medical calculator skills, include score names, aliases, and key patient-note variables.
"""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]


def sra_skill_to_enriched_spec(skill: dict[str, Any], metadata: SraSkillMetadata) -> SkillSpec:
    skill_id = _skill_id(skill)
    name = str(skill.get("name") or skill_id).strip()
    dataset = _dataset_from_skill_id(skill_id)
    tools = skill.get("tools") if isinstance(skill.get("tools"), list) else []
    input_schema = _tools_input_schema(tools)
    tags = _dedupe(["sra-bench", dataset, *metadata.tags])
    spec_data: dict[str, Any] = {
        "id": skill_id,
        "name": name,
        "version": "0.1.0",
        "status": "active",
        "skill_type": "tool_usage_guide" if tools else "instructional",
        "category": {"primary": dataset, "secondary": ["sra-bench"]},
        "description": {
            "short": metadata.short_description,
            "long": metadata.long_description,
        },
        "capabilities": [capability.dict() for capability in metadata.capabilities],
        "interaction": {
            "mode": "read_then_apply",
            "readable": True,
            "executable": False,
            "default_read_level": "overview",
        },
        "content": {
            "format": "markdown",
            "path": "skill.md",
            "sections": ["overview"],
        },
        "when_to_use": metadata.when_to_use,
        "examples": {
            "positive": [example.dict(exclude_none=True) for example in metadata.positive_examples],
        },
        "execution": {"mode": "none"},
        "tags": tags,
    }
    if input_schema:
        spec_data["input_schema"] = input_schema
    return SkillSpec.parse_obj(spec_data)


def build_sra_skill_markdown(skill: dict[str, Any]) -> str:
    name = str(skill.get("name") or _skill_id(skill)).strip()
    content = str(skill.get("content") or "").strip()
    return content or f"# {name}\n\n{skill.get('description') or name}\n"


def write_sra_skill_files(skill: dict[str, Any], metadata: SraSkillMetadata, output_skill_dir: str | Path) -> Path:
    spec = sra_skill_to_enriched_spec(skill, metadata)
    skill_dir = Path(output_skill_dir) / spec.id
    skill_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = skill_dir / "skill.yaml"
    md_path = skill_dir / "skill.md"
    spec_yaml = json.loads(spec.json(exclude={"root_dir"}))
    yaml_path.write_text(yaml.safe_dump(spec_yaml, sort_keys=False), encoding="utf-8")
    md_path.write_text(build_sra_skill_markdown(skill), encoding="utf-8")
    return skill_dir


def run_sra_preprocess(
    *,
    corpus_path: str | Path = SRA_CORPUS_PATH,
    output_skill_dir: str | Path = DEFAULT_SRA_SKILL_DIR,
    checkpoint_path: str | Path = DEFAULT_SRA_PREPROCESS_CHECKPOINT,
    llm: LLMClient,
    model_name: str = DEFAULT_SRA_PREPROCESS_MODEL,
    limit: int | None = None,
    resume: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    corpus = load_sra_corpus(corpus_path)
    if limit is not None:
        corpus = corpus[:limit]
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if force and checkpoint.exists():
        checkpoint.unlink()
    completed = load_sra_preprocess_checkpoint(checkpoint) if resume and not force else {}

    processed = 0
    reused = 0
    failed = 0
    with checkpoint.open("a", encoding="utf-8") as handle:
        for skill in corpus:
            skill_id = _skill_id(skill)
            metadata = completed.get(skill_id)
            if metadata is not None:
                reused += 1
            else:
                try:
                    raw = llm.complete(sra_metadata_messages(skill))
                    metadata = parse_sra_metadata(raw)
                    handle.write(
                        json.dumps(
                            {
                                "skill_id": skill_id,
                                "model": model_name,
                                "metadata": metadata.dict(),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    handle.flush()
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    handle.write(
                        json.dumps(
                            {"skill_id": skill_id, "model": model_name, "error": str(exc)},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    handle.flush()
                    continue
            write_sra_skill_files(skill, metadata, output_skill_dir)
            processed += 1

    return {
        "ok": failed == 0,
        "model": model_name,
        "corpus": str(corpus_path),
        "output_skill_dir": str(output_skill_dir),
        "checkpoint": str(checkpoint),
        "skill_count": len(corpus),
        "processed": processed,
        "reused": reused,
        "failed": failed,
    }


def load_sra_preprocess_checkpoint(path: str | Path) -> dict[str, SraSkillMetadata]:
    checkpoint = Path(path)
    if not checkpoint.exists():
        return {}
    completed: dict[str, SraSkillMetadata] = {}
    for line in checkpoint.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        skill_id = str(record.get("skill_id") or "").strip()
        metadata = record.get("metadata")
        if skill_id and isinstance(metadata, dict):
            completed[skill_id] = SraSkillMetadata.parse_obj(metadata)
    return completed


def _tools_input_schema(tools: list[Any]) -> dict[str, Any] | None:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        parameters = tool.get("parameters")
        if not isinstance(parameters, dict):
            continue
        for name, raw_type in parameters.items():
            param_name = str(name).strip()
            if not param_name:
                continue
            properties[param_name] = {
                "type": _json_schema_type(raw_type),
                "description": f"Parameter {param_name} for {tool.get('name') or 'SRA tool'}.",
            }
            if param_name not in required:
                required.append(param_name)
    if not properties:
        return None
    return {"type": "object", "required": required, "properties": properties}


def _json_schema_type(value: Any) -> str:
    text = str(value or "").lower()
    if "int" in text:
        return "integer"
    if "float" in text or "double" in text or "number" in text:
        return "number"
    if "bool" in text:
        return "boolean"
    if "list" in text or "array" in text:
        return "array"
    if "dict" in text or "object" in text:
        return "object"
    return "string"


def _skill_id(skill: dict[str, Any]) -> str:
    skill_id = str(skill.get("skill_id") or skill.get("id") or "").strip()
    if not skill_id:
        raise ValueError(f"SRA skill is missing skill_id: {skill}")
    return skill_id


def _compact_text(value: Any, max_chars: int) -> str:
    return " ".join(str(value or "").split())[:max_chars]


def _dedupe(values: list[str]) -> list[str]:
    result = []
    for value in values:
        cleaned = _compact_text(value, 80)
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result
