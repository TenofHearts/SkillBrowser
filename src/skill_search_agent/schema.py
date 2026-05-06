from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, root_validator


class SkillType(str, Enum):
    instructional = "instructional"
    workflow = "workflow"
    reference = "reference"
    prompt_pattern = "prompt_pattern"
    code_recipe = "code_recipe"
    tool_usage_guide = "tool_usage_guide"
    tool_wrapper = "tool_wrapper"
    hybrid = "hybrid"


class InteractionMode(str, Enum):
    read_then_apply = "read_then_apply"
    read_then_generate_code = "read_then_generate_code"
    read_then_execute = "read_then_execute"
    execute_directly = "execute_directly"
    reference_only = "reference_only"


class ExecutionMode(str, Enum):
    none = "none"
    python_function = "python_function"
    subprocess = "subprocess"
    http_local = "http_local"
    mock = "mock"


class Description(BaseModel):
    short: str
    long: Optional[str] = None


class Category(BaseModel):
    primary: str
    secondary: list[str] = Field(default_factory=list)


class Capability(BaseModel):
    id: str
    description: str


class Interaction(BaseModel):
    mode: InteractionMode
    readable: bool = True
    executable: bool = False
    default_read_level: str = "overview"


class Content(BaseModel):
    format: Literal["markdown"] = "markdown"
    path: str = "skill.md"
    sections: list[str] = Field(default_factory=list)


class Execution(BaseModel):
    mode: ExecutionMode = ExecutionMode.none
    module: Optional[str] = None
    function: Optional[str] = None

    @root_validator(skip_on_failure=True)
    def validate_python_function(cls, values: dict[str, Any]) -> dict[str, Any]:
        if values.get("mode") == ExecutionMode.python_function and (
            not values.get("module") or not values.get("function")
        ):
            raise ValueError("python_function execution requires module and function")
        return values


class Example(BaseModel):
    user_query: str
    reason: Optional[str] = None


class Examples(BaseModel):
    positive: list[Example] = Field(default_factory=list)
    negative: list[Example] = Field(default_factory=list)


class SkillSpec(BaseModel):
    id: str
    name: str
    version: str = "0.1.0"
    status: str = "active"
    skill_type: SkillType
    category: Category
    description: Description
    capabilities: list[Capability] = Field(default_factory=list)
    interaction: Interaction
    content: Content
    when_to_use: list[str] = Field(default_factory=list)
    when_not_to_use: list[str] = Field(default_factory=list)
    input_types: list[str] = Field(default_factory=list)
    output_types: list[str] = Field(default_factory=list)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    examples: Examples = Field(default_factory=Examples)
    execution: Execution = Field(default_factory=Execution)
    tags: list[str] = Field(default_factory=list)
    root_dir: Optional[Path] = Field(default=None, exclude=True)

    @property
    def execution_available(self) -> bool:
        return self.interaction.executable and self.execution.mode != ExecutionMode.none

    @root_validator(skip_on_failure=True)
    def validate_execution_consistency(cls, values: dict[str, Any]) -> dict[str, Any]:
        interaction = values.get("interaction")
        execution = values.get("execution")
        if not interaction or not execution:
            return values
        has_execution = execution.mode != ExecutionMode.none
        if interaction.executable and not has_execution:
            raise ValueError("interaction.executable=true requires execution.mode other than none")
        if has_execution and not interaction.executable:
            raise ValueError("execution metadata requires interaction.executable=true")
        return values


class SkillSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    task_context: Optional[str] = None
    required_capabilities: list[str] = Field(default_factory=list)
    input_types: list[str] = Field(default_factory=list)
    output_types: list[str] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    lexical: float
    capability: float
    usage: float
    vector: float = 0.0
    rrf: float = 0.0
    input_type: float = 0.0
    output_type: float = 0.0
    contraindication_penalty: float = 0.0


class SkillCard(BaseModel):
    id: str
    name: str
    score: float
    skill_type: SkillType
    interaction_mode: InteractionMode
    execution_available: bool
    description: str
    matched_capabilities: list[str]
    available_sections: list[str]
    read_recommendation: str
    score_breakdown: ScoreBreakdown
    usage_constraints: list[str] = Field(default_factory=list)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None


class SkillSearchResponse(BaseModel):
    query: str
    results: list[SkillCard]


class SkillReadRequest(BaseModel):
    skill_id: str
    section: Optional[str] = None
    max_tokens: int = Field(default=2000, ge=1)


class SkillReadResponse(BaseModel):
    skill_id: str
    name: str
    section: Optional[str]
    content: str
    token_count: int
    truncated: bool
    available_sections: list[str]


class AgentRunRequest(BaseModel):
    task: str
    top_k: int = Field(default=5, ge=1, le=50)
    max_steps: int = Field(default=4, ge=1, le=20)
    read_max_tokens: int = Field(default=2000, ge=1)


class AgentStep(BaseModel):
    step: int
    action: str
    input: Dict[str, Any] = Field(default_factory=dict)
    observation: Dict[str, Any] = Field(default_factory=dict)
    raw_model_output: Optional[str] = None
    error: Optional[str] = None


class AgentRunResult(BaseModel):
    task: str
    final_answer: str
    selected_skill_ids: list[str] = Field(default_factory=list)
    read_skill_ids: list[str] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    parse_errors: list[str] = Field(default_factory=list)


class CandidateTool(BaseModel):
    id: str
    name: str
    description: str = ""


class ToolSelectionRequest(BaseModel):
    prompt: str
    candidates: list[CandidateTool]
    top_k: int = Field(default=5, ge=1, le=50)
    task_context: Optional[str] = None


class ToolSelectionResult(BaseModel):
    ranked_tool_ids: list[str]
    raw_model_output: Optional[str] = None
    parse_error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    token_usage_source: str = "none"
