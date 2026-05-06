from __future__ import annotations

import json
from typing import Any, Protocol

from .llm import LLMClient
from .schema import CandidateTool, SkillSearchRequest, ToolSelectionRequest, ToolSelectionResult
from .search import SkillSearcher


class ToolSelector(Protocol):
    def select(self, request: ToolSelectionRequest) -> ToolSelectionResult:
        ...


class HybridSearchToolSelector:
    def __init__(self, searcher: SkillSearcher):
        self.searcher = searcher

    def select(self, request: ToolSelectionRequest) -> ToolSelectionResult:
        response = self.searcher.search(
            SkillSearchRequest(
                query=request.prompt,
                task_context=request.task_context,
                top_k=request.top_k,
            )
        )
        return ToolSelectionResult(ranked_tool_ids=[card.id for card in response.results])


class BaselineLLMToolSelector:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def select(self, request: ToolSelectionRequest) -> ToolSelectionResult:
        candidate_ids = {candidate.id for candidate in request.candidates}
        completion = self.llm.complete_with_usage(_selection_messages(request))
        raw = completion.content
        parsed, error = parse_selection_json(raw, candidate_ids)
        if error:
            return ToolSelectionResult(
                ranked_tool_ids=[],
                raw_model_output=raw,
                parse_error=error,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                latency_ms=completion.elapsed_ms,
                token_usage_source=completion.token_usage_source,
            )
        return ToolSelectionResult(
            ranked_tool_ids=parsed[: request.top_k],
            raw_model_output=raw,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            latency_ms=completion.elapsed_ms,
            token_usage_source=completion.token_usage_source,
        )


def parse_selection_json(raw: str, valid_ids: set[str]) -> tuple[list[str], str | None]:
    raw = extract_json_object_text(raw)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [], f"Could not parse LLM JSON output: {exc}"
    if not isinstance(payload, dict):
        return [], "LLM JSON output must be an object"
    ranked = payload.get("ranked_tool_ids", payload.get("selected_ids"))
    if isinstance(ranked, str):
        ranked = [ranked]
    if not isinstance(ranked, list):
        return [], "LLM JSON output must include ranked_tool_ids or selected_ids"

    valid_ranked: list[str] = []
    for tool_id in ranked:
        if isinstance(tool_id, str) and tool_id in valid_ids and tool_id not in valid_ranked:
            valid_ranked.append(tool_id)
    if not valid_ranked:
        return [], "LLM output did not include any valid candidate ids"
    return valid_ranked, None


def extract_json_object_text(raw: str) -> str:
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def candidate_tools_from_cards(candidates: list[CandidateTool]) -> list[dict[str, Any]]:
    return [
        {"id": candidate.id, "name": candidate.name, "description": candidate.description}
        for candidate in candidates
    ]


def _selection_messages(request: ToolSelectionRequest) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Select the tools needed for the user request. Return strict JSON only: "
                '{"ranked_tool_ids": ["tool.id"], "reason": "short reason"}. '
                "Use only ids from candidate_tools."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "prompt": request.prompt,
                    "task_context": request.task_context,
                    "top_k": request.top_k,
                    "candidate_tools": candidate_tools_from_cards(request.candidates),
                },
                indent=2,
            ),
        },
    ]
