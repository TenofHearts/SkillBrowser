"""Tool selection strategies and JSON parsers for retrieval-augmented LLM ranking."""

from __future__ import annotations

import json
from typing import Any, Protocol

from llm import LLMClient
from schema import CandidateTool, SkillSearchRequest, ToolSelectionRequest, ToolSelectionResult
from .search import SkillSearcher


class ToolSelector(Protocol):
    def select(self, request: ToolSelectionRequest) -> ToolSelectionResult:
        ...


class HybridSearchToolSelector:
    def __init__(self, searcher: SkillSearcher, llm: LLMClient, candidate_pool_size: int = 20):
        self.searcher = searcher
        self.llm = llm
        self.candidate_pool_size = candidate_pool_size

    def select(self, request: ToolSelectionRequest) -> ToolSelectionResult:
        decision = self.llm.complete_with_usage(_tool_search_decision_messages(request))
        query, decision_error = parse_skill_search_decision(decision.content, default_query=request.prompt)
        if decision_error:
            return ToolSelectionResult(
                ranked_tool_ids=[],
                raw_model_output=decision.content,
                parse_error=decision_error,
                input_tokens=decision.input_tokens,
                output_tokens=decision.output_tokens,
                latency_ms=decision.elapsed_ms,
                token_usage_source=decision.token_usage_source,
            )
        if query is None:
            return ToolSelectionResult(
                ranked_tool_ids=[],
                raw_model_output=decision.content,
                input_tokens=decision.input_tokens,
                output_tokens=decision.output_tokens,
                latency_ms=decision.elapsed_ms,
                token_usage_source=decision.token_usage_source,
            )

        response = self.searcher.search(
            SkillSearchRequest(
                query=request.prompt,
                task_context=request.task_context,
            ),
            top_k=_retrieval_top_k(request.top_k, self.candidate_pool_size),
        )
        search_candidates = [
            CandidateTool(
                id=card.id,
                name=card.name,
                description=(
                    f"{card.description} "
                    f"[retrieval_rank={index}; retrieval_score={card.score:.4f}]"
                ).strip(),
            )
            for index, card in enumerate(response.results, start=1)
        ]
        selection_request = ToolSelectionRequest(
            prompt=request.prompt,
            task_context=request.task_context,
            candidates=search_candidates,
            top_k=request.top_k,
        )
        selection = self.llm.complete_with_usage(_selection_messages(selection_request))
        ranked_ids, selection_error = parse_selection_json(
            selection.content,
            {candidate.id for candidate in search_candidates},
        )
        usage_source = _merge_usage_sources(decision.token_usage_source, selection.token_usage_source)
        return ToolSelectionResult(
            ranked_tool_ids=ranked_ids[: request.top_k],
            raw_model_output=selection.content,
            parse_error=selection_error,
            input_tokens=decision.input_tokens + selection.input_tokens,
            output_tokens=decision.output_tokens + selection.output_tokens,
            latency_ms=decision.elapsed_ms + selection.elapsed_ms,
            token_usage_source=usage_source,
        )


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


def parse_skill_search_decision(raw: str, default_query: str | None = None) -> tuple[str | None, str | None]:
    raw = extract_json_object_text(raw)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"Could not parse LLM JSON output: {exc}"
    if not isinstance(payload, dict):
        return None, "LLM JSON output must be an object"

    action = payload.get("action")
    if action in {"answer_without_tools", "no_tools"}:
        return None, None
    if action != "skill_search":
        return None, "LLM JSON output must choose action skill_search or answer_without_tools"

    query = payload.get("query", default_query)
    if not isinstance(query, str) or not query.strip():
        return None, "skill_search action must include a non-empty query"
    return query.strip(), None


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


def _tool_search_decision_messages(request: ToolSelectionRequest) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an agent that can decide whether a user request needs external tools. "
                "You initially have one framework tool available: skill_search. "
                "If tools may be needed, call skill_search by returning strict JSON only. "
                "Do not rewrite the user request; the framework will search with the original prompt. "
                'Return: {"action": "skill_search", "reason": "short reason"}. '
                "If no tools are needed, return strict JSON only: "
                '{"action": "answer_without_tools", "reason": "short reason"}.'
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "prompt": request.prompt,
                    "task_context": request.task_context,
                    "top_k": request.top_k,
                },
                indent=2,
            ),
        },
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


def _merge_usage_sources(first: str, second: str) -> str:
    if first == second:
        return first
    if first == "none":
        return second
    if second == "none":
        return first
    return "mixed"


def _retrieval_top_k(final_top_k: int, candidate_pool_size: int) -> int:
    return min(max(final_top_k, candidate_pool_size), 50)
