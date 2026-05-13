"""LLM-driven skill selection agent that searches, reads, and answers with relevant skills."""

from __future__ import annotations

import json
from typing import Any

from core.search import SkillSearcher
from loader import SkillLoadError
from llm import LLMClient
from pydantic import ValidationError
from reader import SkillReader
from schema import AgentRunRequest, AgentRunResult, AgentStep, SkillReadRequest, SkillSearchRequest


class SkillAgent:
    def __init__(self, searcher: SkillSearcher, reader: SkillReader, llm: LLMClient):
        self.searcher = searcher
        self.reader = reader
        self.llm = llm

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        steps: list[AgentStep] = []
        parse_errors: list[str] = []

        search_cards: list[dict[str, Any]] = []
        search_request: SkillSearchRequest | None = None
        if request.max_steps >= 1:
            decide_raw = self.llm.complete(skill_search_intent_messages(request.task))
            search_request, error = parse_skill_search_intent(decide_raw, request.task)
            if error:
                parse_errors.append(error)
            steps.append(
                AgentStep(
                    step=1,
                    action="llm_extract_search_intent",
                    input={"task": request.task, "available_tools": ["skill_search"]},
                    observation={"search_request": search_request.dict() if search_request else None},
                    raw_model_output=decide_raw,
                    error=error,
                )
            )

        if search_request and request.max_steps >= 2:
            search_response = self.searcher.search(search_request, top_k=request.top_k)
            search_cards = [
                {
                    "id": card.id,
                    "name": card.name,
                    "description": card.description,
                    "score": card.score,
                    "read_recommendation": card.read_recommendation,
                }
                for card in search_response.results
            ]
            steps.append(
                AgentStep(
                    step=2,
                    action="skill_search",
                    input={"request": search_request.dict(), "top_k": request.top_k},
                    observation={
                        "results": search_cards,
                        "abstained": search_response.abstained,
                        "abstention_reason": search_response.abstention_reason,
                    },
                )
            )
            if not search_cards and request.max_steps >= 3:
                revise_raw = self.llm.complete(
                    _revise_skill_search_intent_messages(
                        request.task,
                        search_request,
                        search_response.abstention_reason,
                    )
                )
                revised_request, error = parse_skill_search_intent(revise_raw, request.task)
                if error:
                    parse_errors.append(error)
                steps.append(
                    AgentStep(
                        step=3,
                        action="llm_revise_search_intent",
                        input={
                            "task": request.task,
                            "previous_request": search_request.dict(),
                            "abstention_reason": search_response.abstention_reason,
                        },
                        observation={"search_request": revised_request.dict() if revised_request else None},
                        raw_model_output=revise_raw,
                        error=error,
                    )
                )
                if revised_request:
                    search_request = revised_request
                    search_response = self.searcher.search(search_request, top_k=request.top_k)
                    search_cards = [
                        {
                            "id": card.id,
                            "name": card.name,
                            "description": card.description,
                            "score": card.score,
                            "read_recommendation": card.read_recommendation,
                        }
                        for card in search_response.results
                    ]
                    steps.append(
                        AgentStep(
                            step=4,
                            action="skill_search",
                            input={"request": search_request.dict(), "top_k": request.top_k},
                            observation={
                                "results": search_cards,
                                "abstained": search_response.abstained,
                                "abstention_reason": search_response.abstention_reason,
                            },
                        )
                    )

        selected_skill_id = search_cards[0]["id"] if search_cards else None
        read_section = search_cards[0]["read_recommendation"] if search_cards else None
        choose_raw = ""
        if search_cards and len(steps) < request.max_steps:
            choose_raw = self.llm.complete(_choose_skill_messages(request.task, search_cards))
            parsed, error = _parse_json_object(choose_raw)
            if error:
                parse_errors.append(error)
            selected = _first_valid_selected_id(parsed, {card["id"] for card in search_cards})
            if selected:
                selected_skill_id = selected
                matching = next(card for card in search_cards if card["id"] == selected)
                read_section = str(matching["read_recommendation"])
            steps.append(
                AgentStep(
                    step=len(steps) + 1,
                    action="llm_choose_skill",
                    input={"candidate_skill_ids": [card["id"] for card in search_cards]},
                    observation={"selected_skill_id": selected_skill_id, "read_section": read_section},
                    raw_model_output=choose_raw,
                    error=error,
                )
            )

        read_content = ""
        read_skill_ids: list[str] = []
        if selected_skill_id and read_section and len(steps) < request.max_steps:
            try:
                read_response = self.reader.read(
                    SkillReadRequest(
                        skill_id=selected_skill_id,
                        section=read_section,
                        max_tokens=request.read_max_tokens,
                    )
                )
                read_content = read_response.content
                read_skill_ids.append(read_response.skill_id)
                steps.append(
                    AgentStep(
                        step=len(steps) + 1,
                        action="skill_read",
                        input={
                            "skill_id": selected_skill_id,
                            "section": read_section,
                            "max_tokens": request.read_max_tokens,
                        },
                        observation={
                            "skill_id": read_response.skill_id,
                            "section": read_response.section,
                            "token_count": read_response.token_count,
                            "truncated": read_response.truncated,
                        },
                    )
                )
            except SkillLoadError as exc:
                steps.append(
                    AgentStep(
                        step=len(steps) + 1,
                        action="skill_read",
                        input={"skill_id": selected_skill_id, "section": read_section},
                        error=str(exc),
                    )
                )

        final_answer = "No relevant skill was found."
        final_raw = ""
        if len(steps) < request.max_steps:
            final_raw = self.llm.complete(_final_answer_messages(request.task, selected_skill_id, read_content))
            parsed, error = _parse_json_object(final_raw)
            if error:
                parse_errors.append(error)
                final_answer = final_raw.strip()
            else:
                final_answer = str(parsed.get("final_answer") or final_raw).strip()
            steps.append(
                AgentStep(
                    step=len(steps) + 1,
                    action="llm_final_answer",
                    input={"skill_id": selected_skill_id},
                    observation={"final_answer": final_answer},
                    raw_model_output=final_raw,
                    error=error,
                )
            )

        return AgentRunResult(
            task=request.task,
            final_answer=final_answer,
            selected_skill_ids=[selected_skill_id] if selected_skill_id else [],
            read_skill_ids=read_skill_ids,
            steps=steps,
            parse_errors=parse_errors,
        )


def _choose_skill_messages(task: str, cards: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Choose the best skill for the user task. Return strict JSON only: "
                '{"selected_ids": ["skill.id"], "reason": "short reason"}.'
            ),
        },
        {"role": "user", "content": json.dumps({"task": task, "candidate_skills": cards}, indent=2)},
    ]


def skill_search_intent_messages(task: str, *, benchmark_retrieval: bool = False) -> list[dict[str, str]]:
    if benchmark_retrieval:
        return [
            {
                "role": "system",
                "content": (
                    "You generate retrieval queries for a tool-search benchmark. "
                    "Your only job is to rewrite the user's task into structured search intent that can match tool documentation. "
                    "Do not solve the task. Do not choose a tool. Do not explain.\n\n"
                    "Return exactly one JSON object with this shape:\n"
                    '{'
                    '"action":"skill_search",'
                    '"retrieval_intent":{'
                    '"query":"...",'
                    '"task_context":"...",'
                    '"required_capabilities":[],'
                    '"desired_capabilities":[],'
                    '"input_types":[],'
                    '"output_types":[],'
                    '"positive_signals":[],'
                    '"negative_signals":[],'
                    '"constraints":{}'
                    "}"
                    "}\n\n"
                    "Field rules:\n"
                    "- query: one concise search string. Preserve the main nouns, verbs, entities, and domain words from the user task. Add only obvious tool-search synonyms.\n"
                    "- task_context: one short sentence describing the tool capability needed, not the answer.\n"
                    "- required_capabilities: leave empty unless the task explicitly names a mandatory operation that a wrong tool must not miss.\n"
                    "- desired_capabilities: 1-5 short capability phrases useful for dense retrieval.\n"
                    "- input_types/output_types: leave empty unless the task explicitly names a concrete data or file type such as pdf, csv, image, json, sql, code, audio, or url.\n"
                    "- positive_signals: 2-6 phrases likely to appear in relevant tool docs or examples.\n"
                    "- negative_signals: only explicit exclusions or clearly confusable sibling tasks.\n"
                    "- constraints: normally {}. Use only for explicit permission, execution, readability, latency, or risk constraints.\n\n"
                    "Keep arrays concise. Do not include top_k. Return JSON only."
                ),
            },
            {"role": "user", "content": json.dumps({"task": task}, indent=2)},
        ]
    else:
        guidance = (
            "Decide whether the user task needs specialized skills or tool knowledge. "
            "You initially have one framework tool available: skill_search. "
            "If useful, return strict JSON only with action=skill_search and a retrieval_intent object. "
            "If no skill is needed, return strict JSON only: "
            '{"action": "answer_without_tools", "reason": "short reason"}.'
        )
    return [
        {
            "role": "system",
            "content": (
                guidance
                +
                "The retrieval_intent fields are: query, task_context, required_capabilities, "
                "desired_capabilities, input_types, output_types, positive_signals, negative_signals, constraints. "
                "Return strict JSON only."
            ),
        },
        {"role": "user", "content": json.dumps({"task": task}, indent=2)},
    ]


def _revise_skill_search_intent_messages(
    task: str,
    previous_request: SkillSearchRequest,
    abstention_reason: str | None,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Revise the structured skill_search retrieval intent after an empty or abstained result. "
                "Return strict JSON only with action=skill_search and retrieval_intent. "
                "Do not include top_k."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": task,
                    "previous_retrieval_intent": previous_request.dict(),
                    "abstention_reason": abstention_reason,
                },
                indent=2,
            ),
        },
    ]


def _final_answer_messages(task: str, skill_id: str | None, skill_content: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": 'Use the selected skill instructions to answer. Return strict JSON: {"final_answer": "..."}',
        },
        {
            "role": "user",
            "content": json.dumps(
                {"task": task, "selected_skill_id": skill_id, "skill_content": skill_content},
                indent=2,
            ),
        },
    ]


def parse_skill_search_intent(raw: str, fallback_query: str) -> tuple[SkillSearchRequest | None, str | None]:
    parsed, error = _parse_json_object(raw)
    if error:
        return SkillSearchRequest(query=fallback_query), error
    action = parsed.get("action")
    if action in {"answer_without_tools", "no_tools"}:
        return None, None
    intent_fields = {
        "query",
        "task_context",
        "required_capabilities",
        "desired_capabilities",
        "input_types",
        "output_types",
        "positive_signals",
        "negative_signals",
        "constraints",
    }
    if action is None and (set(parsed) & intent_fields):
        intent = parsed
    elif action != "skill_search":
        return SkillSearchRequest(query=fallback_query), "LLM JSON output must choose action skill_search or answer_without_tools"
    else:
        intent = parsed.get("retrieval_intent", parsed)
    if not isinstance(intent, dict):
        return SkillSearchRequest(query=fallback_query), "skill_search action must include retrieval_intent object"
    intent = {key: value for key, value in intent.items() if key != "action"}
    if not intent.get("query"):
        intent["query"] = fallback_query
    intent = _normalize_skill_search_intent(intent)
    try:
        return SkillSearchRequest.parse_obj(intent), None
    except ValidationError as exc:
        return SkillSearchRequest(query=fallback_query), f"Invalid skill_search retrieval_intent: {exc}"


def _normalize_skill_search_intent(intent: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(intent)
    for key in [
        "required_capabilities",
        "desired_capabilities",
        "input_types",
        "output_types",
        "positive_signals",
        "negative_signals",
    ]:
        value = normalized.get(key)
        if value is None:
            normalized[key] = []
        elif isinstance(value, str):
            normalized[key] = [value] if value.strip() else []
        elif not isinstance(value, list):
            normalized[key] = [str(value)]
    constraints = normalized.get("constraints")
    if constraints is None:
        normalized["constraints"] = {}
    elif isinstance(constraints, str):
        normalized["constraints"] = {"note": constraints} if constraints.strip() else {}
    elif not isinstance(constraints, dict):
        normalized["constraints"] = {"value": constraints}
    return normalized


def _parse_json_object(raw: str) -> tuple[dict[str, Any], str | None]:
    raw = _extract_json_object_text(raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, f"Could not parse LLM JSON output: {exc}"
    if not isinstance(parsed, dict):
        return {}, "LLM JSON output must be an object"
    return parsed, None


def _extract_json_object_text(raw: str) -> str:
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


def _first_valid_selected_id(parsed: dict[str, Any], valid_ids: set[str]) -> str | None:
    selected_ids = parsed.get("selected_ids")
    if isinstance(selected_ids, str):
        selected_ids = [selected_ids]
    if not isinstance(selected_ids, list):
        return None
    for skill_id in selected_ids:
        if isinstance(skill_id, str) and skill_id in valid_ids:
            return skill_id
    return None
