from __future__ import annotations

import json
from typing import Any

from core.search import SkillSearcher
from core.selectors import parse_skill_search_decision
from loader import SkillLoadError
from llm import LLMClient
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
        search_query: str | None = None
        if request.max_steps >= 1:
            decide_raw = self.llm.complete(_skill_search_decision_messages(request.task, request.top_k))
            search_query, error = parse_skill_search_decision(decide_raw)
            if error:
                parse_errors.append(error)
            steps.append(
                AgentStep(
                    step=1,
                    action="llm_decide_tools",
                    input={"task": request.task, "available_tools": ["skill_search"]},
                    observation={"search_query": search_query},
                    raw_model_output=decide_raw,
                    error=error,
                )
            )

        if search_query and request.max_steps >= 2:
            search_response = self.searcher.search(SkillSearchRequest(query=search_query, top_k=request.top_k))
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
                    input={"query": search_query, "top_k": request.top_k},
                    observation={"results": search_cards},
                )
            )

        selected_skill_id = search_cards[0]["id"] if search_cards else None
        read_section = search_cards[0]["read_recommendation"] if search_cards else None
        choose_raw = ""
        if search_cards and request.max_steps >= 3:
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
                    step=3,
                    action="llm_choose_skill",
                    input={"candidate_skill_ids": [card["id"] for card in search_cards]},
                    observation={"selected_skill_id": selected_skill_id, "read_section": read_section},
                    raw_model_output=choose_raw,
                    error=error,
                )
            )

        read_content = ""
        read_skill_ids: list[str] = []
        if selected_skill_id and read_section and request.max_steps >= 4:
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
                        step=4,
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
                        step=4,
                        action="skill_read",
                        input={"skill_id": selected_skill_id, "section": read_section},
                        error=str(exc),
                    )
                )

        final_answer = "No relevant skill was found."
        final_raw = ""
        if request.max_steps >= 5:
            final_raw = self.llm.complete(_final_answer_messages(request.task, selected_skill_id, read_content))
            parsed, error = _parse_json_object(final_raw)
            if error:
                parse_errors.append(error)
                final_answer = final_raw.strip()
            else:
                final_answer = str(parsed.get("final_answer") or final_raw).strip()
            steps.append(
                AgentStep(
                    step=5,
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


def _skill_search_decision_messages(task: str, top_k: int) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Decide whether the user task needs specialized skills or tool knowledge. "
                "You initially have one framework tool available: skill_search. "
                "If useful, call it by returning strict JSON only: "
                '{"action": "skill_search", "query": "short search query", "reason": "short reason"}. '
                "If no skill is needed, return strict JSON only: "
                '{"action": "answer_without_tools", "reason": "short reason"}.'
            ),
        },
        {"role": "user", "content": json.dumps({"task": task, "top_k": top_k}, indent=2)},
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
