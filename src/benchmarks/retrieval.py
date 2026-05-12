"""Dataset loading and metric calculation for local skill retrieval benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, root_validator

from core.search import SkillSearcher
from schema import SkillSearchRequest


class RetrievalExample(BaseModel):
    query: str
    expected_skill_ids: list[str] = Field(default_factory=list)
    expected_section: Optional[str] = None
    relevance_by_id: Optional[dict[str, str]] = None
    allow_no_result: bool = False

    @root_validator(pre=True)
    def normalize_legacy_expected_skill_id(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "expected_skill_id" in values and "expected_skill_ids" not in values:
            values["expected_skill_ids"] = [values.pop("expected_skill_id")]
        return values

    @root_validator(skip_on_failure=True)
    def validate_expected_ids(cls, values: dict[str, Any]) -> dict[str, Any]:
        expected_skill_ids = values.get("expected_skill_ids") or []
        allow_no_result = values.get("allow_no_result")
        if not expected_skill_ids and not allow_no_result:
            raise ValueError("expected_skill_ids must be non-empty unless allow_no_result=true")
        return values


class RetrievalEvalResult(BaseModel):
    query_count: int
    top_k: int
    recall_at_k: float
    mrr: float
    precision_at_k: float
    f1_at_k: float
    irrelevant_selection_rate: float
    correct_section_recommendation_at_k: float
    misses: list[dict[str, Any]] = Field(default_factory=list)


def load_retrieval_dataset(path: str | Path) -> list[RetrievalExample]:
    dataset_path = Path(path)
    examples: list[RetrievalExample] = []
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            examples.append(RetrievalExample.parse_obj(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"Invalid retrieval example at {dataset_path}:{line_number}: {exc}") from exc
    if not examples:
        raise ValueError(f"Retrieval dataset is empty: {dataset_path}")
    return examples


def evaluate_retrieval(
    searcher: SkillSearcher, examples: list[RetrievalExample], top_k: int
) -> RetrievalEvalResult:
    recall_sum = 0.0
    precision_sum = 0.0
    f1_sum = 0.0
    reciprocal_rank_sum = 0.0
    section_hits = 0
    irrelevant_returned = 0
    returned_count = 0
    misses: list[dict[str, Any]] = []

    for example in examples:
        response = searcher.search(SkillSearchRequest(query=example.query, top_k=top_k))
        ranked_ids = [card.id for card in response.results]

        stats = score_retrieval_result(example, ranked_ids, top_k)
        recall_sum += stats["recall"]
        precision_sum += stats["precision"]
        f1_sum += stats["f1"]
        reciprocal_rank_sum += stats["reciprocal_rank"]
        irrelevant_returned += stats["irrelevant_returned"]
        returned_count += len(ranked_ids)

        if not stats["hit"]:
            misses.append(
                {
                    "query": example.query,
                    "expected_skill_ids": example.expected_skill_ids,
                    "allow_no_result": example.allow_no_result,
                    "returned_skill_ids": ranked_ids,
                }
            )
            continue

        if example.expected_section is None:
            section_hits += 1
        else:
            matching_cards = [card for card in response.results if card.id in set(example.expected_skill_ids)]
            if matching_cards and matching_cards[0].read_recommendation == example.expected_section:
                section_hits += 1

    total = len(examples)
    return RetrievalEvalResult(
        query_count=total,
        top_k=top_k,
        recall_at_k=round(recall_sum / total, 4),
        mrr=round(reciprocal_rank_sum / total, 4),
        precision_at_k=round(precision_sum / total, 4),
        f1_at_k=round(f1_sum / total, 4),
        irrelevant_selection_rate=round(irrelevant_returned / returned_count, 4) if returned_count else 0.0,
        correct_section_recommendation_at_k=round(section_hits / total, 4),
        misses=misses,
    )


def score_retrieval_result(
    example: RetrievalExample, ranked_ids: list[str], top_k: int
) -> dict[str, float | bool | int]:
    expected = set(example.expected_skill_ids)
    returned = ranked_ids

    relevance = example.relevance_by_id or {}
    irrelevant_returned = sum(1 for skill_id in returned if relevance.get(skill_id) == "irrelevant")

    if example.allow_no_result:
        hit = not returned
        return {
            "hit": hit,
            "recall": 1.0 if hit else 0.0,
            "precision": 1.0 if hit else 0.0,
            "f1": 1.0 if hit else 0.0,
            "reciprocal_rank": 1.0 if hit else 0.0,
            "irrelevant_returned": irrelevant_returned,
        }

    relevant_returned = [skill_id for skill_id in returned if skill_id in expected]
    recall = len(set(relevant_returned)) / len(expected) if expected else 0.0
    precision = len(set(relevant_returned)) / top_k
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    reciprocal_rank = 0.0
    for rank, skill_id in enumerate(returned, start=1):
        if skill_id in expected:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "hit": bool(relevant_returned),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "reciprocal_rank": reciprocal_rank,
        "irrelevant_returned": irrelevant_returned,
    }
