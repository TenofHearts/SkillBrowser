from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .schema import SkillSearchRequest
from .search import SkillSearcher


class RetrievalExample(BaseModel):
    query: str
    expected_skill_id: str
    expected_section: Optional[str] = None


class RetrievalEvalResult(BaseModel):
    query_count: int
    top_k: int
    recall_at_k: float
    mrr: float
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
    hits = 0
    reciprocal_rank_sum = 0.0
    section_hits = 0
    misses: list[dict[str, Any]] = []

    for example in examples:
        response = searcher.search(SkillSearchRequest(query=example.query, top_k=top_k))
        ranked_ids = [card.id for card in response.results]
        try:
            rank = ranked_ids.index(example.expected_skill_id) + 1
        except ValueError:
            misses.append(
                {
                    "query": example.query,
                    "expected_skill_id": example.expected_skill_id,
                    "returned_skill_ids": ranked_ids,
                }
            )
            continue

        hits += 1
        reciprocal_rank_sum += 1.0 / rank
        if example.expected_section is None:
            section_hits += 1
        else:
            card = response.results[rank - 1]
            if card.read_recommendation == example.expected_section:
                section_hits += 1

    total = len(examples)
    return RetrievalEvalResult(
        query_count=total,
        top_k=top_k,
        recall_at_k=round(hits / total, 4),
        mrr=round(reciprocal_rank_sum / total, 4),
        correct_section_recommendation_at_k=round(section_hits / total, 4),
        misses=misses,
    )
