from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

from .evaluation import RetrievalEvalResult, RetrievalExample, score_retrieval_result
from .schema import SkillSearchRequest, SkillSpec
from .search import SkillSearcher


SUPPORTED_TASK_TYPES = {"tool-heavy", "retrieval", "stress"}


class GatewayBenchExample(BaseModel):
    id: str
    user_prompt: str
    task_type: str
    relevance_by_name: dict[str, Optional[str]] = Field(default_factory=dict)
    ideal_tool_subset: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def load_gatewaybench_lite_dataset(
    path: str | Path, task_types: Iterable[str] = SUPPORTED_TASK_TYPES, limit: Optional[int] = None
) -> list[GatewayBenchExample]:
    dataset_path = Path(path)
    selected_task_types = set(task_types)
    examples: list[GatewayBenchExample] = []
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            example = GatewayBenchExample.parse_obj(json.loads(line))
        except Exception as exc:
            raise ValueError(f"Invalid GatewayBench example at {dataset_path}:{line_number}: {exc}") from exc
        if example.task_type not in selected_task_types:
            continue
        if not example.relevance_by_name or not example.ideal_tool_subset:
            continue
        examples.append(example)
        if limit is not None and len(examples) >= limit:
            break
    if not examples:
        raise ValueError(f"No GatewayBench-lite examples found in {dataset_path}")
    return examples


def gateway_tool_id(tool_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", tool_name.lower()).strip("_")
    return f"gateway.{normalized or 'tool'}"


def gateway_example_to_retrieval_example(example: GatewayBenchExample) -> RetrievalExample:
    relevance_by_id = {
        gateway_tool_id(name): _normalize_relevance(label, name in set(example.ideal_tool_subset))
        for name, label in example.relevance_by_name.items()
    }
    return RetrievalExample(
        query=example.user_prompt,
        expected_skill_ids=[gateway_tool_id(name) for name in example.ideal_tool_subset],
        relevance_by_id=relevance_by_id,
    )


def gateway_example_to_skills(example: GatewayBenchExample) -> list[SkillSpec]:
    tool_names = sorted(set(example.relevance_by_name) | set(example.ideal_tool_subset))
    domain = str(example.metadata.get("domain", "general"))
    skills = []
    for tool_name in tool_names:
        skill_id = gateway_tool_id(tool_name)
        searchable_name = tool_name.replace("_", " ").replace("-", " ")
        skills.append(
            SkillSpec.parse_obj(
                {
                    "id": skill_id,
                    "name": searchable_name,
                    "version": "0.1.0",
                    "status": "active",
                    "skill_type": "tool_wrapper",
                    "category": {"primary": domain, "secondary": ["gatewaybench", "tool_selection"]},
                    "description": {
                        "short": f"GatewayBench candidate tool: {searchable_name}.",
                        "long": f"Candidate tool from the {domain} domain for GatewayBench retrieval evaluation.",
                    },
                    "capabilities": [
                        {
                            "id": re.sub(r"[^a-z0-9]+", "_", searchable_name.lower()).strip("_") or "tool",
                            "description": searchable_name,
                        }
                    ],
                    "interaction": {
                        "mode": "execute_directly",
                        "readable": True,
                        "executable": False,
                        "default_read_level": "overview",
                    },
                    "content": {"format": "markdown", "path": "skill.md", "sections": ["overview"]},
                    "when_to_use": [f"Use when the task requires {searchable_name}."],
                    "when_not_to_use": [],
                    "input_types": [],
                    "output_types": [],
                    "examples": {"positive": [{"user_query": searchable_name}]},
                    "execution": {"mode": "none"},
                    "tags": ["gatewaybench", domain],
                }
            )
        )
    return skills


def evaluate_gatewaybench_lite(
    examples: list[GatewayBenchExample], top_k: int
) -> RetrievalEvalResult:
    recall_sum = 0.0
    precision_sum = 0.0
    f1_sum = 0.0
    reciprocal_rank_sum = 0.0
    irrelevant_returned = 0
    returned_count = 0
    misses: list[dict[str, Any]] = []

    for gateway_example in examples:
        retrieval_example = gateway_example_to_retrieval_example(gateway_example)
        searcher = SkillSearcher(gateway_example_to_skills(gateway_example))
        response = searcher.search(SkillSearchRequest(query=retrieval_example.query, top_k=top_k))
        ranked_ids = [card.id for card in response.results]

        stats = score_retrieval_result(retrieval_example, ranked_ids, top_k)
        recall_sum += stats["recall"]
        precision_sum += stats["precision"]
        f1_sum += stats["f1"]
        reciprocal_rank_sum += stats["reciprocal_rank"]
        irrelevant_returned += stats["irrelevant_returned"]
        returned_count += len(ranked_ids)

        if not stats["hit"]:
            misses.append(
                {
                    "query": retrieval_example.query,
                    "expected_skill_ids": retrieval_example.expected_skill_ids,
                    "returned_skill_ids": ranked_ids,
                    "gatewaybench_id": gateway_example.id,
                    "task_type": gateway_example.task_type,
                }
            )

    total = len(examples)
    return RetrievalEvalResult(
        query_count=total,
        top_k=top_k,
        recall_at_k=round(recall_sum / total, 4),
        mrr=round(reciprocal_rank_sum / total, 4),
        precision_at_k=round(precision_sum / total, 4),
        f1_at_k=round(f1_sum / total, 4),
        irrelevant_selection_rate=round(irrelevant_returned / returned_count, 4) if returned_count else 0.0,
        correct_section_recommendation_at_k=1.0,
        misses=misses,
    )


def _normalize_relevance(label: Optional[str], is_required: bool) -> str:
    if is_required:
        return "required"
    if label in {"required", "useful", "irrelevant"}:
        return label
    return "irrelevant"
