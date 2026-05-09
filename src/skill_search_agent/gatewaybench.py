from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

from .evaluation import RetrievalEvalResult, RetrievalExample, score_retrieval_result
from .llm import LLMClient
from .schema import CandidateTool, SkillSearchRequest, SkillSpec, ToolSelectionRequest
from .search import SkillSearcher
from .selectors import HybridSearchToolSelector, ToolSelector


SUPPORTED_TASK_TYPES = {"tool-heavy", "retrieval", "stress"}


class GatewayBenchExample(BaseModel):
    id: str
    user_prompt: str
    task_type: str
    relevance_by_name: dict[str, Optional[str]] = Field(default_factory=dict)
    ideal_tool_subset: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GatewayBenchSelectorResult(BaseModel):
    selector: str
    query_count: int
    top_k: int
    recall_at_k: float
    mrr: float
    precision_at_k: float
    f1_at_k: float
    irrelevant_selection_rate: float
    parse_failure_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: int = 0
    avg_latency_ms: float = 0.0
    wall_time_ms: int = 0
    token_usage_sources: dict[str, int] = Field(default_factory=dict)
    misses: list[dict[str, Any]] = Field(default_factory=list)


class GatewayBenchComparisonResult(BaseModel):
    query_count: int
    top_k: int
    results: dict[str, GatewayBenchSelectorResult]


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
        search_terms = _gateway_tool_search_terms(tool_name)
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
                        "long": (
                            f"Candidate tool from the {domain} domain for GatewayBench retrieval evaluation. "
                            f"Search aliases: {search_terms}."
                        ),
                    },
                    "capabilities": [
                        {
                            "id": re.sub(r"[^a-z0-9]+", "_", searchable_name.lower()).strip("_") or "tool",
                            "description": f"{searchable_name} {search_terms}",
                        }
                    ],
                    "interaction": {
                        "mode": "execute_directly",
                        "readable": True,
                        "executable": False,
                        "default_read_level": "overview",
                    },
                    "content": {"format": "markdown", "path": "skill.md", "sections": ["overview"]},
                    "when_to_use": [f"Use when the task requires {searchable_name} or {search_terms}."],
                    "when_not_to_use": [],
                    "input_types": [],
                    "output_types": [],
                    "examples": {"positive": [{"user_query": f"{searchable_name} {search_terms}"}]},
                    "execution": {"mode": "none"},
                    "tags": ["gatewaybench", domain],
                }
            )
        )
    return skills


def gateway_example_to_candidate_tools(example: GatewayBenchExample) -> list[CandidateTool]:
    candidates = []
    for tool_name in sorted(set(example.relevance_by_name) | set(example.ideal_tool_subset)):
        searchable_name = tool_name.replace("_", " ").replace("-", " ")
        candidates.append(
            CandidateTool(
                id=gateway_tool_id(tool_name),
                name=searchable_name,
                description=f"GatewayBench candidate tool: {searchable_name}.",
            )
        )
    return candidates


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


def evaluate_gatewaybench_selector(
    examples: list[GatewayBenchExample],
    selector_name: str,
    selector_factory: Any,
    top_k: int,
    workers: int = 1,
    progress: bool = False,
) -> GatewayBenchSelectorResult:
    recall_sum = 0.0
    precision_sum = 0.0
    f1_sum = 0.0
    reciprocal_rank_sum = 0.0
    irrelevant_returned = 0
    returned_count = 0
    parse_failure_count = 0
    input_tokens = 0
    output_tokens = 0
    total_latency_ms = 0
    token_usage_sources: dict[str, int] = {}
    misses: list[dict[str, Any]] = []

    total = len(examples)
    completed = 0
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as executor:
        futures = [
            executor.submit(_evaluate_gatewaybench_selector_example, gateway_example, selector_factory, top_k)
            for gateway_example in examples
        ]
        for future in as_completed(futures):
            item = future.result()
            stats = item["stats"]
            ranked_ids = item["ranked_ids"]
            retrieval_example = item["retrieval_example"]
            gateway_example = item["gateway_example"]
            parse_error = item["parse_error"]
            input_tokens += item["input_tokens"]
            output_tokens += item["output_tokens"]
            total_latency_ms += item["latency_ms"]
            usage_source = item["token_usage_source"]
            token_usage_sources[usage_source] = token_usage_sources.get(usage_source, 0) + 1

            completed += 1
            if progress:
                print(
                    f"[{selector_name}] {completed}/{total} {gateway_example.id} "
                    f"hit={stats['hit']} parse_error={bool(parse_error)} "
                    f"in={item['input_tokens']} out={item['output_tokens']} "
                    f"latency_ms={item['latency_ms']}",
                    file=sys.stderr,
                    flush=True,
                )

            if parse_error:
                parse_failure_count += 1
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
                        "parse_error": parse_error,
                    }
                )

    return GatewayBenchSelectorResult(
        selector=selector_name,
        query_count=total,
        top_k=top_k,
        recall_at_k=round(recall_sum / total, 4),
        mrr=round(reciprocal_rank_sum / total, 4),
        precision_at_k=round(precision_sum / total, 4),
        f1_at_k=round(f1_sum / total, 4),
        irrelevant_selection_rate=round(irrelevant_returned / returned_count, 4) if returned_count else 0.0,
        parse_failure_count=parse_failure_count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        total_latency_ms=total_latency_ms,
        avg_latency_ms=round(total_latency_ms / total, 2) if total else 0.0,
        wall_time_ms=round((time.perf_counter() - started) * 1000),
        token_usage_sources=token_usage_sources,
        misses=misses,
    )


def compare_gatewaybench_selectors(
    examples: list[GatewayBenchExample],
    selector_factories: dict[str, Any],
    top_k: int,
    workers: int = 1,
    progress: bool = False,
) -> GatewayBenchComparisonResult:
    return GatewayBenchComparisonResult(
        query_count=len(examples),
        top_k=top_k,
        results={
            name: evaluate_gatewaybench_selector(examples, name, factory, top_k, workers=workers, progress=progress)
            for name, factory in selector_factories.items()
        },
    )


def make_gatewaybench_hybrid_selector(example: GatewayBenchExample, llm: LLMClient) -> ToolSelector:
    return HybridSearchToolSelector(SkillSearcher(gateway_example_to_skills(example)), llm)


def _evaluate_gatewaybench_selector_example(
    gateway_example: GatewayBenchExample,
    selector_factory: Any,
    top_k: int,
) -> dict[str, Any]:
    retrieval_example = gateway_example_to_retrieval_example(gateway_example)
    selector = selector_factory(gateway_example)
    started = time.perf_counter()
    selection = selector.select(
        ToolSelectionRequest(
            prompt=gateway_example.user_prompt,
            candidates=gateway_example_to_candidate_tools(gateway_example),
            top_k=top_k,
        )
    )
    ranked_ids = selection.ranked_tool_ids[:top_k]
    elapsed_ms = selection.latency_ms
    if elapsed_ms <= 0:
        elapsed_ms = round((time.perf_counter() - started) * 1000)
    return {
        "gateway_example": gateway_example,
        "retrieval_example": retrieval_example,
        "ranked_ids": ranked_ids,
        "parse_error": selection.parse_error,
        "input_tokens": selection.input_tokens,
        "output_tokens": selection.output_tokens,
        "latency_ms": elapsed_ms,
        "token_usage_source": selection.token_usage_source,
        "stats": score_retrieval_result(retrieval_example, ranked_ids, top_k),
    }


def _normalize_relevance(label: Optional[str], is_required: bool) -> str:
    if is_required:
        return "required"
    if label in {"required", "useful", "irrelevant"}:
        return label
    return "irrelevant"


def _gateway_tool_search_terms(tool_name: str) -> str:
    raw_terms = [term for term in re.split(r"[^a-zA-Z0-9]+", tool_name.lower()) if term]
    aliases: list[str] = []
    for term in raw_terms:
        if term not in aliases:
            aliases.append(term)
        if term.endswith("ies") and len(term) > 3:
            singular = f"{term[:-3]}y"
            if singular not in aliases:
                aliases.append(singular)
        elif term.endswith("s") and len(term) > 3:
            singular = term[:-1]
            if singular not in aliases:
                aliases.append(singular)
        else:
            plural = f"{term}s"
            if plural not in aliases:
                aliases.append(plural)
    return " ".join(aliases)
