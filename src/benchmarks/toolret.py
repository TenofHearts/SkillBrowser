from __future__ import annotations

import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

from core.search import SkillSearcher
from core.selectors import extract_json_object_text
from llm import LLMClient
from schema import (
    SkillSearchRequest,
    SkillSpec,
)


TOOLRET_ID_PREFIX = "toolret."
TOOLRET_CATEGORIES = {"web", "code", "customized"}


class ToolRetQueryExample(BaseModel):
    id: str
    query: str
    instruction: Optional[str] = None
    labels: list[dict[str, Any]] = Field(default_factory=list)
    category: str = "unknown"
    subset: Optional[str] = None


class ToolRetEvalResult(BaseModel):
    query_count: int
    top_k: int
    use_instruction: bool
    recall_at: dict[str, float]
    precision_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    completeness_at_10: float
    parse_failure_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: int = 0
    avg_latency_ms: float = 0.0
    wall_time_ms: int = 0
    token_usage_sources: dict[str, int] = Field(default_factory=dict)
    by_category: dict[str, dict[str, float]] = Field(default_factory=dict)
    misses: list[dict[str, Any]] = Field(default_factory=list)


class ToolRetCandidateDoc(BaseModel):
    id: str
    name: str
    documentation: str


def load_toolret_tools(path: str | Path) -> list[SkillSpec]:
    rows = _load_rows(path)
    skills = [toolret_tool_to_skill(row) for row in rows]
    if not skills:
        raise ValueError(f"No ToolRet tools found in {path}")
    return skills


def load_toolret_queries(
    path: str | Path,
    *,
    category: str = "all",
    subset: str | None = None,
    limit: int | None = None,
) -> list[ToolRetQueryExample]:
    if category != "all" and category not in TOOLRET_CATEGORIES:
        raise ValueError(f"Unsupported ToolRet category: {category}")

    examples: list[ToolRetQueryExample] = []
    for row in _load_rows(path):
        example = toolret_query_from_row(row)
        if category != "all" and example.category != category:
            continue
        if subset and not _matches_subset(example, subset):
            continue
        if not get_toolret_gold_skill_ids(example):
            continue
        examples.append(example)
        if limit is not None and len(examples) >= limit:
            break
    if not examples:
        raise ValueError(f"No ToolRet queries found in {path}")
    return examples


def load_toolret_first_stage_candidates(path: str | Path) -> dict[str, list[str]]:
    rows = _load_rows(path)
    if isinstance(rows, dict):
        return _candidate_mapping_from_dict(rows)

    candidates: dict[str, list[str]] = {}
    for row in rows:
        query_id = str(row.get("id") or row.get("query_id") or row.get("qid") or "").strip()
        if not query_id:
            raise ValueError(f"ToolRet candidate row is missing query id: {row}")
        raw_tools = row.get("tools", row.get("candidates", row.get("candidate_tool_ids", [])))
        candidates[query_id] = _candidate_ids_from_value(raw_tools)
    if not candidates:
        raise ValueError(f"No ToolRet first-stage candidates found in {path}")
    return candidates


def toolret_tool_to_skill(row: dict[str, Any]) -> SkillSpec:
    raw_id = str(row.get("id", "")).strip()
    if not raw_id:
        raise ValueError(f"ToolRet tool row is missing id: {row}")
    doc = _parse_documentation(row.get("documentation", row.get("doc", {})))
    doc_text = _document_text(doc)
    name = _tool_name(raw_id, doc)
    description = _first_text(
        doc,
        ["description", "functionality", "summary", "api_call", "expressions", "domain"],
        fallback=doc_text[:300] or name,
    )
    capability_text = _first_text(
        doc,
        ["functionality", "description", "api_call", "expressions", "parameters"],
        fallback=description,
    )
    category = str(doc.get("category") or doc.get("domain") or row.get("category") or "toolret")
    schema = doc.get("parameters") if isinstance(doc.get("parameters"), dict) else None
    return SkillSpec.parse_obj(
        {
            "id": toolret_skill_id(raw_id),
            "name": name,
            "version": "0.1.0",
            "status": "active",
            "skill_type": "tool_wrapper",
            "category": {"primary": category, "secondary": ["toolret", "tool_retrieval"]},
            "description": {
                "short": _compact_text(description, 500),
                "long": _compact_text(doc_text, 4000),
            },
            "capabilities": [
                {
                    "id": _slugify(name) or "tool",
                    "description": _compact_text(capability_text, 1000),
                }
            ],
            "interaction": {
                "mode": "execute_directly",
                "readable": True,
                "executable": False,
                "default_read_level": "overview",
            },
            "content": {"format": "markdown", "path": "skill.md", "sections": ["overview"]},
            "when_to_use": [_compact_text(f"Use for ToolRet tool {name}. {description}", 1000)],
            "when_not_to_use": [],
            "input_types": _tool_input_types(doc),
            "output_types": _tool_output_types(doc),
            "input_schema": schema,
            "examples": {"positive": [{"user_query": _compact_text(capability_text, 500)}]},
            "execution": {"mode": "none"},
            "tags": ["toolret", category],
        }
    )


def toolret_query_from_row(row: dict[str, Any]) -> ToolRetQueryExample:
    labels = row.get("labels", [])
    if isinstance(labels, str):
        labels = json.loads(labels)
    if not isinstance(labels, list):
        raise ValueError(f"ToolRet query labels must be a list: {row}")
    return ToolRetQueryExample(
        id=str(row.get("id", "")).strip(),
        query=str(row.get("query", "")).strip(),
        instruction=str(row["instruction"]).strip() if row.get("instruction") is not None else None,
        labels=labels,
        category=str(row.get("category", "unknown")).strip() or "unknown",
        subset=str(row["subset"]).strip() if row.get("subset") is not None else None,
    )


def toolret_query_to_search_request(
    example: ToolRetQueryExample, *, top_k: int, use_instruction: bool = True
) -> SkillSearchRequest:
    query = example.query
    task_context = example.instruction if use_instruction and example.instruction else None
    return SkillSearchRequest(query=query, task_context=task_context, top_k=top_k)


def get_toolret_gold_skill_ids(example: ToolRetQueryExample) -> list[str]:
    ids: list[str] = []
    for label in example.labels:
        if not isinstance(label, dict):
            continue
        raw_id = label.get("id")
        if isinstance(raw_id, str) and raw_id.strip():
            skill_id = toolret_skill_id(raw_id)
            if skill_id not in ids:
                ids.append(skill_id)
    return ids


def evaluate_toolret_retrieval(
    searcher: SkillSearcher,
    examples: list[ToolRetQueryExample],
    *,
    top_k: int = 10,
    use_instruction: bool = True,
    workers: int = 1,
) -> ToolRetEvalResult:
    return _evaluate_toolret(
        examples,
        top_k=top_k,
        use_instruction=use_instruction,
        runner=lambda example: _run_hybrid_toolret(searcher, example, top_k, use_instruction),
        workers=workers,
    )


def evaluate_toolret_llm_rerank(
    searcher: SkillSearcher,
    skills: list[SkillSpec],
    examples: list[ToolRetQueryExample],
    llm: LLMClient,
    *,
    top_k: int = 10,
    use_instruction: bool = True,
    candidate_pool_size: int = 50,
    first_stage_candidates: dict[str, list[str]] | None = None,
    window_size: int = 20,
    step_size: int = 10,
    workers: int = 1,
) -> ToolRetEvalResult:
    pool_size = max(candidate_pool_size, top_k)
    skill_by_id = {skill.id: skill for skill in skills}
    return _evaluate_toolret(
        examples,
        top_k=top_k,
        use_instruction=use_instruction,
        runner=lambda example: _run_rankgpt_toolret(
            searcher,
            skill_by_id,
            llm,
            example,
            top_k,
            use_instruction,
            pool_size,
            first_stage_candidates,
            window_size,
            step_size,
        ),
        workers=workers,
    )


def score_toolret_ranking(gold_ids: list[str], ranked_ids: list[str]) -> dict[str, float]:
    gold = set(gold_ids)
    if not gold:
        return {
            "recall@1": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "precision@10": 0.0,
            "mrr@10": 0.0,
            "ndcg@10": 0.0,
            "completeness@10": 0.0,
        }
    return {
        "recall@1": _recall_at(gold, ranked_ids, 1),
        "recall@3": _recall_at(gold, ranked_ids, 3),
        "recall@5": _recall_at(gold, ranked_ids, 5),
        "recall@10": _recall_at(gold, ranked_ids, 10),
        "precision@10": _precision_at(gold, ranked_ids, 10),
        "mrr@10": _mrr_at(gold, ranked_ids, 10),
        "ndcg@10": _ndcg_at(gold, ranked_ids, 10),
        "completeness@10": 1.0 if gold.issubset(set(ranked_ids[:10])) else 0.0,
    }


def toolret_skill_id(raw_id: str) -> str:
    if raw_id.startswith(TOOLRET_ID_PREFIX):
        return raw_id
    return f"{TOOLRET_ID_PREFIX}{raw_id}"


def _evaluate_toolret(
    examples: list[ToolRetQueryExample],
    *,
    top_k: int,
    use_instruction: bool,
    runner: Any,
    workers: int = 1,
) -> ToolRetEvalResult:
    metric_names = ["recall@1", "recall@3", "recall@5", "recall@10", "precision@10", "mrr@10", "ndcg@10", "completeness@10"]
    totals = {name: 0.0 for name in metric_names}
    category_totals: dict[str, dict[str, float]] = {}
    category_counts: dict[str, int] = {}
    misses: list[dict[str, Any]] = []
    parse_failure_count = 0
    input_tokens = 0
    output_tokens = 0
    total_latency_ms = 0
    token_usage_sources: dict[str, int] = {}

    started = time.perf_counter()
    items: list[tuple[ToolRetQueryExample, dict[str, Any]]] = []
    if workers <= 1:
        items = [(example, runner(example)) for example in examples]
    else:
        with ThreadPoolExecutor(max_workers=max(workers, 1)) as executor:
            futures = {executor.submit(runner, example): example for example in examples}
            for future in as_completed(futures):
                items.append((futures[future], future.result()))

    for example, item in items:
        ranked_ids = item["ranked_ids"][:top_k]
        gold_ids = get_toolret_gold_skill_ids(example)
        stats = score_toolret_ranking(gold_ids, ranked_ids)
        for name in metric_names:
            totals[name] += stats[name]
        category = example.category
        category_counts[category] = category_counts.get(category, 0) + 1
        category_bucket = category_totals.setdefault(category, {name: 0.0 for name in metric_names})
        for name in metric_names:
            category_bucket[name] += stats[name]

        parse_error = item.get("parse_error")
        if parse_error:
            parse_failure_count += 1
        input_tokens += item.get("input_tokens", 0)
        output_tokens += item.get("output_tokens", 0)
        total_latency_ms += item.get("latency_ms", 0)
        usage_source = item.get("token_usage_source", "none")
        token_usage_sources[usage_source] = token_usage_sources.get(usage_source, 0) + 1

        if stats["recall@10"] <= 0:
            misses.append(
                {
                    "id": example.id,
                    "query": example.query,
                    "category": example.category,
                    "expected_skill_ids": gold_ids,
                    "returned_skill_ids": ranked_ids,
                    "parse_error": parse_error,
                    "score_breakdowns": item.get("score_breakdowns", {}),
                }
            )

    total = len(examples)
    by_category = {
        category: {name: round(value / category_counts[category], 4) for name, value in values.items()}
        for category, values in category_totals.items()
    }
    return ToolRetEvalResult(
        query_count=total,
        top_k=top_k,
        use_instruction=use_instruction,
        recall_at={
            "1": round(totals["recall@1"] / total, 4),
            "3": round(totals["recall@3"] / total, 4),
            "5": round(totals["recall@5"] / total, 4),
            "10": round(totals["recall@10"] / total, 4),
        },
        precision_at_10=round(totals["precision@10"] / total, 4),
        mrr_at_10=round(totals["mrr@10"] / total, 4),
        ndcg_at_10=round(totals["ndcg@10"] / total, 4),
        completeness_at_10=round(totals["completeness@10"] / total, 4),
        parse_failure_count=parse_failure_count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        total_latency_ms=total_latency_ms,
        avg_latency_ms=round(total_latency_ms / total, 2) if total else 0.0,
        wall_time_ms=round((time.perf_counter() - started) * 1000),
        token_usage_sources=token_usage_sources,
        by_category=by_category,
        misses=misses[:20],
    )


def _run_hybrid_toolret(
    searcher: SkillSearcher,
    example: ToolRetQueryExample,
    top_k: int,
    use_instruction: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = searcher.search(toolret_query_to_search_request(example, top_k=top_k, use_instruction=use_instruction))
    return {
        "ranked_ids": [card.id for card in response.results],
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "token_usage_source": "none",
        "score_breakdowns": {
            card.id: card.score_breakdown.dict()
            for card in response.results[:10]
        },
    }


def _run_rankgpt_toolret(
    searcher: SkillSearcher,
    skill_by_id: dict[str, SkillSpec],
    llm: LLMClient,
    example: ToolRetQueryExample,
    top_k: int,
    use_instruction: bool,
    candidate_pool_size: int,
    first_stage_candidates: dict[str, list[str]] | None,
    window_size: int,
    step_size: int,
) -> dict[str, Any]:
    if first_stage_candidates is not None:
        candidate_ids = [
            skill_id
            for skill_id in first_stage_candidates.get(example.id, [])[:candidate_pool_size]
            if skill_id in skill_by_id
        ]
        score_breakdowns = {}
        first_stage_source = "provided"
    else:
        request_top_k = min(candidate_pool_size, 50)
        request = toolret_query_to_search_request(example, top_k=request_top_k, use_instruction=use_instruction)
        search_response = searcher.search(request)
        candidate_ids = [card.id for card in search_response.results]
        score_breakdowns = {card.id: card.score_breakdown.dict() for card in search_response.results[:10]}
        first_stage_source = "hybrid-fallback"

    candidates = [_toolret_candidate_doc(skill_by_id[skill_id]) for skill_id in candidate_ids]
    if not candidates:
        return {"ranked_ids": [], "token_usage_source": "none", "score_breakdowns": {}}
    ranked_ids, stats = _rankgpt_rerank(
        llm,
        query=example.query,
        instruction=example.instruction if use_instruction else None,
        candidates=candidates,
        top_k=top_k,
        window_size=window_size,
        step_size=step_size,
    )
    fallback_ids = [candidate.id for candidate in candidates if candidate.id not in ranked_ids]
    ranked_ids = [*ranked_ids, *fallback_ids]
    return {
        "ranked_ids": ranked_ids[:top_k],
        "parse_error": stats["parse_error"],
        "input_tokens": stats["input_tokens"],
        "output_tokens": stats["output_tokens"],
        "latency_ms": stats["latency_ms"],
        "token_usage_source": stats["token_usage_source"],
        "score_breakdowns": score_breakdowns,
        "first_stage_source": first_stage_source,
    }


def _rankgpt_rerank(
    llm: LLMClient,
    *,
    query: str,
    instruction: str | None,
    candidates: list[ToolRetCandidateDoc],
    top_k: int,
    window_size: int,
    step_size: int,
) -> tuple[list[str], dict[str, Any]]:
    if not candidates:
        return [], _empty_rankgpt_stats()

    ranked = list(candidates)
    parse_errors: list[str] = []
    input_tokens = 0
    output_tokens = 0
    latency_ms = 0
    usage_sources: list[str] = []
    window_size = max(window_size, top_k, 1)
    step_size = max(step_size, 1)

    # RankGPT reranks local windows from the tail toward the head, allowing better
    # lower-ranked candidates to bubble upward without sending every candidate at once.
    start_positions = list(range(max(len(ranked) - window_size, 0), -1, -step_size))
    if not start_positions or start_positions[-1] != 0:
        start_positions.append(0)
    for start in start_positions:
        end = min(start + window_size, len(ranked))
        window = ranked[start:end]
        messages = _rankgpt_messages(query=query, instruction=instruction, candidates=window)
        completion = llm.complete_with_usage(messages)
        input_tokens += completion.input_tokens
        output_tokens += completion.output_tokens
        latency_ms += completion.elapsed_ms
        usage_sources.append(completion.token_usage_source)

        ordered_ids, error = _parse_rankgpt_ids(completion.content, [candidate.id for candidate in window])
        if error:
            parse_errors.append(error)
            continue
        by_id = {candidate.id: candidate for candidate in window}
        ordered_window = [by_id[skill_id] for skill_id in ordered_ids if skill_id in by_id]
        ordered_window.extend(candidate for candidate in window if candidate.id not in set(ordered_ids))
        ranked[start:end] = ordered_window

    return [candidate.id for candidate in ranked[:top_k]], {
        "parse_error": "; ".join(parse_errors) if parse_errors else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "token_usage_source": _merge_token_usage_sources(usage_sources),
    }


def _rankgpt_messages(
    *,
    query: str,
    instruction: str | None,
    candidates: list[ToolRetCandidateDoc],
) -> list[dict[str, str]]:
    formatted_query = f"Instruct: {instruction}\nQuery: {query}" if instruction else query
    return [
        {
            "role": "system",
            "content": (
                "You are RankGPT, a zero-shot reranking agent for tool retrieval. "
                "Given a query and a numbered list of candidate tool documents, rerank all candidates by usefulness. "
                "Return strict JSON only: {\"ranking\": [1, 2, 3]}. Include every candidate number exactly once."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "query": formatted_query,
                    "candidate_tools": [
                        {
                            "number": index,
                            "id": candidate.id,
                            "documentation": _compact_text(candidate.documentation, 1200),
                        }
                        for index, candidate in enumerate(candidates, start=1)
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]


def _parse_rankgpt_ids(raw: str, valid_ids: list[str]) -> tuple[list[str], str | None]:
    text = raw.strip()
    try:
        payload = json.loads(extract_json_object_text(text))
    except json.JSONDecodeError:
        payload = None

    numbers: list[int] = []
    if isinstance(payload, dict):
        ranking = payload.get("ranking", payload.get("ranked_numbers", payload.get("permutation")))
        if ranking is None:
            ranked_ids = payload.get("ranked_tool_ids")
            if isinstance(ranked_ids, list):
                valid = [skill_id for skill_id in ranked_ids if isinstance(skill_id, str) and skill_id in valid_ids]
                if valid:
                    return _dedupe_keep_order(valid), None
        numbers = _numbers_from_value(ranking)

    if not numbers:
        numbers = [int(match) for match in re.findall(r"\b\d+\b", text)]

    ranked_ids: list[str] = []
    for number in numbers:
        if 1 <= number <= len(valid_ids):
            skill_id = valid_ids[number - 1]
            if skill_id not in ranked_ids:
                ranked_ids.append(skill_id)
    if not ranked_ids:
        return [], "RankGPT output did not contain valid candidate numbers or ids"
    return ranked_ids, None


def _toolret_candidate_doc(skill: SkillSpec) -> ToolRetCandidateDoc:
    documentation = {
        "name": skill.name,
        "description": skill.description.long or skill.description.short,
        "parameters": skill.input_schema or {},
        "capabilities": [capability.dict() for capability in skill.capabilities],
    }
    return ToolRetCandidateDoc(
        id=skill.id,
        name=skill.name,
        documentation=json.dumps(documentation, ensure_ascii=False, sort_keys=True),
    )


def _candidate_mapping_from_dict(data: dict[str, Any]) -> dict[str, list[str]]:
    candidates: dict[str, list[str]] = {}
    for query_id, raw_tools in data.items():
        candidates[str(query_id)] = _candidate_ids_from_value(raw_tools)
    if not candidates:
        raise ValueError("ToolRet first-stage candidate mapping is empty")
    return candidates


def _candidate_ids_from_value(raw_tools: Any) -> list[str]:
    if isinstance(raw_tools, dict):
        items = sorted(raw_tools.items(), key=lambda item: item[1], reverse=True)
        return [toolret_skill_id(str(tool_id)) for tool_id, _score in items]
    if not isinstance(raw_tools, list):
        raise ValueError(f"ToolRet candidates must be a list or score mapping: {raw_tools}")
    scored: list[tuple[str, float, int]] = []
    plain: list[str] = []
    for index, item in enumerate(raw_tools):
        if isinstance(item, str):
            plain.append(toolret_skill_id(item))
        elif isinstance(item, dict):
            raw_id = item.get("id", item.get("tool_id"))
            if isinstance(raw_id, str) and raw_id.strip():
                score = item.get("score", item.get("relevance", item.get("rank_score", 0)))
                scored.append((toolret_skill_id(raw_id), float(score or 0), index))
    if scored:
        scored.sort(key=lambda item: (-item[1], item[2]))
        return _dedupe_keep_order([tool_id for tool_id, _score, _index in scored])
    return _dedupe_keep_order(plain)


def _numbers_from_value(value: Any) -> list[int]:
    if isinstance(value, str):
        return [int(match) for match in re.findall(r"\b\d+\b", value)]
    if isinstance(value, list):
        numbers: list[int] = []
        for item in value:
            if isinstance(item, int):
                numbers.append(item)
            elif isinstance(item, str) and item.strip().isdigit():
                numbers.append(int(item.strip()))
        return numbers
    return []


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _merge_token_usage_sources(sources: list[str]) -> str:
    unique = {source for source in sources if source}
    if not unique:
        return "none"
    if len(unique) == 1:
        return next(iter(unique))
    return "mixed"


def _empty_rankgpt_stats() -> dict[str, Any]:
    return {
        "parse_error": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0,
        "token_usage_source": "none",
    }


def _load_rows(path: str | Path) -> Any:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise ValueError(f"ToolRet data file not found: {dataset_path}")
    suffix = dataset_path.suffix.lower()
    if suffix == ".jsonl":
        return [
            json.loads(line)
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".json":
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data
        raise ValueError(f"ToolRet JSON file must contain a list or object: {dataset_path}")
    if suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ValueError(
                "Reading parquet ToolRet files requires pandas with a parquet engine such as pyarrow. "
                "Export to JSONL or install the optional parquet dependencies."
            ) from exc
        return pd.read_parquet(dataset_path).to_dict(orient="records")
    raise ValueError(f"Unsupported ToolRet data file format: {dataset_path}")


def _parse_documentation(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            return {"text": parsed}
        except json.JSONDecodeError:
            return {"text": text}
    return {"text": str(value)}


def _document_text(doc: Any) -> str:
    if isinstance(doc, dict):
        return json.dumps(doc, ensure_ascii=False, sort_keys=True)
    return str(doc)


def _tool_name(raw_id: str, doc: dict[str, Any]) -> str:
    for key in ["name", "function_name", "api_name", "tool_name", "api_call"]:
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return raw_id


def _first_text(doc: dict[str, Any], keys: Iterable[str], *, fallback: str) -> str:
    for key in keys:
        value = doc.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return fallback


def _tool_input_types(doc: dict[str, Any]) -> list[str]:
    params = doc.get("parameters") or doc.get("api_arguments")
    if isinstance(params, dict):
        return list(params.keys())[:20]
    if isinstance(params, list):
        return [str(item) for item in params[:20]]
    if isinstance(params, str) and params.strip():
        return [_compact_text(params, 200)]
    return []


def _tool_output_types(doc: dict[str, Any]) -> list[str]:
    output = doc.get("output") or doc.get("returns") or doc.get("output_type")
    if isinstance(output, dict):
        return list(output.keys())[:20]
    if isinstance(output, list):
        return [str(item) for item in output[:20]]
    if isinstance(output, str) and output.strip():
        return [_compact_text(output, 200)]
    return []


def _matches_subset(example: ToolRetQueryExample, subset: str) -> bool:
    if example.subset == subset:
        return True
    return example.id.startswith(f"{subset}_") or example.id.startswith(f"{subset}-")


def _recall_at(gold: set[str], ranked_ids: list[str], k: int) -> float:
    return len(gold & set(ranked_ids[:k])) / len(gold)


def _precision_at(gold: set[str], ranked_ids: list[str], k: int) -> float:
    return len(gold & set(ranked_ids[:k])) / k


def _mrr_at(gold: set[str], ranked_ids: list[str], k: int) -> float:
    for rank, skill_id in enumerate(ranked_ids[:k], start=1):
        if skill_id in gold:
            return 1.0 / rank
    return 0.0


def _ndcg_at(gold: set[str], ranked_ids: list[str], k: int) -> float:
    dcg = 0.0
    for rank, skill_id in enumerate(ranked_ids[:k], start=1):
        if skill_id in gold:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(gold), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 0.0


def _compact_text(text: str, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", str(text)).strip()
    return compact[:max_chars]


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
