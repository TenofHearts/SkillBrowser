"""Command runners for ToolRet candidate generation and benchmark evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from config import load_app_config_if_exists
from core.search import SkillSearcher
from llm import LLMClient

from .toolret import (
    NVEmbedV1Embedder,
    build_toolret_first_stage_candidates,
    compare_toolret_hybrid_with_paper_llm,
    evaluate_toolret_llm_rerank,
    evaluate_toolret_paper_llm_baseline,
    evaluate_toolret_retrieval,
    load_toolret_first_stage_candidates,
    load_toolret_queries,
    load_toolret_tools,
)


BuildLLM = Callable[[str, str], LLMClient]


def run_build_toolret_candidates(args: argparse.Namespace):
    examples = load_toolret_queries(
        args.queries,
        category=args.category,
        subset=args.subset,
        limit=args.limit,
    )
    tool_skills = load_toolret_tools(args.tools)
    embedder = NVEmbedV1Embedder(
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    return build_toolret_first_stage_candidates(
        tool_skills,
        examples,
        embedder,
        top_k=args.top_k,
        use_instruction=args.use_instruction,
        output_path=args.output,
    )


def run_eval_toolret(args: argparse.Namespace, build_llm: BuildLLM):
    config = load_app_config_if_exists(args.config)
    toolret_config = config.toolret
    queries = args.queries or toolret_config.queries
    tools = args.tools or toolret_config.tools
    if not queries:
        raise ValueError("ToolRet queries must be set by --queries or toolret.queries")
    if not tools:
        raise ValueError("ToolRet tools must be set by --tools or toolret.tools")

    first_stage_candidates_path = (
        args.first_stage_candidates
        if args.first_stage_candidates is not None
        else toolret_config.first_stage_candidates
    )
    top_k = args.top_k if args.top_k is not None else toolret_config.top_k
    limit = args.limit if args.limit is not None else toolret_config.limit
    subset = args.subset if args.subset is not None else toolret_config.subset
    category = args.category if args.category is not None else toolret_config.category
    use_instruction = (
        args.use_instruction
        if args.use_instruction is not None
        else toolret_config.use_instruction
    )
    baseline = args.baseline or toolret_config.baseline
    workers = args.workers if args.workers is not None else toolret_config.workers
    output = args.output or toolret_config.output

    examples = load_toolret_queries(queries, category=category, subset=subset, limit=limit)
    tool_skills = load_toolret_tools(tools)
    searcher = SkillSearcher(tool_skills)

    if baseline == "hybrid":
        result = evaluate_toolret_retrieval(
            searcher,
            examples,
            top_k=top_k,
            use_instruction=use_instruction,
            workers=workers,
        )
    elif baseline in {"rankgpt", "llm-rerank", "toolret-rankgpt", "paper-rankgpt", "compare"}:
        result = _run_llm_toolret_eval(
            args,
            build_llm,
            toolret_config,
            baseline,
            searcher,
            tool_skills,
            examples,
            top_k,
            use_instruction,
            workers,
            first_stage_candidates_path,
        )
    else:
        raise ValueError(f"Unsupported ToolRet baseline: {baseline}")

    if output:
        Path(output).write_text(result.json(indent=2), encoding="utf-8")
    return result


def _run_llm_toolret_eval(
    args: argparse.Namespace,
    build_llm: BuildLLM,
    toolret_config,
    baseline: str,
    searcher: SkillSearcher,
    tool_skills,
    examples,
    top_k: int,
    use_instruction: bool,
    workers: int,
    first_stage_candidates_path: str | None,
):
    llm_mode = args.llm or toolret_config.llm
    llm = build_llm(llm_mode, args.config)
    candidate_pool_size = (
        args.candidate_pool_size
        if args.candidate_pool_size is not None
        else toolret_config.candidate_pool_size
    )
    window_size = (
        args.rankgpt_window_size
        if args.rankgpt_window_size is not None
        else toolret_config.rankgpt_window_size
    )
    step_size = (
        args.rankgpt_step_size
        if args.rankgpt_step_size is not None
        else toolret_config.rankgpt_step_size
    )
    first_stage_candidates = (
        load_toolret_first_stage_candidates(first_stage_candidates_path)
        if first_stage_candidates_path
        else None
    )

    if baseline == "compare":
        if first_stage_candidates is None:
            raise ValueError(
                "ToolRet hybrid-vs-LLM comparison requires --first-stage-candidates "
                "from NV-Embed-v1 for the LLM side"
            )
        return compare_toolret_hybrid_with_paper_llm(
            searcher,
            tool_skills,
            examples,
            llm,
            top_k=top_k,
            use_instruction=use_instruction,
            candidate_pool_size=candidate_pool_size,
            first_stage_candidates=first_stage_candidates,
            window_size=window_size,
            step_size=step_size,
            workers=workers,
        )

    if baseline in {"toolret-rankgpt", "paper-rankgpt"}:
        if first_stage_candidates is None:
            raise ValueError(
                "ToolRet paper RankGPT baseline requires --first-stage-candidates "
                "from NV-Embed-v1; it does not use the hybrid retriever fallback"
            )
        return evaluate_toolret_paper_llm_baseline(
            tool_skills,
            examples,
            llm,
            top_k=top_k,
            use_instruction=use_instruction,
            candidate_pool_size=candidate_pool_size,
            first_stage_candidates=first_stage_candidates,
            window_size=window_size,
            step_size=step_size,
            workers=workers,
        )

    return evaluate_toolret_llm_rerank(
        searcher,
        tool_skills,
        examples,
        llm,
        top_k=top_k,
        use_instruction=use_instruction,
        candidate_pool_size=candidate_pool_size,
        first_stage_candidates=first_stage_candidates,
        window_size=window_size,
        step_size=step_size,
        workers=workers,
    )
