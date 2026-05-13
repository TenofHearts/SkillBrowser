"""Command runners for ToolRet candidate generation and benchmark evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from config import load_app_config_if_exists
from core.embeddings import build_embedder
from core.search import SearchWeights, SkillSearcher
from llm import LLMClient

from .toolret import (
    HFTransformersEmbedder,
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
    config = load_app_config_if_exists(args.config)
    toolret_config = config.toolret
    examples = load_toolret_queries(
        args.queries,
        category=args.category,
        subset=args.subset,
        limit=args.limit,
    )
    tool_skills = load_toolret_tools(args.tools)
    model_name = args.model or toolret_config.first_stage_model
    backend = args.embedding_backend or toolret_config.first_stage_backend
    batch_size = args.batch_size if args.batch_size is not None else toolret_config.embed_batch_size
    max_length = args.max_length if args.max_length is not None else toolret_config.embed_max_length
    device = args.device if args.device is not None else (toolret_config.embed_device or None)
    embedder = _build_toolret_embedder(
        model_name=model_name,
        backend=backend,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    return build_toolret_first_stage_candidates(
        tool_skills,
        examples,
        embedder,
        top_k=args.top_k,
        use_instruction=args.use_instruction,
        output_path=args.output,
    )


def _build_toolret_embedder(
    *,
    model_name: str,
    backend: str,
    batch_size: int,
    max_length: int,
    device: str | None,
):
    resolved_backend = _resolve_embedding_backend(model_name, backend)
    if resolved_backend == "nv-embed":
        return NVEmbedV1Embedder(
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
    if resolved_backend == "hf-transformers":
        return HFTransformersEmbedder(
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
    raise ValueError(f"Unsupported ToolRet embedding backend: {backend}")


def _resolve_embedding_backend(model_name: str, backend: str) -> str:
    if backend != "auto":
        return backend
    normalized = model_name.replace("\\", "/").rstrip("/").lower()
    if normalized.endswith("nv-embed-v1") or normalized == "nvidia/nv-embed-v1":
        return "nv-embed"
    return "hf-transformers"


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
    searcher = _build_skill_searcher(tool_skills, args, config)

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


def _build_skill_searcher(tool_skills, args: argparse.Namespace, config):
    embedding_config = config.embedding
    mode = getattr(args, "retrieval_mode", None)
    backend = getattr(args, "embedding_backend", None) or embedding_config.backend
    dense_enabled = embedding_config.enabled and backend != "none"

    if mode == "bm25":
        dense_enabled = False
        backend = "none"
        bm25_enabled = True
        sparse_view_enabled = False
    elif mode == "dense":
        dense_enabled = True
        if backend == "none":
            backend = "hf-transformers"
        bm25_enabled = False
        sparse_view_enabled = False
    elif mode == "hybrid":
        dense_enabled = backend != "none"
        bm25_enabled = True
        sparse_view_enabled = True
    else:
        bm25_enabled = True
        sparse_view_enabled = True

    embedder = build_embedder(
        backend,
        model_name=getattr(args, "embedding_model", None) or embedding_config.model,
        batch_size=getattr(args, "embedding_batch_size", None) or embedding_config.batch_size,
        max_length=getattr(args, "embedding_max_length", None) or embedding_config.max_length,
        device=getattr(args, "embedding_device", None) or embedding_config.device,
    )
    return SkillSearcher(
        tool_skills,
        embedder=embedder,
        dense_enabled=dense_enabled,
        bm25_enabled=bm25_enabled,
        sparse_view_enabled=sparse_view_enabled,
        dense_view_names={"description"},
        weights=_build_search_weights(args),
        dense_cache_dir=args.embedding_cache_dir or "data/eval/toolret/embedding_cache",
    )


def _build_search_weights(args: argparse.Namespace) -> SearchWeights:
    defaults = SearchWeights()
    return SearchWeights(
        lexical=args.weight_lexical if args.weight_lexical is not None else defaults.lexical,
        sparse_view=args.weight_sparse_view if args.weight_sparse_view is not None else defaults.sparse_view,
        dense=args.weight_dense if args.weight_dense is not None else defaults.dense,
        rrf=args.weight_rrf if args.weight_rrf is not None else defaults.rrf,
        capability=args.weight_capability if args.weight_capability is not None else defaults.capability,
        usage=args.weight_usage if args.weight_usage is not None else defaults.usage,
        input_type=args.weight_input_type if args.weight_input_type is not None else defaults.input_type,
        output_type=args.weight_output_type if args.weight_output_type is not None else defaults.output_type,
        penalty=args.weight_penalty if args.weight_penalty is not None else defaults.penalty,
    )


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
