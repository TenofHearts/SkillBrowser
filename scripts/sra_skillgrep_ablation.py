"""Run a retrieval-only SkillGrep component ablation on SRA-Bench."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmarks.sra import (  # noqa: E402
    SRA_CORPUS_PATH,
    SRA_INSTANCES_DIR,
    SRA_RESULTS_DIR,
    SRA_SUBMODULE_DIR,
    build_sra_query,
    compute_sra_retrieval_metrics,
    ensure_sra_corpus,
    load_sra_corpus,
    load_sra_instances,
)
from cli import build_configured_embedder, build_minimum_score_threshold, build_search_weights  # noqa: E402
from config import load_app_config_if_exists  # noqa: E402
from core.search import SearchWeights, SkillSearcher  # noqa: E402
from schema import SkillSearchRequest  # noqa: E402
from sra_bench import SRA_METADATA_DENSE_VIEW_NAMES, load_sra_search_specs  # noqa: E402


DEFAULT_DATASETS = ["theoremqa", "logicbench", "champ", "medcalcbench"]
DEFAULT_OUTPUT_DIR = SRA_RESULTS_DIR / "ablation" / "skillgrep_components"


@dataclass(frozen=True)
class AblationVariant:
    name: str
    bm25_enabled: bool
    sparse_view_enabled: bool
    dense_enabled: bool
    weights: SearchWeights


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_skillgrep_ablation(args)
    print(json.dumps({"summary_json": payload["paths"]["summary_json"], "rows": len(payload["summary_rows"])}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--variants", nargs="+", help="Optional subset of ablation variants to run")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, help="Optional per-dataset instance limit for smoke tests")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--corpus", default=str(SRA_CORPUS_PATH))
    parser.add_argument("--sra-skill-dir", help="Optional preprocessed SRA SkillSpec directory")
    parser.add_argument("--instances-dir", default=str(SRA_INSTANCES_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--embedding-backend", choices=["none", "fake", "hf-transformers"])
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-batch-size", type=int)
    parser.add_argument("--embedding-max-length", type=int)
    parser.add_argument("--embedding-device")
    parser.add_argument("--embedding-cache-dir")
    parser.add_argument("--minimum-score-threshold", type=float)
    return parser


def run_skillgrep_ablation(args: argparse.Namespace) -> dict[str, Any]:
    if args.top_k != 5:
        raise ValueError("SkillGrep component ablation keeps top_k fixed at 5.")
    ensure_sra_corpus(args.corpus)
    corpus = load_sra_corpus(args.corpus)
    skills = load_sra_search_specs(args, corpus)
    has_preprocessed = bool(getattr(args, "sra_skill_dir", None)) or bool(
        load_app_config_if_exists(args.config).sra.skill_dirs
    )
    dense_view_names = SRA_METADATA_DENSE_VIEW_NAMES if has_preprocessed else None
    variants = build_ablation_variants(args)
    if args.variants:
        requested = set(args.variants)
        known = {variant.name for variant in variants}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown ablation variant(s): {', '.join(unknown)}")
        variants = [variant for variant in variants if variant.name in requested]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_records = load_detail_records(output_dir / "skillgrep_component_ablation_details.jsonl")
    summary_rows = summarize_detail_records(detail_records, top_k=args.top_k)
    searcher = build_variant_searcher(
        skills,
        args,
        variant=variants[0],
        dense_view_names=dense_view_names,
    )

    for dataset in args.datasets:
        instances_path = Path(args.instances_dir) / f"{dataset}.json"
        instances = [
            instance
            for instance in load_sra_instances(instances_path)
            if instance.get("skill_annotations")
        ]
        if args.limit is not None:
            instances = instances[: args.limit]
        for variant in variants:
            existing = [
                record
                for record in detail_records
                if record.get("dataset") == dataset and record.get("variant") == variant.name
            ]
            if len(existing) >= len(instances):
                print(
                    json.dumps(
                        {
                            "event": "ablation_skip_completed",
                            "dataset": dataset,
                            "variant": variant.name,
                            "records": len(existing),
                        }
                    ),
                    flush=True,
                )
                continue
            print(
                json.dumps(
                    {
                        "event": "ablation_variant_start",
                        "dataset": dataset,
                        "variant": variant.name,
                        "instances": len(instances),
                    }
                ),
                flush=True,
            )
            apply_variant_to_searcher(searcher, variant)
            records = run_dataset_variant(
                dataset=dataset,
                instances=instances,
                searcher=searcher,
                variant=variant.name,
                top_k=args.top_k,
            )
            detail_records.extend(records)
            summary_rows = summarize_detail_records(detail_records, top_k=args.top_k)
            write_outputs(
                output_dir=output_dir,
                detail_records=detail_records,
                summary_rows=summary_rows,
                args=args,
                variants=variants,
            )
            print(
                json.dumps(
                    {
                        "event": "ablation_variant_done",
                        "dataset": dataset,
                        "variant": variant.name,
                        "records": len(records),
                    }
                ),
                flush=True,
            )
    del searcher
    gc.collect()

    summary_rows = summarize_detail_records(detail_records, top_k=args.top_k)
    paths = write_outputs(
        output_dir=output_dir,
        detail_records=detail_records,
        summary_rows=summary_rows,
        args=args,
        variants=variants,
    )
    return {"paths": paths, "summary_rows": summary_rows}


def load_detail_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def summarize_detail_records(records: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((str(record["dataset"]), str(record["variant"])), []).append(record)
    rows = []
    for (dataset, variant), group_records in sorted(grouped.items()):
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                **compute_ablation_metrics(group_records, top_k=top_k),
                **compute_sra_retrieval_metrics(to_sra_retrieval_records(group_records), top_k=top_k),
            }
        )
    return [*rows, *macro_average_rows(rows)]


def build_ablation_variants(args: argparse.Namespace) -> list[AblationVariant]:
    base = build_search_weights(args)
    lexical_only = replace_weights(base, sparse_view=0.0, dense=0.0, rrf=0.0, capability=0.0, usage=0.0, input_type=0.0, output_type=0.0, penalty=0.0)
    sparse_only = replace_weights(base, lexical=0.0, dense=0.0, rrf=0.0, capability=0.0, usage=0.0, input_type=0.0, output_type=0.0, penalty=0.0)
    dense_only = replace_weights(base, lexical=0.0, sparse_view=0.0, rrf=0.0, capability=0.0, usage=0.0, input_type=0.0, output_type=0.0, penalty=0.0)
    return [
        AblationVariant("full_hybrid", True, True, True, base),
        AblationVariant("bm25_only", True, False, False, lexical_only),
        AblationVariant("dense_only", False, False, True, dense_only),
        AblationVariant("sparse_view_only", False, True, False, sparse_only),
        AblationVariant("no_dense", True, True, False, replace_weights(base, dense=0.0)),
        AblationVariant("no_sparse_view", True, False, True, replace_weights(base, sparse_view=0.0)),
        AblationVariant("no_rrf_score", True, True, True, replace_weights(base, rrf=0.0)),
        AblationVariant(
            "no_capability_type_signals",
            True,
            True,
            True,
            replace_weights(base, capability=0.0, usage=0.0, input_type=0.0, output_type=0.0),
        ),
        AblationVariant("no_penalty", True, True, True, replace_weights(base, penalty=0.0)),
    ]


def replace_weights(weights: SearchWeights, **overrides: float) -> SearchWeights:
    values = {
        "lexical": weights.lexical,
        "sparse_view": weights.sparse_view,
        "dense": weights.dense,
        "rrf": weights.rrf,
        "capability": weights.capability,
        "usage": weights.usage,
        "input_type": weights.input_type,
        "output_type": weights.output_type,
        "penalty": weights.penalty,
    }
    values.update(overrides)
    return SearchWeights(**values)


def build_variant_searcher(
    skills: list[Any],
    args: argparse.Namespace,
    variant: AblationVariant,
    *,
    dense_view_names: set[str] | None,
) -> SkillSearcher:
    embed_args = SimpleNamespace(**vars(args))
    if not variant.dense_enabled and getattr(embed_args, "embedding_backend", None) is None:
        embed_args.embedding_backend = "none"
    embedder = build_configured_embedder(embed_args)
    return SkillSearcher(
        skills,
        embedder=embedder,
        dense_enabled=variant.dense_enabled,
        bm25_enabled=variant.bm25_enabled,
        sparse_view_enabled=variant.sparse_view_enabled,
        dense_view_names=dense_view_names,
        weights=variant.weights,
        minimum_score_threshold=build_minimum_score_threshold(args),
        dense_cache_dir=getattr(args, "embedding_cache_dir", None)
        or load_app_config_if_exists(args.config).embedding.cache_dir,
    )


def apply_variant_to_searcher(searcher: SkillSearcher, variant: AblationVariant) -> None:
    searcher.bm25_enabled = variant.bm25_enabled
    searcher.sparse_view_enabled = variant.sparse_view_enabled
    searcher.dense_enabled = variant.dense_enabled
    searcher.weights = variant.weights


def run_dataset_variant(
    *,
    dataset: str,
    instances: list[dict[str, Any]],
    searcher: SkillSearcher,
    variant: str,
    top_k: int,
) -> list[dict[str, Any]]:
    records = []
    for instance in instances:
        gold_ids = [str(item) for item in instance.get("skill_annotations", []) if str(item).strip()]
        query = build_sra_query(instance, sra_repo=ROOT / SRA_SUBMODULE_DIR)
        response = searcher.search(SkillSearchRequest(query=query), top_k=top_k)
        retrieved = [
            {
                "rank": rank,
                "skill_id": card.id,
                "score": float(card.score),
                "score_breakdown": card.score_breakdown.dict(),
                "is_gold": card.id in set(gold_ids),
            }
            for rank, card in enumerate(response.results, start=1)
        ]
        best = best_gold_hit(gold_ids, retrieved)
        records.append(
            {
                "dataset": dataset,
                "instance_id": str(instance["instance_id"]),
                "variant": variant,
                "gold_skill_ids": gold_ids,
                "retrieved": retrieved,
                "gold_hit_at_5": best is not None,
                "best_gold_rank": best["rank"] if best else None,
                "best_gold_score": best["score"] if best else None,
            }
        )
    return records


def best_gold_hit(gold_ids: list[str], retrieved: list[dict[str, Any]]) -> dict[str, Any] | None:
    gold = set(gold_ids)
    hits = [item for item in retrieved if item["skill_id"] in gold]
    if not hits:
        return None
    return min(hits, key=lambda item: int(item["rank"]))


def compute_ablation_metrics(records: list[dict[str, Any]], *, top_k: int) -> dict[str, float | int | None]:
    total = len(records)
    hits = [record for record in records if record.get("gold_hit_at_5")]
    reciprocal_ranks = [
        1.0 / float(record["best_gold_rank"])
        if record.get("best_gold_rank") is not None and int(record["best_gold_rank"]) <= top_k
        else 0.0
        for record in records
    ]
    ranks = [float(record["best_gold_rank"]) for record in hits if record.get("best_gold_rank") is not None]
    scores = [float(record["best_gold_score"]) for record in hits if record.get("best_gold_score") is not None]
    return {
        "instances": total,
        "GoldHit@5": round(len(hits) / total, 6) if total else 0.0,
        "GoldMRR@5": round(sum(reciprocal_ranks) / total, 6) if total else 0.0,
        "GoldMeanRank@5": round(sum(ranks) / len(ranks), 6) if ranks else None,
        "GoldMeanScore@5": round(sum(scores) / len(scores), 6) if scores else None,
    }


def to_sra_retrieval_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "instance_id": record["instance_id"],
            "gold_skill_ids": record["gold_skill_ids"],
            "retrieved": [
                {"skill_id": item["skill_id"], "score": item["score"]}
                for item in record["retrieved"]
            ],
        }
        for record in records
    ]


def macro_average_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants = sorted({row["variant"] for row in rows})
    averaged = []
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant and row["dataset"] != "macro"]
        if not variant_rows:
            continue
        metric_names = [
            name
            for name in variant_rows[0]
            if name not in {"dataset", "variant"} and isinstance(variant_rows[0][name], (int, float))
        ]
        average = {"dataset": "macro", "variant": variant}
        for name in metric_names:
            values = [float(row[name]) for row in variant_rows if isinstance(row.get(name), (int, float))]
            average[name] = round(sum(values) / len(values), 6) if values else None
        averaged.append(average)
    return averaged


def write_outputs(
    *,
    output_dir: Path,
    detail_records: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    variants: list[AblationVariant],
) -> dict[str, str]:
    details_path = output_dir / "skillgrep_component_ablation_details.jsonl"
    summary_json_path = output_dir / "skillgrep_component_ablation_summary.json"
    summary_csv_path = output_dir / "skillgrep_component_ablation_summary.csv"
    summary_md_path = output_dir / "skillgrep_component_ablation_summary.md"
    summary_tex_path = output_dir / "skillgrep_component_ablation_summary.tex"

    with details_path.open("w", encoding="utf-8") as handle:
        for record in detail_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    payload = {
        "metadata": {
            "datasets": args.datasets,
            "top_k": args.top_k,
            "config": args.config,
            "corpus": args.corpus,
            "sra_skill_dir": args.sra_skill_dir,
            "limit": args.limit,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "variants": [variant.name for variant in variants],
        },
        "summary_rows": summary_rows,
        "details_path": str(details_path),
    }
    summary_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(summary_csv_path, summary_rows)
    summary_md_path.write_text(format_markdown_table(summary_rows), encoding="utf-8")
    summary_tex_path.write_text(format_latex_table(summary_rows), encoding="utf-8")
    return {
        "details_jsonl": str(details_path),
        "summary_json": str(summary_json_path),
        "summary_csv": str(summary_csv_path),
        "summary_md": str(summary_md_path),
        "summary_tex": str(summary_tex_path),
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "dataset",
        "variant",
        "instances",
        "GoldHit@5",
        "GoldMRR@5",
        "GoldMeanRank@5",
        "GoldMeanScore@5",
        "Recall@1",
        "Recall@5",
        "nDCG@1",
        "nDCG@5",
    ]
    fieldnames = [name for name in preferred if name in fieldnames] + [
        name for name in fieldnames if name not in preferred
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = ["dataset", "variant", "instances", "GoldHit@5", "GoldMRR@5", "GoldMeanRank@5", "GoldMeanScore@5", "Recall@1", "Recall@5", "nDCG@1", "nDCG@5"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(column)) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def format_latex_table(rows: list[dict[str, Any]]) -> str:
    columns = ["dataset", "variant", "GoldHit@5", "GoldMRR@5", "GoldMeanScore@5", "Recall@5", "nDCG@5"]
    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Dataset & Variant & GoldHit@5 & GoldMRR@5 & GoldScore@5 & Recall@5 & nDCG@5 \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = [latex_escape(format_cell(row.get(column))) for column in columns]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def latex_escape(value: str) -> str:
    return value.replace("_", "\\_").replace("@", "\\@")


if __name__ == "__main__":
    raise SystemExit(main())
