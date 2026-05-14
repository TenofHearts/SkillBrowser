"""SRA-Bench adapters for evaluating this project's hybrid skill retriever."""

from __future__ import annotations

import json
import math
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.search import SkillSearcher
from schema import SkillSearchRequest, SkillSpec


SRA_SUBMODULE_DIR = Path("benchmarks/SR-Agents")
SRA_BENCH_DIR = SRA_SUBMODULE_DIR / "data/bench"
SRA_CORPUS_PATH = SRA_BENCH_DIR / "corpus/corpus.json"
SRA_CORPUS_ZIP_PATH = SRA_BENCH_DIR / "corpus/corpus.json.zip"
SRA_INSTANCES_DIR = SRA_BENCH_DIR / "instances"
SRA_RESULTS_DIR = Path("data/eval/sra/results")
SRA_DATASETS = ["theoremqa", "logicbench", "toolqa", "champ", "medcalcbench", "bigcodebench"]


def ensure_sra_corpus(corpus_path: str | Path = SRA_CORPUS_PATH) -> Path:
    """Unzip the bundled SRA corpus when only corpus.json.zip is present."""

    path = Path(corpus_path)
    if path.exists():
        return path
    zip_path = path.with_suffix(".json.zip")
    if not zip_path.exists():
        raise ValueError(
            f"SRA corpus not found at {path}. Initialize the submodule and make sure {zip_path} exists."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extract(path.name, path.parent)
    return path


def load_sra_corpus(path: str | Path = SRA_CORPUS_PATH) -> list[dict[str, Any]]:
    corpus_path = ensure_sra_corpus(path)
    return json.loads(corpus_path.read_text(encoding="utf-8"))


def sra_skill_to_spec(skill: dict[str, Any]) -> SkillSpec:
    skill_id = str(skill.get("skill_id") or skill.get("id") or "").strip()
    if not skill_id:
        raise ValueError(f"SRA skill is missing skill_id: {skill}")
    name = str(skill.get("name") or skill_id).strip()
    description = _compact_text(skill.get("description", ""), 1000) or name
    content = _compact_text(skill.get("content", ""), 8000)
    long_description = "\n\n".join(part for part in [description, content] if part)
    dataset = _dataset_from_skill_id(skill_id)
    capability_text = "\n".join(part for part in [description, content] if part).strip() or name
    return SkillSpec.parse_obj(
        {
            "id": skill_id,
            "name": name,
            "version": "0.1.0",
            "status": "active",
            "skill_type": "instructional",
            "category": {"primary": dataset, "secondary": ["sra-bench"]},
            "description": {"short": description, "long": long_description or description},
            "capabilities": [{"id": _slugify(name) or "skill", "description": capability_text[:1200]}],
            "interaction": {"mode": "read_then_apply", "readable": True, "executable": False},
            "content": {"format": "markdown", "path": "skill.md", "sections": ["overview"]},
            "when_to_use": [description],
            "examples": {"positive": []},
            "execution": {"mode": "none"},
            "tags": ["sra-bench", dataset],
        }
    )


def sra_corpus_to_specs(corpus: list[dict[str, Any]]) -> list[SkillSpec]:
    specs = [sra_skill_to_spec(skill) for skill in corpus]
    if not specs:
        raise ValueError("SRA corpus is empty")
    return specs


def load_sra_instances(path: str | Path) -> list[dict[str, Any]]:
    instances_path = Path(path)
    if not instances_path.exists():
        raise ValueError(f"SRA instances not found: {instances_path}")
    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    if not isinstance(instances, list):
        raise ValueError(f"SRA instances file must contain a list: {instances_path}")
    return instances


def build_sra_query(instance: dict[str, Any], *, sra_repo: str | Path = SRA_SUBMODULE_DIR) -> str:
    """Build the same retrieval query text used by the upstream SR-Agents CLI."""

    repo_path = Path(sra_repo).resolve()
    src_path = repo_path / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    try:
        from sragents.prompts import build_prompt  # type: ignore[import-not-found]

        system, user = build_prompt(instance)
        parts = [user]
        if system:
            parts.append(system)
        if instance.get("dataset") == "toolqa":
            from sragents.toolqa.fewshots import TOOLQA_EXAMPLES  # type: ignore[import-not-found]

            parts.append(TOOLQA_EXAMPLES)
        return "\n".join(parts)
    except Exception:
        parts = [str(instance.get("question", ""))]
        if instance.get("dataset"):
            parts.append(str(instance["dataset"]))
        return "\n".join(part for part in parts if part)


def run_sra_retrieval(
    *,
    searcher: SkillSearcher,
    corpus_size: int,
    instances_path: str | Path,
    output_path: str | Path,
    top_k: int = 50,
    limit: int | None = None,
    retriever_name: str = "skill-search-hybrid",
    sra_repo: str | Path = SRA_SUBMODULE_DIR,
) -> dict[str, Any]:
    instances = load_sra_instances(instances_path)
    records = []
    for instance in instances:
        gold_ids = [str(item) for item in instance.get("skill_annotations", []) if str(item).strip()]
        if not gold_ids:
            continue
        if limit is not None and len(records) >= limit:
            break
        query = build_sra_query(instance, sra_repo=sra_repo)
        response = searcher.search(SkillSearchRequest(query=query), top_k=top_k)
        records.append(
            {
                "instance_id": instance["instance_id"],
                "gold_skill_ids": gold_ids,
                "retrieved": [
                    {"skill_id": card.id, "score": float(card.score)}
                    for card in response.results
                ],
            }
        )
    dataset = instances[0].get("dataset") if instances else None
    metrics = compute_sra_retrieval_metrics(records, top_k=top_k)
    payload = {
        "metadata": {
            "dataset": dataset,
            "retriever": retriever_name,
            "top_k": top_k,
            "corpus_size": corpus_size,
            "n_queries": len(records),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "extra": {"source": "SkillBrowser Hybrid SkillSearcher"},
        },
        "metrics": metrics,
        "results": records,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def compute_sra_retrieval_metrics(records: list[dict[str, Any]], *, top_k: int) -> dict[str, float]:
    ks = [k for k in (1, 5, 10, 50) if k <= top_k]
    metrics: dict[str, float] = {}
    for k in ks:
        recalls = []
        ndcgs = []
        for record in records:
            gold = set(record["gold_skill_ids"])
            ranked = [item["skill_id"] for item in record["retrieved"]]
            if not gold:
                continue
            hits = sum(1 for skill_id in ranked[:k] if skill_id in gold)
            recalls.append(hits / len(gold))
            dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank, skill_id in enumerate(ranked[:k], start=1)
                if skill_id in gold
            )
            ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
            ndcgs.append(dcg / ideal if ideal else 0.0)
        metrics[f"Recall@{k}"] = _mean(recalls)
        metrics[f"nDCG@{k}"] = _mean(ndcgs)
    return metrics


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _dataset_from_skill_id(skill_id: str) -> str:
    for dataset in SRA_DATASETS:
        if skill_id.startswith(f"{dataset}_"):
            return dataset
    if skill_id.startswith("web_"):
        return "web"
    return "sra"


def _compact_text(value: Any, max_chars: int) -> str:
    return " ".join(str(value or "").split())[:max_chars]


def _slugify(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")
