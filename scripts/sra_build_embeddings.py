"""Build SRA dense embedding caches explicitly before agent/retrieval runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_app_config  # noqa: E402
from core.embeddings import build_embedder  # noqa: E402
from core.search import dense_cache_path, read_dense_cache, write_dense_cache  # noqa: E402
from loader import find_skill_files, load_skill, load_skills  # noqa: E402


DEFAULT_SRA_DENSE_VIEW_NAMES = ["description", "capability", "usage", "examples", "schema"]
DEFAULT_CHUNK_SIZE = 512


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--skill-dir", action="append", help="SkillSpec dir. Defaults to [sra].skill_dirs.")
    parser.add_argument("--cache-dir", help="Dense cache dir. Defaults to [embedding].cache_dir.")
    parser.add_argument("--checkpoint", default="data/eval/sra/embedding_cache/build.checkpoint.jsonl")
    parser.add_argument("--log", default="data/eval/sra/embedding_cache/build.log.jsonl")
    parser.add_argument("--backend", choices=["hf-transformers", "fake"], help="Defaults to [embedding].backend.")
    parser.add_argument("--model", help="Defaults to [embedding].model.")
    parser.add_argument("--batch-size", type=int, help="Defaults to [embedding].batch_size.")
    parser.add_argument("--max-length", type=int, help="Defaults to [embedding].max_length.")
    parser.add_argument("--device", help="Defaults to [embedding].device.")
    parser.add_argument("--view-name", action="append", help="Dense view to build. Repeatable.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--force", action="store_true", help="Rebuild views even when cache/checkpoint exists.")
    parser.add_argument("--dry-run", action="store_true", help="Plan cache paths without embedding or writing.")
    return parser.parse_args(argv)


def main() -> int:
    return main_with_args()


def main_with_args(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    config = load_app_config(args.config)
    skill_dirs = args.skill_dir or config.sra.skill_dirs
    if not skill_dirs:
        raise ValueError("No skill dirs provided. Set [sra].skill_dirs or pass --skill-dir.")
    cache_dir = Path(args.cache_dir or config.embedding.cache_dir or "data/eval/sra/embedding_cache")
    checkpoint_path = Path(args.checkpoint)
    log_path = Path(args.log)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    view_names = args.view_name or DEFAULT_SRA_DENSE_VIEW_NAMES
    backend = args.backend or config.embedding.backend
    if backend == "none":
        raise ValueError("Embedding backend cannot be 'none' when building dense caches.")
    configured_model_name = args.model or config.embedding.model
    cache_model_name = "fake-semantic" if backend == "fake" else configured_model_name

    log_event(
        log_path,
        "start",
        skill_dirs=skill_dirs,
        requested_views=view_names,
        backend=backend,
        model=configured_model_name,
        cache_dir=str(cache_dir),
        chunk_size=args.chunk_size,
    )
    plan_started = time.perf_counter()
    skill_files = list_skill_files(skill_dirs)
    log_event(log_path, "skill_files_listed", skill_file_count=len(skill_files))
    plans = plan_dense_views_streaming(
        skill_files,
        cache_dir=cache_dir,
        model_name=cache_model_name,
        view_names=view_names,
        chunk_size=args.chunk_size,
        log_path=log_path,
    )
    log_event(log_path, "views_planned", view_count=len(plans), elapsed_seconds=round(time.perf_counter() - plan_started, 3))
    completed = load_completed_checkpoint(checkpoint_path)

    if args.dry_run:
        for plan in plans:
            print(json.dumps({"view_name": plan["view_name"], "cache_path": str(plan["cache_path"])}))
        log_event(log_path, "dry_run", elapsed_seconds=round(time.perf_counter() - started, 3))
        return 0

    embedder = build_embedder(
        backend,
        model_name=configured_model_name,
        batch_size=args.batch_size or config.embedding.batch_size,
        max_length=args.max_length or config.embedding.max_length,
        device=args.device if args.device is not None else config.embedding.device,
    )

    built = 0
    reused = 0
    for plan in plans:
        view_started = time.perf_counter()
        cache_path = plan["cache_path"]
        key = checkpoint_key(plan)
        if args.force:
            partial_cache_path(cache_path).unlink(missing_ok=True)
        if not args.force and key in completed and valid_cache(cache_path, plan["text_count"]):
            reused += 1
            log_event(log_path, "reuse", view_name=plan["view_name"], cache_path=str(cache_path))
            continue
        if not args.force and valid_cache(cache_path, plan["text_count"]):
            write_checkpoint(checkpoint_path, plan, model_name=embedder.model_name, status="cached")
            reused += 1
            log_event(log_path, "cache_exists", view_name=plan["view_name"], cache_path=str(cache_path))
            continue

        log_event(
            log_path,
            "embed_start",
            view_name=plan["view_name"],
            text_count=plan["text_count"],
            cache_path=str(cache_path),
        )
        embed_view_streaming(
            plan,
            skill_files,
            embedder,
            chunk_size=args.chunk_size,
            log_path=log_path,
        )
        write_checkpoint(checkpoint_path, plan, model_name=embedder.model_name, status="built")
        built += 1
        log_event(
            log_path,
            "embed_done",
            view_name=plan["view_name"],
            cache_path=str(cache_path),
            elapsed_seconds=round(time.perf_counter() - view_started, 3),
        )

    summary = {
        "ok": True,
        "skill_count": len(skill_files),
        "view_count": len(plans),
        "built": built,
        "reused": reused,
        "cache_dir": str(cache_dir),
        "checkpoint": str(checkpoint_path),
        "log": str(log_path),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    log_event(log_path, "finish", **summary)
    print(json.dumps(summary, indent=2))
    return 0


def list_skill_files(skill_dirs: list[str]) -> list[Path]:
    skill_files = []
    for skill_dir in skill_dirs:
        root = Path(skill_dir)
        skill_files.extend(find_skill_files(root))
    return skill_files


def load_skill_dirs(skill_dirs: list[str]) -> list[Any]:
    skills = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    for skill_dir in skill_dirs:
        for skill in load_skills(Path(skill_dir)):
            if skill.id in seen:
                duplicates.add(skill.id)
            seen.add(skill.id)
            skills.append(skill)
    if duplicates:
        raise ValueError(f"Duplicate skill id(s) across skill dirs: {', '.join(sorted(duplicates))}")
    return skills


def plan_dense_views_streaming(
    skill_files: list[Path],
    *,
    cache_dir: Path,
    model_name: str,
    view_names: list[str],
    chunk_size: int,
    log_path: Path,
) -> list[dict[str, Any]]:
    validate_chunk_size(chunk_size)
    requested = sorted(set(view_names))
    states = {view_name: DensePlanState(model_name, view_name) for view_name in requested}
    total = 0
    seen: set[str] = set()
    duplicates: set[str] = set()
    for chunk_index, chunk_paths in enumerate(chunked(skill_files, chunk_size), start=1):
        chunk_skills = [load_skill(path) for path in chunk_paths]
        for skill in chunk_skills:
            if skill.id in seen:
                duplicates.add(skill.id)
            seen.add(skill.id)
            total += 1
            for state in states.values():
                text = _metadata_view_text(skill, state.view_name)
                state.add(skill.id, text)
        log_event(log_path, "plan_chunk", chunk_index=chunk_index, planned_skills=total)
    if duplicates:
        raise ValueError(f"Duplicate skill id(s) across skill dirs: {', '.join(sorted(duplicates))}")

    plans = []
    for view_name, state in states.items():
        if not state.has_text:
            continue
        cache_path = state.cache_path(cache_dir)
        plans.append(
            {
                "view_name": view_name,
                "cache_path": cache_path,
                "text_count": state.text_count,
                "skill_count": state.skill_count,
            }
        )
    return plans


def plan_dense_views(skills: list[Any], *, cache_dir: Path, model_name: str, view_names: list[str]) -> list[dict[str, Any]]:
    requested = sorted(set(view_names))
    plans = []
    skill_ids = [skill.id for skill in skills]
    for view_name in requested:
        texts = [_metadata_view_text(skill, view_name) for skill in skills]
        if not any(text.strip() for text in texts):
            continue
        path = dense_cache_path(cache_dir, model_name, view_name, skill_ids, texts)
        if path is None:
            raise ValueError("cache_dir is required")
        plans.append({"view_name": view_name, "skill_ids": skill_ids, "texts": texts, "cache_path": path})
    return plans


class DensePlanState:
    def __init__(self, model_name: str, view_name: str) -> None:
        import hashlib
        import re

        self.view_name = view_name
        self.skill_count = 0
        self.text_count = 0
        self.has_text = False
        self._safe_view_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", view_name)
        self._signature = hashlib.sha256()
        self._signature.update(model_name.encode("utf-8"))
        self._signature.update(view_name.encode("utf-8"))
        self._hashlib = hashlib

    def add(self, skill_id: str, text: str) -> None:
        self.skill_count += 1
        self.text_count += 1
        if text.strip():
            self.has_text = True
        self._signature.update(skill_id.encode("utf-8"))
        self._signature.update(self._hashlib.sha256(text.encode("utf-8")).digest())

    def cache_path(self, cache_dir: Path) -> Path:
        return cache_dir / f"{self._safe_view_name}-{self._signature.hexdigest()[:16]}.jsonl"


def embed_view_streaming(
    plan: dict[str, Any],
    skill_files: list[Path],
    embedder: Any,
    *,
    chunk_size: int,
    log_path: Path,
) -> None:
    validate_chunk_size(chunk_size)
    cache_path = plan["cache_path"]
    partial_path = partial_cache_path(cache_path)
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = count_cache_rows(partial_path)
    if existing_rows > plan["text_count"]:
        partial_path.unlink()
        existing_rows = 0
    written_rows = existing_rows
    if existing_rows:
        log_event(
            log_path,
            "partial_resume",
            view_name=plan["view_name"],
            cache_path=str(cache_path),
            existing_rows=existing_rows,
        )
    else:
        partial_path.write_text("", encoding="utf-8")

    seen_rows = 0
    for chunk_index, chunk_paths in enumerate(chunked(skill_files, chunk_size), start=1):
        chunk_skills = [load_skill(path) for path in chunk_paths]
        texts = [_metadata_view_text(skill, plan["view_name"]) for skill in chunk_skills]
        chunk_start = seen_rows
        chunk_end = seen_rows + len(texts)
        seen_rows = chunk_end
        if chunk_end <= existing_rows:
            continue
        if existing_rows > chunk_start:
            texts = texts[existing_rows - chunk_start :]
        log_event(
            log_path,
            "embed_chunk_start",
            view_name=plan["view_name"],
            chunk_index=chunk_index,
            text_count=len(texts),
            written_rows=written_rows,
        )
        embeddings = embedder.embed_texts(texts)
        append_embeddings(partial_path, embeddings, start_position=written_rows)
        written_rows += len(embeddings)
        log_event(
            log_path,
            "embed_chunk_done",
            view_name=plan["view_name"],
            chunk_index=chunk_index,
            written_rows=written_rows,
        )
    if written_rows != plan["text_count"]:
        raise ValueError(
            f"Embedded row count mismatch for {plan['view_name']}: "
            f"expected {plan['text_count']}, wrote {written_rows}"
        )
    partial_path.replace(cache_path)


def _metadata_view_text(skill: Any, view_name: str) -> str:
    if view_name == "description":
        return "\n".join(
            [
                skill.name,
                skill.description.short,
                skill.description.long or "",
                skill.category.primary,
                " ".join(skill.category.secondary),
                " ".join(skill.tags),
            ]
        )
    if view_name == "capability":
        return "\n".join(f"{capability.id}: {capability.description}" for capability in skill.capabilities)
    if view_name == "usage":
        return "\n".join(skill.when_to_use)
    if view_name == "examples":
        return "\n".join(example.user_query for example in skill.examples.positive)
    if view_name == "schema":
        if not skill.input_schema and not skill.output_schema:
            return ""
        return "\n".join(
            [
                " ".join(skill.input_types),
                " ".join(skill.output_types),
                str(skill.input_schema or ""),
                str(skill.output_schema or ""),
            ]
        )
    raise ValueError(
        f"Unsupported explicit embedding view: {view_name}. "
        "This script intentionally builds metadata views only."
    )


def load_completed_checkpoint(path: Path) -> set[str]:
    completed = set()
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") in {"built", "cached"} and record.get("cache_path"):
            completed.add(str(record.get("cache_path")))
    return completed


def checkpoint_key(plan: dict[str, Any]) -> str:
    return str(plan["cache_path"])


def valid_cache(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    cached = read_dense_cache(path)
    return cached is not None and len(cached) == expected_rows


def write_cache_atomic(path: Path, embeddings: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    write_dense_cache(tmp_path, embeddings)
    tmp_path.replace(path)


def append_embeddings(path: Path, embeddings: list[list[float]], *, start_position: int) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for offset, vector in enumerate(embeddings):
            record = {"position": start_position + offset, "vector": vector}
            handle.write(json.dumps(record) + "\n")


def partial_cache_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.name}.partial")


def count_cache_rows(path: Path) -> int:
    if not path.exists():
        return 0
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for rows, _line in enumerate(handle, start=1):
            pass
    return rows


def chunked(items: list[Path], chunk_size: int):
    validate_chunk_size(chunk_size)
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def validate_chunk_size(chunk_size: int) -> None:
    if chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")


def write_checkpoint(path: Path, plan: dict[str, Any], *, model_name: str, status: str) -> None:
    record = {
        "status": status,
        "view_name": plan["view_name"],
        "model": model_name,
        "cache_path": str(plan["cache_path"]),
        "text_count": plan["text_count"],
        "timestamp": time.time(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_event(path: Path, event: str, **payload: Any) -> None:
    record = {"event": event, "timestamp": time.time(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
