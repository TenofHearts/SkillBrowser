"""General-purpose SRA agent loop backed by the local hybrid skill searcher."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from core.search import SkillSearcher
from llm import LLMClient
from schema import SkillSearchRequest

from .sra import SRA_SUBMODULE_DIR, load_sra_instances


COMPACT_SEARCH_SKILL_DESCRIPTION = (
    "name: skill_search\n"
    "description: Use this tool when a task may require finding an appropriate skill. "
    "It searches the skill library and returns candidate metadata so you can choose "
    "which skill to load and use."
)

FULL_SEARCH_SKILL_DESCRIPTION = (
    f"{COMPACT_SEARCH_SKILL_DESCRIPTION}\n\n"
    "This is a multi-stage loading skill. First call operation=search with a "
    "retrieval_intent object describing the capability you need. The tool returns "
    "metadata only: skill_id, name, description, score, matched signals, and loading "
    "hints. If a candidate is useful, call operation=load_skill with that skill_id "
    "and optional section. The tool then returns the selected skill content. You may "
    "repeat search or load_skill as needed, but no other tools are available.\n\n"
    "Search call JSON:\n"
    '{"tool":"skill_search","operation":"search","retrieval_intent":{"query":"...",'
    '"task_context":"...","required_capabilities":[],"desired_capabilities":[],'
    '"input_types":[],"output_types":[],"positive_signals":[],"negative_signals":[],'
    '"constraints":{}}}\n\n'
    "Load call JSON:\n"
    '{"tool":"skill_search","operation":"load_skill","skill_id":"...","section":"overview"}\n\n'
    "When you have enough information, answer the benchmark task directly instead "
    "of calling the tool."
)


@dataclass
class SRAAgentRecord:
    instance_id: str
    dataset: str
    method: str
    model: str
    raw_output: str
    transcript: str | None = None
    skill_ids_used: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "instance_id": self.instance_id,
            "dataset": self.dataset,
            "method": self.method,
            "model": self.model,
            "raw_output": self.raw_output,
        }
        if self.transcript is not None:
            record["transcript"] = self.transcript
        if self.skill_ids_used:
            record["skill_ids_used"] = self.skill_ids_used
        if self.meta:
            record["meta"] = self.meta
        if self.error is not None:
            record["error"] = self.error
        return record


class SRAGeneralPurposeAgent:
    """LLM task solver whose only callable tool is the skill_search skill."""

    def __init__(
        self,
        *,
        searcher: SkillSearcher,
        corpus: list[dict[str, Any]],
        llm: LLMClient,
        model_name: str,
        top_k: int = 5,
        max_rounds: int = 6,
        inject_full_search_skill: bool = False,
        sra_repo: str | Path = SRA_SUBMODULE_DIR,
        method: str = "skillbrowser_general_agent",
    ):
        self.searcher = searcher
        self.corpus_by_id = {
            str(skill.get("skill_id") or skill.get("id")): skill
            for skill in corpus
            if str(skill.get("skill_id") or skill.get("id") or "").strip()
        }
        self.llm = llm
        self.model_name = model_name
        self.top_k = top_k
        self.max_rounds = max_rounds
        self.inject_full_search_skill = inject_full_search_skill
        self.sra_repo = Path(sra_repo)
        self.method = method

    def run_instance(self, instance: dict[str, Any]) -> SRAAgentRecord:
        started = time.perf_counter()
        search_call_count = 0
        load_call_count = 0
        parse_errors: list[str] = []
        loaded_skill_ids: list[str] = []
        input_tokens = 0
        output_tokens = 0
        total_latency_ms = 0
        token_usage_sources: list[str] = []
        transcript_parts: list[str] = []

        try:
            base_system, user_prompt = build_sra_prompt(instance, sra_repo=self.sra_repo)
            messages = [
                {"role": "system", "content": self._system_prompt(base_system)},
                {"role": "user", "content": user_prompt},
            ]
            final_output = ""

            for _round_index in range(self.max_rounds):
                completion = self.llm.complete_with_usage(messages)
                input_tokens += completion.input_tokens
                output_tokens += completion.output_tokens
                total_latency_ms += completion.elapsed_ms
                token_usage_sources.append(completion.token_usage_source)
                raw = completion.content.strip()
                transcript_parts.append(f"ASSISTANT:\n{raw}")

                call, parse_error = parse_sra_agent_tool_call(raw)
                if parse_error:
                    parse_errors.append(parse_error)

                if call is None:
                    final_output = _extract_final_answer(raw)
                    break

                messages.append({"role": "assistant", "content": raw})
                observation: dict[str, Any]
                if call.get("tool") != "skill_search":
                    observation = {"error": "Only the skill_search tool is available."}
                    parse_errors.append("LLM attempted to call a tool other than skill_search")
                elif call.get("operation") == "search":
                    search_call_count += 1
                    observation = self._search_observation(call)
                elif call.get("operation") == "load_skill":
                    load_call_count += 1
                    observation = self._load_observation(call)
                    skill_id = observation.get("skill_id")
                    if isinstance(skill_id, str) and "content" in observation and skill_id not in loaded_skill_ids:
                        loaded_skill_ids.append(skill_id)
                else:
                    observation = {"error": "Unsupported skill_search operation. Use search or load_skill."}
                    parse_errors.append("Unsupported skill_search operation")

                observation_text = "skill_search result:\n" + json.dumps(
                    observation,
                    ensure_ascii=False,
                    indent=2,
                )
                transcript_parts.append(f"USER:\n{observation_text}")
                messages.append({"role": "user", "content": observation_text})
            else:
                final_output = raw if "raw" in locals() else ""

            gold_ids = _gold_skill_ids(instance)
            recalled_gold = bool(set(gold_ids) & set(loaded_skill_ids)) if gold_ids else False
            meta = {
                "search_call_count": search_call_count,
                "load_call_count": load_call_count,
                "loaded_skill_ids": loaded_skill_ids,
                "gold_skill_ids": gold_ids,
                "recalled_gold": recalled_gold,
                "parse_errors": parse_errors,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "total_latency_ms": total_latency_ms,
                "wall_time_ms": round((time.perf_counter() - started) * 1000),
                "token_usage_source": _merge_token_usage_sources(token_usage_sources),
                "inject_full_search_skill": self.inject_full_search_skill,
            }
            return SRAAgentRecord(
                instance_id=str(instance["instance_id"]),
                dataset=str(instance["dataset"]),
                method=self.method,
                model=_model_short_name(self.model_name),
                raw_output=final_output,
                transcript="\n\n".join(transcript_parts),
                skill_ids_used=loaded_skill_ids,
                meta=meta,
            )
        except Exception as exc:  # noqa: BLE001
            return SRAAgentRecord(
                instance_id=str(instance.get("instance_id", "")),
                dataset=str(instance.get("dataset", "")),
                method=self.method,
                model=_model_short_name(self.model_name),
                raw_output="",
                error=str(exc),
                meta={"wall_time_ms": round((time.perf_counter() - started) * 1000)},
            )

    def _system_prompt(self, base_system: str) -> str:
        search_skill = (
            FULL_SEARCH_SKILL_DESCRIPTION
            if self.inject_full_search_skill
            else COMPACT_SEARCH_SKILL_DESCRIPTION
        )
        general = (
            "You are a general-purpose agent solving benchmark tasks. "
            "Solve the user's task accurately. Your only callable tool or skill is "
            "skill_search. If a specialized method may help, call skill_search by "
            "returning exactly one JSON object. If you do not need a skill, answer "
            "the task directly. Do not claim access to any other tools.\n\n"
            f"Available skill metadata:\n{search_skill}"
        )
        return f"{base_system}\n\n{general}" if base_system else general

    def _search_observation(self, call: dict[str, Any]) -> dict[str, Any]:
        intent = call.get("retrieval_intent")
        if not isinstance(intent, dict):
            intent = {"query": str(call.get("query") or "")}
        if not intent.get("query"):
            intent["query"] = str(call.get("query") or call.get("task_context") or "")
        try:
            request = SkillSearchRequest.parse_obj(_normalize_search_intent(intent))
        except ValidationError as exc:
            return {"error": f"Invalid retrieval_intent: {exc}"}
        response = self.searcher.search(request, top_k=self.top_k)
        return {
            "operation": "search",
            "query": response.query,
            "abstained": response.abstained,
            "abstention_reason": response.abstention_reason,
            "candidates": [
                {
                    "skill_id": card.id,
                    "name": card.name,
                    "description": card.description,
                    "score": float(card.score),
                    "matched_signals": {
                        "capabilities": card.matched_capabilities,
                        "negative": card.negative_matches,
                    },
                    "available_sections": card.available_sections,
                    "loading_hint": {
                        "operation": "load_skill",
                        "skill_id": card.id,
                        "section": card.read_recommendation,
                    },
                }
                for card in response.results
            ],
        }

    def _load_observation(self, call: dict[str, Any]) -> dict[str, Any]:
        skill_id = str(call.get("skill_id") or "").strip()
        if not skill_id:
            return {"error": "load_skill requires skill_id"}
        skill = self.corpus_by_id.get(skill_id)
        if not skill:
            return {"error": f"Skill not found: {skill_id}", "skill_id": skill_id}
        content = str(skill.get("content") or skill.get("description") or "")
        return {
            "operation": "load_skill",
            "skill_id": skill_id,
            "name": skill.get("name") or skill_id,
            "section": call.get("section") or "overview",
            "content": content,
        }


def run_sra_agent_inference(
    *,
    searcher: SkillSearcher,
    corpus: list[dict[str, Any]],
    instances_path: str | Path,
    output_path: str | Path,
    llm: LLMClient,
    model_name: str,
    top_k: int = 5,
    max_rounds: int = 6,
    limit: int | None = None,
    inject_full_search_skill: bool = False,
    force: bool = False,
    sra_repo: str | Path = SRA_SUBMODULE_DIR,
) -> dict[str, Any]:
    instances = load_sra_instances(instances_path)
    if limit is not None:
        instances = instances[:limit]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if force and output.exists():
        output.unlink()

    done = _already_done(output)
    agent = SRAGeneralPurposeAgent(
        searcher=searcher,
        corpus=corpus,
        llm=llm,
        model_name=model_name,
        top_k=top_k,
        max_rounds=max_rounds,
        inject_full_search_skill=inject_full_search_skill,
        sra_repo=sra_repo,
    )
    written = 0
    with output.open("a", encoding="utf-8") as handle:
        for instance in instances:
            if instance["instance_id"] in done:
                continue
            record = agent.run_instance(instance)
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
            written += 1

    records = load_sra_agent_records(output)
    metrics = compute_sra_agent_skill_metrics(records, instances)
    return {
        "inference_output": str(output),
        "written": written,
        "total_records": len(records),
        "metrics": metrics,
    }


def load_sra_agent_records(path: str | Path) -> list[dict[str, Any]]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def compute_sra_agent_skill_metrics(
    inference_records: list[dict[str, Any]],
    instances: list[dict[str, Any]],
) -> dict[str, float]:
    by_instance = {str(instance["instance_id"]): instance for instance in instances}
    total = len(inference_records)
    searched = []
    recalled_when_searched = []
    recalled_loaded = []
    search_counts = []
    loaded_counts = []
    for record in inference_records:
        meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
        search_count = int(meta.get("search_call_count") or 0)
        loaded_ids = [str(item) for item in record.get("skill_ids_used", [])]
        if not loaded_ids:
            loaded_ids = [str(item) for item in meta.get("loaded_skill_ids", [])]
        instance = by_instance.get(str(record.get("instance_id")), {})
        gold = set(_gold_skill_ids(instance))
        recalled = bool(gold & set(loaded_ids)) if gold else False
        search_counts.append(search_count)
        loaded_counts.append(len(loaded_ids))
        if search_count > 0:
            searched.append(record)
            recalled_when_searched.append(1.0 if recalled else 0.0)
        if loaded_ids:
            recalled_loaded.append(1.0 if recalled else 0.0)
    return {
        "record_count": float(total),
        "search_call_rate": _round(len(searched) / total if total else 0.0),
        "skill_recall_when_searched": _round(_mean(recalled_when_searched)),
        "skill_recall_at_loaded": _round(_mean(recalled_loaded)),
        "avg_search_calls": _round(_mean([float(value) for value in search_counts])),
        "avg_loaded_skills": _round(_mean([float(value) for value in loaded_counts])),
    }


def write_sra_agent_summary(
    *,
    inference_path: str | Path,
    instances_path: str | Path,
    output_path: str | Path,
    eval_path: str | Path | None = None,
) -> dict[str, Any]:
    records = load_sra_agent_records(inference_path)
    instances = load_sra_instances(instances_path)
    payload: dict[str, Any] = {
        "metadata": {
            "inference": str(inference_path),
            "instances": str(instances_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "agent_metrics": compute_sra_agent_skill_metrics(records, instances),
    }
    if eval_path and Path(eval_path).exists():
        eval_payload = json.loads(Path(eval_path).read_text(encoding="utf-8"))
        payload["task_metrics"] = eval_payload.get("metrics", {})
        payload["dataset"] = eval_payload.get("dataset")
        payload["method"] = eval_payload.get("method")
        payload["model"] = eval_payload.get("model")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_sra_prompt(instance: dict[str, Any], *, sra_repo: str | Path = SRA_SUBMODULE_DIR) -> tuple[str, str]:
    repo_path = Path(sra_repo).resolve()
    src_path = repo_path / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    try:
        from sragents.prompts import build_prompt  # type: ignore[import-not-found]

        return build_prompt(instance)
    except Exception:
        return "", str(instance.get("question", ""))


def parse_sra_agent_tool_call(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    text = _extract_json_object_text(raw)
    if not text:
        return None, None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"Could not parse JSON tool/final output: {exc}"
    if not isinstance(parsed, dict):
        return None, "JSON output must be an object"
    if parsed.get("tool") == "skill_search" or parsed.get("operation") in {"search", "load_skill"}:
        parsed.setdefault("tool", "skill_search")
        return parsed, None
    return None, None


def _extract_json_object_text(raw: str) -> str:
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return ""


def _extract_final_answer(raw: str) -> str:
    text = raw.strip()
    json_text = _extract_json_object_text(text)
    if json_text:
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            return text
        if isinstance(parsed, dict) and "final_answer" in parsed:
            return str(parsed["final_answer"]).strip()
    return text


def _normalize_search_intent(intent: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(intent)
    for key in [
        "required_capabilities",
        "desired_capabilities",
        "input_types",
        "output_types",
        "positive_signals",
        "negative_signals",
    ]:
        value = normalized.get(key)
        if value is None:
            normalized[key] = []
        elif isinstance(value, str):
            normalized[key] = [value] if value.strip() else []
        elif not isinstance(value, list):
            normalized[key] = [str(value)]
    constraints = normalized.get("constraints")
    if constraints is None:
        normalized["constraints"] = {}
    elif isinstance(constraints, str):
        normalized["constraints"] = {"note": constraints} if constraints.strip() else {}
    elif not isinstance(constraints, dict):
        normalized["constraints"] = {"value": constraints}
    normalized.setdefault("query", "")
    return normalized


def _gold_skill_ids(instance: dict[str, Any]) -> list[str]:
    return [str(item) for item in instance.get("skill_annotations", []) if str(item).strip()]


def _already_done(output: Path) -> set[str]:
    if not output.exists():
        return set()
    done: set[str] = set()
    for line in output.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        instance_id = record.get("instance_id")
        if isinstance(instance_id, str):
            done.add(instance_id)
    return done


def _model_short_name(model: str) -> str:
    return Path(model).name


def _merge_token_usage_sources(sources: list[str]) -> str:
    unique = {source for source in sources if source}
    if not unique:
        return "none"
    if len(unique) == 1:
        return next(iter(unique))
    return "mixed"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _round(value: float) -> float:
    return round(value, 6)
