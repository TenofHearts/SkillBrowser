"""General-purpose SRA agent loop backed by the local SkillBrowser searcher."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

try:
    from json_repair import repair_json
except ImportError:  # pragma: no cover - exercised in unsynced local envs
    repair_json = None  # type: ignore[assignment]

from core.search import SkillSearcher
from llm import LLMClient
from schema import SkillSearchRequest

from .sra import SRA_SUBMODULE_DIR, load_sra_instances

SEARCH_SKILL_DESCRIPTION = (
    "name: skill_search\n"
    "description: This is the primary way to discover task-relevant skills. "
    "Prefer using it near the start of benchmark tasks to retrieve candidate "
    "skills before solving. It returns candidate metadata so you can decide "
    "which skill, if any, to load and use."
)

SEARCH_DECISION_SYSTEM_PROMPT = (
    "Decide whether to search the skill library before solving the benchmark "
    "task. Do not solve the task in this phase.\n\n"
    "Prefer searching whenever the task has a reusable pattern, named or "
    "unnamed formula, calculation rule, theorem, counting method, recurrence, "
    "domain convention, or standard procedure. A task does not need to "
    "explicitly name the method for search to be useful.\n\n"
    "Do not skip just because the solution looks familiar, short, or derivable. "
    "Skip only when the task is purely direct reading or trivial arithmetic and "
    "there is no plausible reusable skill that could check or improve the "
    "solution.\n\n"
    "If search is useful, first reason briefly, then output exactly one "
    "<tool>...</tool> block:\n"
    "<tool>\n"
    '{"operation":"search","retrieval_intent":{"query":"...",'
    '"task_context":"...","required_capabilities":[],"positive_signals":[]}}\n'
    "</tool>\n\n"
    "Keep query short and ability-focused. Use task_context for a brief problem "
    "description. Use required_capabilities only for explicit hard requirements "
    "and positive_signals only for named formulas, methods, domains, or task "
    "clues useful for retrieval. If search is not useful, do not output a tool "
    "block."
)

DECISION_SRA_SYSTEM_PROMPTS = {
    "theoremqa": (
        "Dataset context: TheoremQA tasks often involve science, mathematics, "
        "theorems, formulas, definitions, or calculation methods."
    ),
    "logicbench": "",
    "toolqa": (
        "Dataset context: ToolQA tasks may require external capabilities such as "
        "calculation, agenda or paper retrieval, database loading/filtering, graph "
        "lookup, SQL, or Python execution."
    ),
    "champ": (
        "Dataset context: CHAMP tasks are mathematics problems that often involve "
        "combinatorics, algebra, number theory, geometry, recurrence relations, "
        "or proof-style techniques."
    ),
    "medcalcbench": (
        "Dataset context: MedCalcBench tasks usually ask for a named medical "
        "calculator, clinical score, formula, risk rule, or unit-sensitive "
        "calculation based on a patient note."
    ),
    "bigcodebench": (
        "Dataset context: BigCodeBench tasks involve programming problems, APIs, "
        "code behavior, algorithms, or implementation constraints."
    ),
}


def build_search_decision_system_prompt(base_system: str) -> str:
    if base_system.strip():
        return (
            f"{SEARCH_DECISION_SYSTEM_PROMPT}\n\n"
            "SRA-Bench dataset context for retrieval decision:\n"
            f"{base_system.strip()}"
        )
    return SEARCH_DECISION_SYSTEM_PROMPT


def decision_sra_system_prompt(instance: dict[str, Any]) -> str:
    return DECISION_SRA_SYSTEM_PROMPTS.get(str(instance.get("dataset", "")), "")


class SRASolveEngine(Protocol):
    def run(
        self,
        instance: dict[str, Any],
        skills: list[dict[str, Any]],
        client: Any,
        model: str,
        **kwargs: Any,
    ) -> Any: ...


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
            base_system, user_prompt = build_sra_prompt(
                instance, sra_repo=self.sra_repo
            )
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
                    parse_errors.append(
                        "LLM attempted to call a tool other than skill_search"
                    )
                elif call.get("operation") == "search":
                    search_call_count += 1
                    observation = self._search_observation(call)
                elif call.get("operation") == "load_skill":
                    load_call_count += 1
                    observation = self._load_observation(call)
                    skill_id = observation.get("skill_id")
                    if (
                        isinstance(skill_id, str)
                        and "content" in observation
                        and skill_id not in loaded_skill_ids
                    ):
                        loaded_skill_ids.append(skill_id)
                else:
                    observation = {
                        "error": "Unsupported skill_search operation. Use search or load_skill."
                    }
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
            recalled_gold = (
                bool(set(gold_ids) & set(loaded_skill_ids)) if gold_ids else False
            )
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
        general = (
            "You are a general-purpose agent solving benchmark tasks. "
            "Solve the user's task accurately. Your only callable tool or skill is "
            "skill_search. Treat skill_search as your default first move for "
            "benchmark tasks: in almost all cases, search before solving so you can "
            "check whether the skill library contains a relevant method, formula, "
            "procedure, or tool-use guide. Return exactly one JSON object with "
            "operation=search when you search. After you see the search results, "
            "actively look for a candidate whose name, description, capabilities, "
            "or loading hint matches the task. If a top candidate is plausibly "
            "relevant, prefer calling operation=load_skill for that skill_id before "
            "solving, then answer using the loaded skill. Directly answering without "
            "searching should be rare and reserved for clearly trivial tasks where "
            "a retrieved skill would not add value. If you searched and the results "
            "are not relevant, you may answer from your own reasoning without "
            "loading a skill. Do not claim access to any other tools.\n\n"
            f"Available skill metadata:\n{SEARCH_SKILL_DESCRIPTION}"
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


class SRASearchDecisionAgent:
    """Route through SkillBrowser search, then solve with an upstream SRA engine."""

    def __init__(
        self,
        *,
        searcher: SkillSearcher,
        corpus: list[dict[str, Any]],
        decision_llm: LLMClient,
        solve_engine: SRASolveEngine,
        solve_client: Any,
        model_name: str,
        top_k: int = 5,
        sra_repo: str | Path = SRA_SUBMODULE_DIR,
        method: str = "skillbrowser_agent_top5_direct",
        solve_engine_name: str = "direct",
    ):
        self.searcher = searcher
        self.corpus_by_id = {
            str(skill.get("skill_id") or skill.get("id")): skill
            for skill in corpus
            if str(skill.get("skill_id") or skill.get("id") or "").strip()
        }
        self.decision_llm = decision_llm
        self.solve_engine = solve_engine
        self.solve_client = solve_client
        self.model_name = model_name
        self.top_k = top_k
        self.sra_repo = Path(sra_repo)
        self.method = method
        self.solve_engine_name = solve_engine_name

    def run_instance(self, instance: dict[str, Any]) -> SRAAgentRecord:
        started = time.perf_counter()
        parse_errors: list[str] = []
        search_call_count = 0
        retrieved_skill_ids: list[str] = []
        decision_raw = ""

        try:
            _base_system, user_prompt = build_sra_prompt(
                instance, sra_repo=self.sra_repo
            )
            decision_messages = [
                {
                    "role": "system",
                    "content": build_search_decision_system_prompt(
                        decision_sra_system_prompt(instance)
                    ),
                },
                {"role": "user", "content": user_prompt},
            ]
            decision = self.decision_llm.complete_with_usage(decision_messages)
            decision_raw = decision.content.strip()
            call, parse_error = parse_sra_search_decision(decision_raw)
            if parse_error:
                parse_errors.append(parse_error)

            route = "skip"
            skills: list[dict[str, Any]] = []
            if call and call.get("operation") == "search":
                search_call_count = 1
                route = "search"
                retrieved_skill_ids = self._search(call)
                skills = [
                    self.corpus_by_id[skill_id]
                    for skill_id in retrieved_skill_ids
                    if skill_id in self.corpus_by_id
                ]
            elif call is None and parse_error is None:
                route = "skip"
            else:
                route = "skip_parse_fallback"

            solve_result = self.solve_engine.run(
                instance,
                skills,
                self.solve_client,
                self.model_name,
                corpus=self.corpus_by_id,
            )
            raw_output = str(getattr(solve_result, "raw_output", ""))
            solve_skill_ids = [
                str(item) for item in getattr(solve_result, "skill_ids_used", [])
            ]
            if not solve_skill_ids and self.solve_engine_name == "direct":
                solve_skill_ids = retrieved_skill_ids

            gold_ids = _gold_skill_ids(instance)
            recalled_gold = (
                bool(set(gold_ids) & set(retrieved_skill_ids)) if gold_ids else False
            )
            transcript_parts = [f"DECISION:\n{decision_raw}"]
            if retrieved_skill_ids:
                transcript_parts.append(
                    "RETRIEVED:\n" + json.dumps(retrieved_skill_ids, ensure_ascii=False)
                )
            solve_transcript = getattr(solve_result, "transcript", None)
            if solve_transcript:
                transcript_parts.append(f"SOLVE_TRANSCRIPT:\n{solve_transcript}")

            meta = {
                "route": route,
                "search_call_count": search_call_count,
                "load_call_count": 0,
                "retrieved_skill_ids": retrieved_skill_ids,
                "loaded_skill_ids": solve_skill_ids,
                "gold_skill_ids": gold_ids,
                "recalled_gold": recalled_gold,
                "parse_errors": parse_errors,
                "decision_input_tokens": decision.input_tokens,
                "decision_output_tokens": decision.output_tokens,
                "decision_total_tokens": decision.input_tokens + decision.output_tokens,
                "decision_latency_ms": decision.elapsed_ms,
                "decision_token_usage_source": decision.token_usage_source,
                "solve_engine": self.solve_engine_name,
                "solve_meta": getattr(solve_result, "meta", {}) or {},
                "wall_time_ms": round((time.perf_counter() - started) * 1000),
            }
            return SRAAgentRecord(
                instance_id=str(instance["instance_id"]),
                dataset=str(instance["dataset"]),
                method=self.method,
                model=_model_short_name(self.model_name),
                raw_output=raw_output,
                transcript="\n\n".join(transcript_parts),
                skill_ids_used=solve_skill_ids,
                meta=meta,
            )
        except Exception as exc:  # noqa: BLE001
            return SRAAgentRecord(
                instance_id=str(instance.get("instance_id", "")),
                dataset=str(instance.get("dataset", "")),
                method=self.method,
                model=_model_short_name(self.model_name),
                raw_output="",
                transcript=f"DECISION:\n{decision_raw}" if decision_raw else None,
                error=str(exc),
                meta={"wall_time_ms": round((time.perf_counter() - started) * 1000)},
            )

    def _search(self, call: dict[str, Any]) -> list[str]:
        intent = call.get("retrieval_intent")
        if not isinstance(intent, dict):
            intent = {"query": str(call.get("query") or "")}
        if not intent.get("query"):
            intent["query"] = str(call.get("query") or call.get("task_context") or "")
        try:
            request = SkillSearchRequest.parse_obj(_normalize_search_intent(intent))
        except ValidationError:
            return []
        response = self.searcher.search(request, top_k=self.top_k)
        return [card.id for card in response.results]


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
    pending = [instance for instance in instances if instance["instance_id"] not in done]
    agent = SRAGeneralPurposeAgent(
        searcher=searcher,
        corpus=corpus,
        llm=llm,
        model_name=model_name,
        top_k=top_k,
        max_rounds=max_rounds,
        sra_repo=sra_repo,
    )
    written = 0
    with output.open("a", encoding="utf-8") as handle:
        with _ProgressBar(
            total=len(pending),
            label=f"sra-agent {output.name}",
            initial_done=len(done),
        ) as progress:
            for instance in pending:
                record = agent.run_instance(instance)
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                handle.flush()
                written += 1
                progress.update()

    records = load_sra_agent_records(output)
    metrics = compute_sra_agent_skill_metrics(records, instances)
    return {
        "inference_output": str(output),
        "written": written,
        "total_records": len(records),
        "metrics": metrics,
    }


def run_sra_search_decision_inference(
    *,
    searcher: SkillSearcher,
    corpus: list[dict[str, Any]],
    instances_path: str | Path,
    output_path: str | Path,
    decision_llm: LLMClient,
    solve_engine: SRASolveEngine,
    solve_client: Any,
    model_name: str,
    top_k: int = 5,
    limit: int | None = None,
    force: bool = False,
    sra_repo: str | Path = SRA_SUBMODULE_DIR,
    solve_engine_name: str = "direct",
    method: str | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    instances = load_sra_instances(instances_path)
    if limit is not None:
        instances = instances[:limit]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if force and output.exists():
        output.unlink()

    done = _already_done(output)
    pending = [
        instance for instance in instances if instance["instance_id"] not in done
    ]
    agent = SRASearchDecisionAgent(
        searcher=searcher,
        corpus=corpus,
        decision_llm=decision_llm,
        solve_engine=solve_engine,
        solve_client=solve_client,
        model_name=model_name,
        top_k=top_k,
        sra_repo=sra_repo,
        method=method or f"skillbrowser_agent_top{top_k}_{solve_engine_name}",
        solve_engine_name=solve_engine_name,
    )

    written = 0
    with output.open("a", encoding="utf-8") as handle:
        with _ProgressBar(
            total=len(pending),
            label=f"sra-decision-agent {output.name}",
            initial_done=len(done),
        ) as progress:
            if workers <= 1:
                for instance in pending:
                    record = agent.run_instance(instance)
                    handle.write(
                        json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
                    )
                    handle.flush()
                    written += 1
                    progress.update()
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [
                        pool.submit(agent.run_instance, instance)
                        for instance in pending
                    ]
                    for future in as_completed(futures):
                        record = future.result()
                        handle.write(
                            json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
                        )
                        handle.flush()
                        written += 1
                        progress.update()

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
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def build_sra_prompt(
    instance: dict[str, Any], *, sra_repo: str | Path = SRA_SUBMODULE_DIR
) -> tuple[str, str]:
    """Return the dataset-native SRA prompt used by the solving agent.

    The search-decision agent intentionally keeps its own fixed decision system
    prompt. This helper is for task-solving prompts only.
    """
    repo_path = Path(sra_repo).resolve()
    src_path = repo_path / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from sragents.prompts import build_prompt  # type: ignore[import-not-found]

    return build_prompt(instance)


def parse_sra_agent_tool_call(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    text = _extract_json_object_text(raw)
    if not text:
        return None, None
    parsed, parse_error = _loads_json_object_with_repair(text)
    if parse_error:
        return None, f"Could not parse JSON tool/final output: {parse_error}"
    if not isinstance(parsed, dict):
        return None, "JSON output must be an object"
    if parsed.get("tool") == "skill_search" or parsed.get("operation") in {
        "search",
        "load_skill",
    }:
        parsed.setdefault("tool", "skill_search")
        return parsed, None
    return None, None


def parse_sra_search_decision(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    tool_text, tool_error = _extract_tool_block(raw)
    if tool_error:
        return None, tool_error
    if tool_text is None:
        return None, None
    text = _extract_json_object_text(tool_text)
    if not text:
        return None, "Tool block did not contain a JSON object"
    parsed, parse_error = _loads_json_object_with_repair(text)
    if parse_error:
        return None, f"Could not parse JSON tool call: {parse_error}"
    if not isinstance(parsed, dict):
        return None, "JSON tool call must be an object"
    operation = parsed.get("operation")
    if parsed.get("tool") == "skill_search" or operation == "search":
        parsed.setdefault("tool", "skill_search")
        parsed["operation"] = "search"
        return parsed, None
    return None, f"Unsupported tool operation: {operation}"


def _loads_json_object_with_repair(text: str) -> tuple[Any | None, str | None]:
    try:
        return json.loads(text), None
    except json.JSONDecodeError as strict_exc:
        if repair_json is None:
            return None, str(strict_exc)
        try:
            repaired = repair_json(text)
            return json.loads(repaired), None
        except Exception as repair_exc:  # noqa: BLE001
            return None, f"{strict_exc}; json_repair failed: {repair_exc}"


def _extract_tool_block(raw: str) -> tuple[str | None, str | None]:
    start_tag = "<tool>"
    end_tag = "</tool>"
    start = raw.find(start_tag)
    if start < 0:
        return None, None
    end = raw.find(end_tag, start + len(start_tag))
    if end < 0:
        return None, "Tool call is missing closing </tool> tag"
    return raw[start + len(start_tag) : end].strip(), None


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
    # SRA-generated SkillSpecs do not reliably carry input/output type labels.
    # Treat model-proposed types as explanatory text, not retrieval gates.
    normalized["input_types"] = []
    normalized["output_types"] = []
    for key in [
        "required_capabilities",
        "desired_capabilities",
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
    return [
        str(item) for item in instance.get("skill_annotations", []) if str(item).strip()
    ]


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


class _ProgressBar:
    def __init__(self, *, total: int, label: str, initial_done: int = 0):
        self.total = total
        self.label = label
        self.initial_done = initial_done
        self.current = 0
        self.started = time.perf_counter()
        self.last_render = 0.0

    def __enter__(self) -> "_ProgressBar":
        self._render(force=True)
        return self

    def __exit__(self, *_exc: object) -> None:
        self._render(force=True, final=True)
        if self.total:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def update(self, step: int = 1) -> None:
        self.current += step
        self._render()

    def _render(self, *, force: bool = False, final: bool = False) -> None:
        if self.total <= 0:
            return
        now = time.perf_counter()
        if not force and now - self.last_render < 0.2 and self.current < self.total:
            return
        self.last_render = now
        width = 24
        fraction = min(max(self.current / self.total, 0.0), 1.0)
        filled = round(width * fraction)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = max(now - self.started, 0.001)
        rate = self.current / elapsed
        remaining = (self.total - self.current) / rate if rate > 0 else 0.0
        status = "done" if final and self.current >= self.total else f"eta {_format_seconds(remaining)}"
        prefix = f"{self.label}: "
        resume = f" resumed after {self.initial_done}" if self.initial_done else ""
        sys.stderr.write(
            f"\r{prefix}[{bar}] {self.current}/{self.total} "
            f"({fraction:>6.1%}, {rate:.2f}/s, {status}){resume}"
        )
        sys.stderr.flush()


def _format_seconds(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


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
