"""Run SRA-Bench with this project's Hybrid skill searcher.

This script keeps SR-Agents as a git submodule while producing retrieval files
that its native infer/evaluate stages can consume.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.sra import (  # noqa: E402
    SRA_CORPUS_PATH,
    SRA_INSTANCES_DIR,
    SRA_RESULTS_DIR,
    SRA_SUBMODULE_DIR,
    ensure_sra_corpus,
    build_sra_query,
    compute_sra_retrieval_metrics,
    load_sra_corpus,
    load_sra_instances,
    run_sra_retrieval,
    sra_corpus_to_specs,
)
from benchmarks.sra_agent import (  # noqa: E402
    run_sra_agent_inference,
    run_sra_search_decision_inference,
    write_sra_agent_summary,
)
from benchmarks.sra_preprocess import (  # noqa: E402
    DETERMINISTIC_PREPROCESS_MODEL,
    DEFAULT_SRA_PREPROCESS_CHECKPOINT,
    DEFAULT_SRA_PREPROCESS_MODEL,
    DEFAULT_SRA_SKILL_DIR,
    run_sra_preprocess,
)
from cli import add_retrieval_arguments, build_searcher  # noqa: E402
from config import load_app_config, load_app_config_if_exists  # noqa: E402
from loader import SkillLoadError, load_skills  # noqa: E402
from llm import MockLLMClient, OpenAICompatibleLLMClient  # noqa: E402
from schema import SkillSearchRequest  # noqa: E402


SRA_METADATA_DENSE_VIEW_NAMES = {"description", "capability", "usage", "examples", "schema"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sra-bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Prepare the SR-Agents submodule data")
    prepare.add_argument("--corpus", default=str(SRA_CORPUS_PATH))

    preprocess = sub.add_parser("preprocess", help="Preprocess SRA corpus into local SkillSpec files")
    preprocess.add_argument("--corpus", default=str(SRA_CORPUS_PATH))
    preprocess.add_argument("--output-skill-dir")
    preprocess.add_argument("--checkpoint")
    preprocess.add_argument("--dataset", help="Optional SRA dataset filter, e.g. web, theoremqa, champ, or all")
    preprocess.add_argument("--mode", choices=["llm", "deterministic"], default="llm")
    preprocess.add_argument("--model", default=DEFAULT_SRA_PREPROCESS_MODEL)
    preprocess.add_argument("--llm", choices=["mock", "openai-compatible"], default="openai-compatible")
    preprocess.add_argument("--config", default="config.toml")
    preprocess.add_argument("--api-base", help="OpenAI-compatible endpoint override")
    preprocess.add_argument("--temperature", type=float, default=0.0)
    preprocess.add_argument("--max-tokens", type=int, default=1200)
    preprocess.add_argument("--limit", type=int)
    preprocess.add_argument("--resume", action="store_true")
    preprocess.add_argument("--force", action="store_true")

    retrieve = sub.add_parser("retrieve", help="Run SkillBrowser Hybrid retrieval on SRA-Bench")
    add_common_retrieval_args(retrieve)

    infer = sub.add_parser("infer", help="Run SR-Agents inference using a Hybrid retrieval file")
    add_common_infer_args(infer)

    infer_agent = sub.add_parser("infer-agent", help="Run the local SRA general-purpose skill_search agent")
    add_common_agent_infer_args(infer_agent)

    infer_decision_agent = sub.add_parser(
        "infer-decision-agent",
        help="Run LLM skill-search decision, then solve with an SR-Agents engine",
    )
    add_common_decision_agent_infer_args(infer_decision_agent)

    evaluate = sub.add_parser("evaluate", help="Score an SR-Agents inference JSONL file")
    add_common_eval_args(evaluate)

    summarize_agent = sub.add_parser("summarize-agent", help="Merge task accuracy with SRA agent skill metrics")
    add_common_agent_summary_args(summarize_agent)

    run = sub.add_parser("run", help="Run retrieve, infer, and evaluate per instance")
    add_common_e2e_args(run)

    run_staged = sub.add_parser("run-staged", help="Run staged prepare, retrieve, infer, and evaluate")
    add_common_retrieval_args(run_staged)
    add_common_infer_args(
        run_staged,
        include_dataset=False,
        include_retrieval_output=False,
        include_config=False,
        include_instances=False,
    )
    add_common_eval_args(run_staged, include_dataset=False, include_input=False, include_instances=False)

    run_e2e = sub.add_parser("run-e2e", help="Run retrieve, infer, and evaluate per instance")
    add_common_e2e_args(run_e2e)

    run_agent = sub.add_parser("run-agent", help="Run local SRA agent inference, evaluation, and summary")
    add_common_agent_infer_args(run_agent)
    add_common_eval_args(run_agent, include_dataset=False, include_input=False, include_instances=False)

    run_decision_agent = sub.add_parser(
        "run-decision-agent",
        help="Run search-decision agent inference, evaluation, and summary",
    )
    add_common_decision_agent_infer_args(run_decision_agent)
    add_common_eval_args(run_decision_agent, include_dataset=False, include_input=False, include_instances=False)
    return parser


def add_common_retrieval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="theoremqa", help="SRA-Bench dataset name")
    parser.add_argument("--corpus", default=str(SRA_CORPUS_PATH), help="Path to SRA corpus.json")
    parser.add_argument("--sra-skill-dir", help="Optional preprocessed SRA SkillSpec directory")
    parser.add_argument("--instances", help="Path to SRA instances JSON")
    parser.add_argument("--retrieval-output", help="Path for SR-Agents-compatible retrieval JSON")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--limit", type=int, help="Optional query limit for smoke tests")
    parser.add_argument("--config", default="config.toml")
    add_retrieval_arguments(parser, include_config=False)


def add_common_infer_args(
    parser: argparse.ArgumentParser,
    *,
    include_dataset: bool = True,
    include_retrieval_output: bool = True,
    include_config: bool = True,
    include_instances: bool = True,
) -> None:
    if include_dataset:
        parser.add_argument("--dataset", default="theoremqa")
    if include_instances:
        parser.add_argument("--instances", help="Path to SRA instances JSON")
    if include_retrieval_output:
        parser.add_argument("--retrieval-output", help="Path to retrieval JSON from the retrieve stage")
    parser.add_argument("--inference-output", help="Path for SR-Agents inference JSONL")
    parser.add_argument("--provider", choices=["topk", "oracle"], default="topk")
    parser.add_argument("--provider-k", type=int, default=1, help="How many retrieved skills to expose")
    parser.add_argument("--engine", default="direct", help="SR-Agents inference engine")
    parser.add_argument("--model", required=True, help="Model identifier passed to SR-Agents")
    parser.add_argument("--api-base", help="OpenAI-compatible endpoint")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--force", action="store_true")
    if include_config:
        parser.add_argument("--config", default="config.toml")


def add_common_eval_args(
    parser: argparse.ArgumentParser,
    *,
    include_dataset: bool = True,
    include_input: bool = True,
    include_instances: bool = True,
) -> None:
    if include_dataset:
        parser.add_argument("--dataset", default="theoremqa")
    if include_instances:
        parser.add_argument("--instances", help="Path to SRA instances JSON")
    if include_input:
        parser.add_argument("--inference-output", required=True, help="Path to inference JSONL")
    parser.add_argument("--eval-output", help="Path for SR-Agents eval JSON")
    parser.add_argument("--eval-workers", type=int, default=8)
    parser.add_argument("--eval-force", action="store_true")


def add_common_agent_infer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="theoremqa", help="SRA-Bench dataset name")
    parser.add_argument("--corpus", default=str(SRA_CORPUS_PATH), help="Path to SRA corpus.json")
    parser.add_argument("--sra-skill-dir", help="Optional preprocessed SRA SkillSpec directory")
    parser.add_argument("--instances", help="Path to SRA instances JSON")
    parser.add_argument("--inference-output", help="Path for local agent inference JSONL")
    parser.add_argument("--agent-top-k", type=int, default=5, help="Number of search candidates returned to the agent")
    parser.add_argument("--max-rounds", type=int, default=6, help="Maximum LLM/tool rounds per instance")
    parser.add_argument("--limit", type=int, help="Optional instance limit for smoke tests")
    parser.add_argument("--inject-full-search-skill", action="store_true")
    parser.add_argument("--llm", choices=["mock", "openai-compatible"], default="openai-compatible")
    parser.add_argument("--model", required=True, help="Model identifier recorded in inference output")
    parser.add_argument("--api-base", help="OpenAI-compatible endpoint override")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for agent-style runs that support it")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config", default="config.toml")
    add_retrieval_arguments(parser, include_config=False)


def add_common_e2e_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="theoremqa", help="SRA-Bench dataset name")
    parser.add_argument("--corpus", default=str(SRA_CORPUS_PATH), help="Path to SRA corpus.json")
    parser.add_argument("--sra-skill-dir", help="Optional preprocessed SRA SkillSpec directory")
    parser.add_argument("--instances", help="Path to SRA instances JSON")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, help="Optional instance limit for smoke tests")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--e2e-output", help="JSONL checkpoint with retrieval, inference, and eval per instance")
    parser.add_argument("--inference-output", help="SR-Agents-compatible inference JSONL")
    parser.add_argument("--eval-output", help="Aggregated SR-Agents-style eval JSON")
    parser.add_argument("--engine", default="direct", help="SR-Agents inference engine")
    parser.add_argument("--model", required=True, help="Model identifier passed to SR-Agents")
    parser.add_argument("--api-base", help="OpenAI-compatible endpoint override")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--force", action="store_true")
    add_retrieval_arguments(parser, include_config=False)


def add_common_decision_agent_infer_args(parser: argparse.ArgumentParser) -> None:
    add_common_agent_infer_args(parser)
    parser.add_argument("--solve-engine", default="direct", help="SR-Agents engine used after the decision phase")


def add_common_agent_summary_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="theoremqa")
    parser.add_argument("--instances", help="Path to SRA instances JSON")
    parser.add_argument("--inference-output", required=True, help="Path to local agent inference JSONL")
    parser.add_argument("--eval-output", help="Optional SR-Agents eval JSON to merge")
    parser.add_argument("--summary-output", help="Path for merged local agent summary JSON")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        path = ensure_sra_corpus(args.corpus)
        print(json.dumps({"ok": True, "corpus": str(path)}, indent=2))
        return 0
    if args.command == "preprocess":
        preprocess(args)
        return 0
    if args.command == "retrieve":
        retrieve(args)
        return 0
    if args.command == "infer":
        infer(args)
        return 0
    if args.command == "infer-agent":
        infer_agent(args)
        return 0
    if args.command == "infer-decision-agent":
        infer_decision_agent(args)
        return 0
    if args.command == "evaluate":
        evaluate(args)
        return 0
    if args.command == "summarize-agent":
        summarize_agent(args)
        return 0
    if args.command == "run":
        ensure_sra_corpus(args.corpus)
        run_e2e(args)
        return 0
    if args.command == "run-staged":
        ensure_sra_corpus(args.corpus)
        retrieve(args)
        infer(args)
        evaluate(args)
        return 0
    if args.command == "run-e2e":
        ensure_sra_corpus(args.corpus)
        run_e2e(args)
        return 0
    if args.command == "run-agent":
        ensure_sra_corpus(args.corpus)
        infer_agent(args)
        evaluate(args)
        summarize_agent(args)
        return 0
    if args.command == "run-decision-agent":
        ensure_sra_corpus(args.corpus)
        infer_decision_agent(args)
        evaluate(args)
        summarize_agent(args)
        return 0
    raise ValueError(f"Unhandled command: {args.command}")


def retrieve(args: argparse.Namespace) -> Path:
    corpus = load_sra_corpus(args.corpus)
    skills = load_sra_search_specs(args, corpus)
    searcher = build_sra_searcher(skills, args)
    instances = Path(args.instances) if args.instances else default_instances(args.dataset)
    output = Path(args.retrieval_output) if args.retrieval_output else default_retrieval_output(args.dataset)
    result = run_sra_retrieval(
        searcher=searcher,
        corpus_size=len(skills),
        instances_path=instances,
        output_path=output,
        top_k=args.top_k,
        limit=args.limit,
        sra_repo=ROOT / SRA_SUBMODULE_DIR,
    )
    print(json.dumps({"retrieval_output": str(output), "metrics": result["metrics"]}, indent=2))
    args.retrieval_output = str(output)
    return output


def infer(args: argparse.Namespace) -> Path:
    config = load_app_config_if_exists(getattr(args, "config", "config.toml"))
    retrieval = Path(args.retrieval_output) if args.retrieval_output else default_retrieval_output(args.dataset)
    instances = Path(getattr(args, "instances", "") or default_instances(args.dataset))
    output = Path(args.inference_output) if args.inference_output else default_inference_output(args.dataset, args.model, args.provider_k)
    command = [
        "uv",
        "run",
        "--project",
        str(ROOT / SRA_SUBMODULE_DIR),
        "sragents",
        "infer",
        "--instances",
        str(instances),
        "--output",
        str(output),
        "--model",
        args.model,
        "--provider",
        args.provider,
    ]
    corpus_path = Path(args.corpus) if hasattr(args, "corpus") else ROOT / SRA_CORPUS_PATH
    if args.provider == "topk":
        command.extend(
            [
                "--provider-arg",
                f"source={retrieval}",
                "--provider-arg",
                f"k={args.provider_k}",
                "--provider-arg",
                f"corpus_path={corpus_path}",
            ]
        )
        retrieval_name = retrieval.name.lower()
        if "sragents_bm25" in retrieval_name or "sragents-bm25" in retrieval_name:
            label = f"sragents_bm25_top{args.provider_k}_{args.engine}"
        else:
            label = f"skillbrowser_hybrid_top{args.provider_k}_{args.engine}"
    else:
        command.extend(["--provider-arg", f"corpus_path={corpus_path}"])
        label = f"oracle_{args.engine}"
    command.extend(
        [
            "--engine",
            args.engine,
            "--workers",
            str(args.workers),
            "--temperature",
            str(args.temperature),
            "--max-tokens",
            str(args.max_tokens),
            "--label",
            label,
        ]
    )
    api_base = args.api_base or (config.llm.base_url if config.llm else None)
    if api_base:
        command.extend(["--api-base", api_base])
    if args.force:
        command.append("--force")
    env = {**os.environ, "PYTHONUTF8": "1"}
    if config.llm and config.llm.api_key:
        env["OPENAI_API_KEY"] = config.llm.api_key
    run_command(command, env=env)
    args.inference_output = str(output)
    return output


def infer_agent(args: argparse.Namespace) -> Path:
    corpus = load_sra_corpus(args.corpus)
    skills = load_sra_search_specs(args, corpus)
    searcher = build_sra_searcher(skills, args)
    instances = Path(args.instances) if args.instances else default_instances(args.dataset)
    output = (
        Path(args.inference_output)
        if args.inference_output
        else default_agent_inference_output(args.dataset, args.model, args.agent_top_k)
    )
    llm = build_agent_llm(args)
    result = run_sra_agent_inference(
        searcher=searcher,
        corpus=corpus,
        instances_path=instances,
        output_path=output,
        llm=llm,
        model_name=args.model,
        top_k=args.agent_top_k,
        max_rounds=args.max_rounds,
        limit=args.limit,
        inject_full_search_skill=args.inject_full_search_skill,
        force=args.force,
        sra_repo=ROOT / SRA_SUBMODULE_DIR,
    )
    print(json.dumps(result, indent=2))
    args.inference_output = str(output)
    return output


def infer_decision_agent(args: argparse.Namespace) -> Path:
    corpus = load_sra_corpus(args.corpus)
    skills = load_sra_search_specs(args, corpus)
    searcher = build_sra_searcher(skills, args)
    instances = Path(args.instances) if args.instances else default_instances(args.dataset)
    output = (
        Path(args.inference_output)
        if args.inference_output
        else default_decision_agent_inference_output(
            args.dataset,
            args.model,
            args.agent_top_k,
            args.solve_engine,
        )
    )
    decision_llm = build_agent_llm(args)
    solve_engine, solve_client = build_sra_solve_runtime(args)
    result = run_sra_search_decision_inference(
        searcher=searcher,
        corpus=corpus,
        instances_path=instances,
        output_path=output,
        decision_llm=decision_llm,
        solve_engine=solve_engine,
        solve_client=solve_client,
        model_name=args.model,
        top_k=args.agent_top_k,
        limit=args.limit,
        force=args.force,
        sra_repo=ROOT / SRA_SUBMODULE_DIR,
        solve_engine_name=args.solve_engine,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2))
    args.inference_output = str(output)
    return output


def run_e2e(args: argparse.Namespace) -> Path:
    config = load_app_config(args.config)
    if config.llm is None:
        raise ValueError("Config file must include an [llm] section for run-e2e")

    corpus = load_sra_corpus(args.corpus)
    corpus_by_id = {str(skill.get("skill_id") or skill.get("id")): skill for skill in corpus}
    skills = load_sra_search_specs(args, corpus)
    searcher = build_sra_searcher(skills, args)
    instances_path = Path(args.instances) if args.instances else default_instances(args.dataset)
    instances = load_sra_instances(instances_path)
    if args.limit is not None:
        instances = instances[: args.limit]

    e2e_output = Path(args.e2e_output) if args.e2e_output else default_e2e_output(args.dataset, args.model, args.top_k)
    inference_output = (
        Path(args.inference_output)
        if args.inference_output
        else default_e2e_inference_output(args.dataset, args.model, args.top_k)
    )
    eval_output = Path(args.eval_output) if args.eval_output else default_eval_output(args.dataset, inference_output)
    if args.force:
        for path in (e2e_output, inference_output, eval_output):
            if path.exists():
                path.unlink()
    e2e_output.parent.mkdir(parents=True, exist_ok=True)
    inference_output.parent.mkdir(parents=True, exist_ok=True)
    eval_output.parent.mkdir(parents=True, exist_ok=True)

    done = _e2e_done(e2e_output)
    pending = [instance for instance in instances if instance["instance_id"] not in done]
    sra_src = (ROOT / SRA_SUBMODULE_DIR / "src").resolve()
    if str(sra_src) not in sys.path:
        sys.path.insert(0, str(sra_src))
    from sragents.evaluate import evaluate as sra_evaluate  # type: ignore[import-not-found]
    from sragents.evaluate.metrics import compute_accuracy  # type: ignore[import-not-found]
    from sragents.infer import get_engine  # type: ignore[import-not-found]
    from sragents.infer.schema import InferenceRecord  # type: ignore[import-not-found]
    from sragents.llm import create_llm_client  # type: ignore[import-not-found]

    client = create_llm_client(api_base=args.api_base or config.llm.base_url, api_key=config.llm.api_key)
    engine = get_engine(args.engine, temperature=args.temperature, max_tokens=args.max_tokens)
    label = f"skillbrowser_e2e_hybrid_top{args.top_k}_{args.engine}"
    model_name = Path(args.model).name.replace(":", "_")
    completed_records = _load_e2e_records(e2e_output)
    details = [record["evaluation"] for record in completed_records if record.get("evaluation")]
    retrieval_records = [record["retrieval"] for record in completed_records if record.get("retrieval")]
    print(
        json.dumps(
            {
                "event": "e2e_start",
                "instances": len(instances),
                "already_done": len(done),
                "pending": len(pending),
                "e2e_output": str(e2e_output),
            }
        ),
        flush=True,
    )

    with e2e_output.open("a", encoding="utf-8") as e2e_handle, inference_output.open("a", encoding="utf-8") as infer_handle:
        for index, instance in enumerate(pending, start=len(done) + 1):
            gold_ids = [str(item) for item in instance.get("skill_annotations", []) if str(item).strip()]
            query = build_sra_query(instance, sra_repo=ROOT / SRA_SUBMODULE_DIR)
            response = searcher.search(SkillSearchRequest(query=query), top_k=args.top_k)
            retrieved = [{"skill_id": card.id, "score": float(card.score)} for card in response.results]
            retrieved_skill_ids = [item["skill_id"] for item in retrieved]
            selected_skills = [corpus_by_id[skill_id] for skill_id in retrieved_skill_ids if skill_id in corpus_by_id]
            retrieval_record = {
                "instance_id": instance["instance_id"],
                "gold_skill_ids": gold_ids,
                "retrieved": retrieved,
            }
            try:
                result = engine.run(instance, selected_skills, client, args.model)
                inference_record = InferenceRecord(
                    instance_id=instance["instance_id"],
                    dataset=instance["dataset"],
                    method=label,
                    model=model_name,
                    raw_output=result.raw_output,
                    transcript=result.transcript,
                    skill_ids_used=result.skill_ids_used,
                    meta=result.meta,
                ).to_dict()
                evaluation = {
                    "instance_id": instance["instance_id"],
                    "dataset": instance["dataset"],
                    "method": label,
                    "model": model_name,
                    **sra_evaluate(result.raw_output, instance),
                }
            except Exception as exc:  # noqa: BLE001
                inference_record = {
                    "instance_id": instance["instance_id"],
                    "dataset": instance["dataset"],
                    "method": label,
                    "model": model_name,
                    "raw_output": "",
                    "error": str(exc),
                }
                evaluation = {
                    "instance_id": instance["instance_id"],
                    "dataset": instance["dataset"],
                    "method": label,
                    "model": model_name,
                    "extracted_answer": "",
                    "correct": False,
                    "error": str(exc),
                }

            e2e_record = {
                "instance_id": instance["instance_id"],
                "dataset": instance["dataset"],
                "retrieval": retrieval_record,
                "inference": inference_record,
                "evaluation": evaluation,
            }
            e2e_handle.write(json.dumps(e2e_record, ensure_ascii=False) + "\n")
            e2e_handle.flush()
            infer_handle.write(json.dumps(inference_record, ensure_ascii=False) + "\n")
            infer_handle.flush()
            retrieval_records.append(retrieval_record)
            details.append(evaluation)
            if index % 10 == 0 or index == len(instances):
                metrics = compute_accuracy(details)
                retrieval_metrics = compute_sra_retrieval_metrics(retrieval_records, top_k=args.top_k)
                print(
                    json.dumps(
                        {
                            "event": "e2e_progress",
                            "done": index,
                            "total": len(instances),
                            "accuracy": round(metrics.get("accuracy", 0.0), 6),
                            "retrieval": retrieval_metrics,
                        }
                    ),
                    flush=True,
                )

    metrics = compute_accuracy(details)
    payload = {
        "dataset": args.dataset,
        "method": label,
        "model": model_name,
        "metrics": metrics,
        "retrieval_metrics": compute_sra_retrieval_metrics(retrieval_records, top_k=args.top_k),
        "details": details,
        "e2e_output": str(e2e_output),
        "inference_output": str(inference_output),
    }
    eval_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"eval_output": str(eval_output), **metrics}, indent=2), flush=True)
    return eval_output


def preprocess(args: argparse.Namespace) -> Path:
    resolve_preprocess_defaults(args)
    llm = None if args.mode == "deterministic" else build_preprocess_llm(args)
    result = run_sra_preprocess(
        corpus_path=args.corpus,
        output_skill_dir=args.output_skill_dir,
        checkpoint_path=args.checkpoint,
        llm=llm,
        model_name=args.model,
        limit=args.limit,
        resume=args.resume,
        force=args.force,
        dataset=args.dataset,
        mode=args.mode,
    )
    print(json.dumps(result, indent=2))
    return Path(args.output_skill_dir)


def load_sra_search_specs(args: argparse.Namespace, corpus: list[dict]):
    skill_dir = getattr(args, "sra_skill_dir", None)
    if skill_dir:
        return load_skills(Path(skill_dir))
    config = load_app_config_if_exists(getattr(args, "config", "config.toml"))
    if config.sra.skill_dirs:
        return load_sra_skill_dirs(config.sra.skill_dirs)
    return sra_corpus_to_specs(corpus)


def build_sra_searcher(skills, args: argparse.Namespace):
    has_preprocessed = bool(getattr(args, "sra_skill_dir", None)) or bool(
        load_app_config_if_exists(getattr(args, "config", "config.toml")).sra.skill_dirs
    )
    dense_view_names = SRA_METADATA_DENSE_VIEW_NAMES if has_preprocessed else None
    return build_searcher(skills, args, dense_view_names=dense_view_names)


def resolve_preprocess_defaults(args: argparse.Namespace) -> None:
    dataset = (args.dataset or "").strip().lower()
    if args.mode == "deterministic" and args.model == DEFAULT_SRA_PREPROCESS_MODEL:
        args.model = DETERMINISTIC_PREPROCESS_MODEL
    if args.output_skill_dir is None:
        if dataset:
            args.output_skill_dir = str(ROOT / "data" / "eval" / "sra" / dataset)
        else:
            args.output_skill_dir = str(DEFAULT_SRA_SKILL_DIR)
    if args.checkpoint is None:
        if dataset:
            args.checkpoint = str(ROOT / "data" / "eval" / "sra" / "preprocess" / f"{dataset}.jsonl")
        else:
            args.checkpoint = str(DEFAULT_SRA_PREPROCESS_CHECKPOINT)


def load_sra_skill_dirs(skill_dirs: list[str]):
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
        raise SkillLoadError(f"Duplicate skill id(s) across SRA skill dirs: {', '.join(sorted(duplicates))}")
    return skills


def evaluate(args: argparse.Namespace) -> Path:
    inference = Path(args.inference_output)
    instances = Path(getattr(args, "instances", "") or default_instances(args.dataset))
    output = Path(args.eval_output) if args.eval_output else default_eval_output(args.dataset, inference)
    command = [
        "uv",
        "run",
        "--project",
        str(ROOT / SRA_SUBMODULE_DIR),
        "sragents",
        "evaluate",
        "--input",
        str(inference),
        "--instances",
        str(instances),
        "--output",
        str(output),
        "--workers",
        str(args.eval_workers),
    ]
    if args.eval_force:
        command.append("--force")
    run_command(command, env={**os.environ, "PYTHONUTF8": "1"})
    args.eval_output = str(output)
    return output


def summarize_agent(args: argparse.Namespace) -> Path:
    inference = Path(args.inference_output)
    instances = Path(getattr(args, "instances", "") or default_instances(args.dataset))
    eval_output = getattr(args, "eval_output", None)
    eval_path = Path(eval_output) if eval_output else None
    output = (
        Path(args.summary_output)
        if getattr(args, "summary_output", None)
        else default_agent_summary_output(args.dataset, inference)
    )
    payload = write_sra_agent_summary(
        inference_path=inference,
        instances_path=instances,
        output_path=output,
        eval_path=eval_path,
    )
    print(json.dumps({"summary_output": str(output), **payload.get("agent_metrics", {})}, indent=2))
    return output


def _load_e2e_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    good_bytes = 0
    data = path.read_bytes()
    for raw in data.splitlines(keepends=True):
        stripped = raw.decode("utf-8", errors="replace").strip()
        if not stripped:
            good_bytes += len(raw)
            continue
        if not raw.endswith(b"\n"):
            break
        try:
            records.append(json.loads(stripped))
            good_bytes += len(raw)
        except json.JSONDecodeError:
            break
    if good_bytes < len(data):
        with path.open("rb+") as handle:
            handle.truncate(good_bytes)
    return records


def _e2e_done(path: Path) -> set[str]:
    return {
        record["instance_id"]
        for record in _load_e2e_records(path)
        if isinstance(record.get("instance_id"), str)
    }


def build_agent_llm(args: argparse.Namespace):
    if args.llm == "mock":
        return MockLLMClient()
    config = load_app_config(args.config)
    if config.llm is None:
        raise ValueError("Config file must include an [llm] section for openai-compatible SRA agent mode")
    return OpenAICompatibleLLMClient(
        base_url=args.api_base or config.llm.base_url,
        api_key=config.llm.api_key,
        model=args.model or config.llm.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=config.llm.timeout,
    )


def build_sra_solve_runtime(args: argparse.Namespace):
    if args.llm == "mock":
        class _MockSolveEngine:
            def run(self, instance, skills, client, model, **kwargs):
                return SimpleNamespace(
                    raw_output="Therefore, the answer is 0.5",
                    transcript=None,
                    skill_ids_used=[skill["skill_id"] for skill in skills],
                    meta={},
                )

        return _MockSolveEngine(), None

    sra_src = (ROOT / SRA_SUBMODULE_DIR / "src").resolve()
    if str(sra_src) not in sys.path:
        sys.path.insert(0, str(sra_src))
    from sragents.infer import get_engine  # type: ignore[import-not-found]
    from sragents.llm import create_llm_client  # type: ignore[import-not-found]

    config = load_app_config(args.config)
    api_base = args.api_base or (config.llm.base_url if config.llm else None)
    api_key = config.llm.api_key if config.llm else None
    engine_kwargs = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.solve_engine in {"progressive_disclosure", "react_progressive_disclosure"}:
        engine_kwargs["max_rounds"] = args.max_rounds
    engine = get_engine(args.solve_engine, **engine_kwargs)
    client = create_llm_client(api_base=api_base, api_key=api_key)
    return engine, client


def build_preprocess_llm(args: argparse.Namespace):
    if args.llm == "mock":
        return MockLLMClient(
            [
                json.dumps(
                    {
                        "short_description": "Use the named SRA skill for matching benchmark tasks.",
                        "long_description": (
                            "This generated smoke metadata describes the skill using its title, "
                            "description, and source content. Use it when a query asks for the same "
                            "method, formula, tool, or reasoning pattern."
                        ),
                        "capabilities": [
                            {
                                "id": "apply_sra_skill",
                                "description": "Apply the SR-Agents skill to a matching benchmark task.",
                            }
                        ],
                        "when_to_use": ["The task asks for the same method or tool described by this skill."],
                        "positive_examples": [
                            {
                                "user_query": "Find the matching SR-Agents skill for this task.",
                                "reason": "Smoke metadata for preprocessing tests.",
                            }
                        ],
                        "tags": ["sra-bench", "smoke"],
                    }
                )
            ]
        )
    config = load_app_config(args.config)
    if config.llm is None:
        raise ValueError("Config file must include an [llm] section for openai-compatible SRA preprocessing")
    return OpenAICompatibleLLMClient(
        base_url=args.api_base or config.llm.base_url,
        api_key=config.llm.api_key,
        model=args.model or config.llm.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=config.llm.timeout,
    )


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=env)


def default_instances(dataset: str) -> Path:
    return ROOT / SRA_INSTANCES_DIR / f"{dataset}.json"


def default_retrieval_output(dataset: str) -> Path:
    return ROOT / SRA_RESULTS_DIR / "retrieval" / f"{dataset}-skillbrowser-hybrid.json"


def default_inference_output(dataset: str, model: str, provider_k: int) -> Path:
    model_name = Path(model).name.replace(":", "_")
    return ROOT / SRA_RESULTS_DIR / "inference" / f"{dataset}-{model_name}-hybrid_top{provider_k}.jsonl"


def default_agent_inference_output(dataset: str, model: str, agent_top_k: int) -> Path:
    model_name = Path(model).name.replace(":", "_")
    return ROOT / SRA_RESULTS_DIR / "inference" / f"{dataset}-{model_name}-general_agent_top{agent_top_k}.jsonl"


def default_decision_agent_inference_output(dataset: str, model: str, agent_top_k: int, solve_engine: str) -> Path:
    model_name = Path(model).name.replace(":", "_")
    engine_name = solve_engine.replace(":", "_")
    return (
        ROOT
        / SRA_RESULTS_DIR
        / "inference"
        / f"{dataset}-{model_name}-search_decision_top{agent_top_k}_{engine_name}.jsonl"
    )


def default_e2e_output(dataset: str, model: str, top_k: int) -> Path:
    model_name = Path(model).name.replace(":", "_")
    return ROOT / SRA_RESULTS_DIR / "e2e" / f"{dataset}-{model_name}-hybrid_top{top_k}-e2e.jsonl"


def default_e2e_inference_output(dataset: str, model: str, top_k: int) -> Path:
    model_name = Path(model).name.replace(":", "_")
    return ROOT / SRA_RESULTS_DIR / "inference" / f"{dataset}-{model_name}-hybrid_top{top_k}-e2e.jsonl"


def default_eval_output(dataset: str, inference: Path) -> Path:
    return ROOT / SRA_RESULTS_DIR / "eval" / f"{dataset}-{inference.stem}.json"


def default_agent_summary_output(dataset: str, inference: Path) -> Path:
    return ROOT / SRA_RESULTS_DIR / "eval" / f"{dataset}-{inference.stem}-agent-summary.json"


if __name__ == "__main__":
    raise SystemExit(main())
