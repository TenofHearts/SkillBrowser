"""Post-hoc selective procedural Skill exposure analysis for SRA outputs.

Usage:
  python scripts/analyze_selective_skill_exposure.py

The script reads existing Direct/No-Skill, Oracle, and SkillBrowser outputs. It
does not run inference. By default it analyzes qwen3-8b and qwen3.6-plus on the
non-code direct benchmarks where all available per-instance outputs can be
aligned.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "data" / "eval" / "sra" / "results"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "analysis" / "selective_exposure"
DEFAULT_NOSKILL_TARBALL = DEFAULT_RESULTS / "remote_downloads" / "results_qwen_noskill_bundle.tar.gz"
DEFAULT_MODELS = ("qwen3-8b", "qwen3.6-plus")
DEFAULT_BENCHMARKS = ("theoremqa", "logicbench", "champ", "medcalcbench")
CANONICAL_SKILLBROWSER_METHOD = "skillbrowser_agent_top5_direct"
PREFERRED_SKILLBROWSER_SUFFIXES = (
    "skillbrowser_agent_top5_direct-full-reusable-skill-prompt",
    CANONICAL_SKILLBROWSER_METHOD,
)
FALLBACK_SKILLBROWSER_SUFFIXES = (
    "skillbrowser_agent_top5_direct-full-tooltags",
    "skillbrowser_agent_top5_direct-current-prompt-rerun",
)
EXPERIMENTAL_SUFFIX_MARKERS = (
    "no_gold",
    "postfilter",
    "wrong-rerun",
    "wrong_fixed",
    "subset100",
    "threshold",
    ".bak",
)


@dataclass
class SourcePaths:
    direct: str
    oracle_eval: Path
    skillbrowser_eval: Path | None
    skillbrowser_inference: Path | None


@dataclass
class AnalysisRecord:
    model: str
    benchmark: str
    instance_id: str
    direct_correct: bool
    oracle_correct: bool
    skillbrowser_correct: bool
    skillbrowser_search_invoked: bool
    skillbrowser_skill_exposed: bool
    skillbrowser_loaded_skill_names: list[str] = field(default_factory=list)
    skillbrowser_exposed_skill_count: int = 0
    gold_skill_names: list[str] = field(default_factory=list)
    has_skillbrowser: bool = True


@dataclass
class WarningLog:
    messages: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        self.messages.append(message)
        print(f"WARNING: {message}")


def pct(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "NA"
    return f"{100.0 * value:.1f}"


def fmt_num(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def model_tar_name(model: str) -> str:
    return model.replace(".", "_")


def load_eval_details(path: Path, warnings: WarningLog, label: str) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    details = payload.get("details")
    if not isinstance(details, list):
        warnings.warn(f"{label} has no details list: {path}")
        return {}
    out: dict[str, dict[str, Any]] = {}
    counts = Counter(str(item.get("instance_id")) for item in details if item.get("instance_id") is not None)
    dupes = sorted(k for k, v in counts.items() if v > 1)
    if dupes:
        warnings.warn(f"{label} has duplicated instance IDs in {path}: {dupes[:5]}")
    for item in details:
        instance_id = item.get("instance_id")
        if instance_id is None:
            continue
        out.setdefault(str(instance_id), item)
    return out


def load_direct_details_from_tar(
    tarball: Path,
    model: str,
    benchmark: str,
    warnings: WarningLog,
) -> tuple[dict[str, dict[str, Any]], str | None]:
    if not tarball.exists():
        warnings.warn(f"Direct/No-Skill tarball missing: {tarball}")
        return {}, None
    tar_model = model_tar_name(model)
    member = (
        f"results_{tar_model}_noskill/eval/{benchmark}/"
        f"{tar_model}_noskill_direct_seed0.json"
    )
    try:
        with tarfile.open(tarball, "r:gz") as tf:
            try:
                handle = tf.extractfile(member)
            except KeyError:
                warnings.warn(f"Direct/No-Skill eval missing in tarball: {member}")
                return {}, member
            if handle is None:
                warnings.warn(f"Direct/No-Skill eval unreadable in tarball: {member}")
                return {}, member
            payload = json.loads(handle.read().decode("utf-8"))
    except tarfile.TarError as exc:
        warnings.warn(f"Could not read Direct/No-Skill tarball {tarball}: {exc}")
        return {}, member
    details = payload.get("details")
    if not isinstance(details, list):
        warnings.warn(f"Direct/No-Skill eval has no details list: {member}")
        return {}, member
    counts = Counter(str(item.get("instance_id")) for item in details if item.get("instance_id") is not None)
    dupes = sorted(k for k, v in counts.items() if v > 1)
    if dupes:
        warnings.warn(f"Direct/No-Skill has duplicated instance IDs in {member}: {dupes[:5]}")
    out: dict[str, dict[str, Any]] = {}
    for item in details:
        instance_id = item.get("instance_id")
        if instance_id is not None:
            out.setdefault(str(instance_id), item)
    return out, f"{tarball}!{member}"


def load_direct_details(
    *,
    results_root: Path,
    tarball: Path,
    model: str,
    benchmark: str,
    warnings: WarningLog,
) -> tuple[dict[str, dict[str, Any]], str | None]:
    eval_dir = results_root / "eval" / model
    for suffix in ("llm_direct-limit300", "llm_direct"):
        path = eval_dir / f"{benchmark}-{suffix}.json"
        if path.exists():
            return load_eval_details(path, warnings, f"Direct/No-Skill {model}/{benchmark}"), str(path)
    return load_direct_details_from_tar(tarball, model, benchmark, warnings)


def load_jsonl_records(path: Path, warnings: WarningLog, label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            warnings.warn(f"{label} JSONL parse error at {path}:{line_number}: {exc}")
            continue
        instance_id = record.get("instance_id")
        if instance_id is None:
            continue
        key = str(instance_id)
        counts[key] += 1
        out.setdefault(key, record)
    dupes = sorted(k for k, v in counts.items() if v > 1)
    if dupes:
        warnings.warn(f"{label} has duplicated instance IDs in {path}: {dupes[:5]}")
    return out


def resolve_skillbrowser_paths(
    results_root: Path,
    model: str,
    benchmark: str,
    allow_fallback: bool,
    warnings: WarningLog,
    preferred_suffixes: Iterable[str] | None = None,
) -> tuple[Path | None, Path | None]:
    eval_dir = results_root / "eval" / model
    inf_dir = results_root / "inference" / model
    suffixes = tuple(preferred_suffixes or PREFERRED_SKILLBROWSER_SUFFIXES)
    for suffix in suffixes:
        if any(marker in suffix for marker in EXPERIMENTAL_SUFFIX_MARKERS):
            warnings.warn(f"Skipping experimental SkillBrowser suffix for {model}/{benchmark}: {suffix}")
            continue
        eval_path = eval_dir / f"{benchmark}-{suffix}.json"
        inf_path = inf_dir / f"{benchmark}-{suffix}.jsonl"
        if eval_path.exists() and inf_path.exists():
            return eval_path, inf_path
        if eval_path.exists() != inf_path.exists():
            warnings.warn(
                f"Partial preferred SkillBrowser files for {model}/{benchmark}/{suffix}: "
                f"eval={eval_path.exists()} inference={inf_path.exists()}"
            )
    if not allow_fallback:
        warnings.warn(f"Missing preferred SkillBrowser pair for {model}/{benchmark}")
        return None, None
    for suffix in FALLBACK_SKILLBROWSER_SUFFIXES:
        if any(marker in suffix for marker in EXPERIMENTAL_SUFFIX_MARKERS):
            continue
        eval_path = eval_dir / f"{benchmark}-{suffix}.json"
        inf_path = inf_dir / f"{benchmark}-{suffix}.jsonl"
        if eval_path.exists() and inf_path.exists():
            warnings.warn(f"Using generic SkillBrowser fallback for {model}/{benchmark}: {suffix}")
            return eval_path, inf_path
    warnings.warn(f"No SkillBrowser fallback pair found for {model}/{benchmark}")
    return None, None


def derive_exposure(record: dict[str, Any]) -> tuple[bool, bool, list[str], int]:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    search_count = int(meta.get("search_call_count") or 0)
    loaded = record.get("skill_ids_used")
    if not isinstance(loaded, list) or not loaded:
        loaded = meta.get("loaded_skill_ids")
    if not isinstance(loaded, list):
        loaded = []
    loaded_names = [str(item) for item in loaded if str(item)]
    return search_count > 0, bool(loaded_names), loaded_names, len(loaded_names)


def normalize_pair(
    *,
    model: str,
    benchmark: str,
    direct: dict[str, dict[str, Any]],
    oracle: dict[str, dict[str, Any]],
    skill_eval: dict[str, dict[str, Any]],
    skill_inf: dict[str, dict[str, Any]],
    warnings: WarningLog,
) -> list[AnalysisRecord]:
    direct_oracle = set(direct) & set(oracle)
    if set(direct) != direct_oracle:
        warnings.warn(
            f"Coverage mismatch for {model}/{benchmark}/direct: "
            f"{len(set(direct) - direct_oracle)} extra, {len(direct_oracle - set(direct))} missing relative to Direct+Oracle set"
        )
    if set(oracle) != direct_oracle:
        warnings.warn(
            f"Coverage mismatch for {model}/{benchmark}/oracle: "
            f"{len(set(oracle) - direct_oracle)} extra, {len(direct_oracle - set(oracle))} missing relative to Direct+Oracle set"
        )
    for name, keys in {"skillbrowser_eval": set(skill_eval), "skillbrowser_inference": set(skill_inf)}.items():
        missing_count = len(direct_oracle - keys)
        extra_count = len(keys - direct_oracle)
        if missing_count or extra_count:
            warnings.warn(
                f"Coverage mismatch for {model}/{benchmark}/{name}: "
                f"{extra_count} extra, {missing_count} missing relative to Direct+Oracle set"
            )
    records: list[AnalysisRecord] = []
    for instance_id in sorted(direct_oracle):
        has_skillbrowser = instance_id in skill_eval and instance_id in skill_inf
        if has_skillbrowser:
            search_invoked, exposed, loaded_names, loaded_count = derive_exposure(skill_inf[instance_id])
            skillbrowser_correct = bool(skill_eval[instance_id].get("correct"))
        else:
            search_invoked, exposed, loaded_names, loaded_count = False, False, [], 0
            skillbrowser_correct = False
        records.append(
            AnalysisRecord(
                model=model,
                benchmark=benchmark,
                instance_id=instance_id,
                direct_correct=bool(direct[instance_id].get("correct")),
                oracle_correct=bool(oracle[instance_id].get("correct")),
                skillbrowser_correct=skillbrowser_correct,
                skillbrowser_search_invoked=search_invoked,
                skillbrowser_skill_exposed=exposed,
                skillbrowser_loaded_skill_names=loaded_names,
                skillbrowser_exposed_skill_count=loaded_count,
                gold_skill_names=[str(x) for x in oracle[instance_id].get("skill_annotations", [])],
                has_skillbrowser=has_skillbrowser,
            )
        )
    return records


def category_for(record: AnalysisRecord) -> str:
    if not record.direct_correct and record.oracle_correct:
        return "beneficial"
    if record.direct_correct and record.oracle_correct:
        return "unnecessary"
    if record.direct_correct and not record.oracle_correct:
        return "harmful"
    return "unresolved"


def rate(records: list[AnalysisRecord], field: str) -> float:
    if not records:
        return math.nan
    return sum(1 for r in records if bool(getattr(r, field))) / len(records)


def evaluable(records: list[AnalysisRecord]) -> list[AnalysisRecord]:
    return [record for record in records if record.has_skillbrowser]


def mean_count(records: list[AnalysisRecord]) -> float:
    if not records:
        return math.nan
    return sum(r.skillbrowser_exposed_skill_count for r in records) / len(records)


def bootstrap_rate(records: list[AnalysisRecord], field: str, seed: int, n: int = 1000) -> tuple[float, float]:
    if not records:
        return math.nan, math.nan
    rng = Random(seed)
    values = []
    size = len(records)
    for _ in range(n):
        sample = [records[rng.randrange(size)] for _ in range(size)]
        values.append(rate(sample, field))
    values.sort()
    lo = values[int(0.025 * (n - 1))]
    hi = values[int(0.975 * (n - 1))]
    return lo, hi


def summarize_group(records: list[AnalysisRecord], *, seed: int, bootstrap_samples: int = 1000) -> dict[str, Any]:
    groups: dict[str, list[AnalysisRecord]] = defaultdict(list)
    for record in records:
        groups[category_for(record)].append(record)
    total_categorized = sum(len(v) for v in groups.values())
    assert total_categorized == len(records), "Direct/Oracle categories do not cover all aligned records"
    beneficial = groups["beneficial"]
    unnecessary = groups["unnecessary"]
    beneficial_eval = evaluable(beneficial)
    unnecessary_eval = evaluable(unnecessary)
    ber_lo, ber_hi = bootstrap_rate(beneficial_eval, "skillbrowser_skill_exposed", seed, bootstrap_samples)
    uer_lo, uer_hi = bootstrap_rate(unnecessary_eval, "skillbrowser_skill_exposed", seed + 1, bootstrap_samples)
    return {
        "n_total": len(records),
        "n_evaluable": len(evaluable(records)),
        "n_beneficial": len(beneficial),
        "n_beneficial_evaluable": len(beneficial_eval),
        "ber": rate(beneficial_eval, "skillbrowser_skill_exposed"),
        "ber_ci_low": ber_lo,
        "ber_ci_high": ber_hi,
        "acc_beneficial": rate(beneficial_eval, "skillbrowser_correct"),
        "beneficial_search_invocation_rate": rate(beneficial_eval, "skillbrowser_search_invoked"),
        "beneficial_avg_exposed_skill_count": mean_count(beneficial_eval),
        "acc_beneficial_exposed": rate(
            [r for r in beneficial_eval if r.skillbrowser_skill_exposed], "skillbrowser_correct"
        ),
        "acc_beneficial_not_exposed": rate(
            [r for r in beneficial_eval if not r.skillbrowser_skill_exposed], "skillbrowser_correct"
        ),
        "n_unnecessary": len(unnecessary),
        "n_unnecessary_evaluable": len(unnecessary_eval),
        "uer": rate(unnecessary_eval, "skillbrowser_skill_exposed"),
        "uer_ci_low": uer_lo,
        "uer_ci_high": uer_hi,
        "acc_unnecessary": rate(unnecessary_eval, "skillbrowser_correct"),
        "unnecessary_search_invocation_rate": rate(unnecessary_eval, "skillbrowser_search_invoked"),
        "unnecessary_avg_exposed_skill_count": mean_count(unnecessary_eval),
        "n_harmful": len(groups["harmful"]),
        "n_unresolved": len(groups["unresolved"]),
    }


def per_pair_rows(records: list[AnalysisRecord], seed: int, bootstrap_samples: int = 1000) -> list[dict[str, Any]]:
    rows = []
    grouped: dict[tuple[str, str], list[AnalysisRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.model, record.benchmark)].append(record)
    assert sum(len(items) for items in grouped.values()) == len(records)
    for idx, ((model, benchmark), items) in enumerate(sorted(grouped.items())):
        summary = summarize_group(items, seed=seed + idx * 10, bootstrap_samples=bootstrap_samples)
        rows.append({"model": model, "benchmark": benchmark, **summary})
    return rows


def aggregate_by_benchmark(
    records: list[AnalysisRecord],
    pair_rows: list[dict[str, Any]],
    seed: int,
    bootstrap_samples: int = 1000,
) -> list[dict[str, Any]]:
    out = []
    by_bench_records: dict[str, list[AnalysisRecord]] = defaultdict(list)
    by_bench_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_bench_records[record.benchmark].append(record)
    for row in pair_rows:
        by_bench_pairs[str(row["benchmark"])].append(row)
    for idx, benchmark in enumerate(sorted(by_bench_records)):
        micro = summarize_group(
            by_bench_records[benchmark],
            seed=seed + 1000 + idx * 10,
            bootstrap_samples=bootstrap_samples,
        )
        pairs = by_bench_pairs[benchmark]
        macro_ber_values = [r["ber"] for r in pairs if not math.isnan(float(r["ber"]))]
        macro_uer_values = [r["uer"] for r in pairs if not math.isnan(float(r["uer"]))]
        out.append(
            {
                "benchmark": benchmark,
                "model_cells": len(pairs),
                "n_total": micro["n_total"],
                "n_evaluable": micro["n_evaluable"],
                "n_beneficial": micro["n_beneficial"],
                "n_beneficial_evaluable": micro["n_beneficial_evaluable"],
                "ber_micro": micro["ber"],
                "ber_micro_ci_low": micro["ber_ci_low"],
                "ber_micro_ci_high": micro["ber_ci_high"],
                "ber_macro": mean(macro_ber_values) if macro_ber_values else math.nan,
                "acc_beneficial_micro": micro["acc_beneficial"],
                "n_unnecessary": micro["n_unnecessary"],
                "n_unnecessary_evaluable": micro["n_unnecessary_evaluable"],
                "uer_micro": micro["uer"],
                "uer_micro_ci_low": micro["uer_ci_low"],
                "uer_micro_ci_high": micro["uer_ci_high"],
                "uer_macro": mean(macro_uer_values) if macro_uer_values else math.nan,
                "acc_unnecessary_micro": micro["acc_unnecessary"],
                "beneficial_search_invocation_rate_micro": micro["beneficial_search_invocation_rate"],
                "unnecessary_search_invocation_rate_micro": micro["unnecessary_search_invocation_rate"],
                "n_harmful": micro["n_harmful"],
                "n_unresolved": micro["n_unresolved"],
            }
        )
    return out


def overall_rows(
    records: list[AnalysisRecord],
    pair_rows: list[dict[str, Any]],
    seed: int,
    bootstrap_samples: int = 1000,
) -> list[dict[str, Any]]:
    micro = summarize_group(records, seed=seed + 2000, bootstrap_samples=bootstrap_samples)
    def macro(name: str) -> float:
        vals = [float(row[name]) for row in pair_rows if not math.isnan(float(row[name]))]
        return mean(vals) if vals else math.nan
    return [
        {
            "aggregation": "macro_model_benchmark",
            "model_benchmark_cells": len(pair_rows),
            "n_total": sum(int(r["n_total"]) for r in pair_rows),
            "n_evaluable": sum(int(r["n_evaluable"]) for r in pair_rows),
            "n_beneficial": sum(int(r["n_beneficial"]) for r in pair_rows),
            "n_beneficial_evaluable": sum(int(r["n_beneficial_evaluable"]) for r in pair_rows),
            "ber": macro("ber"),
            "acc_beneficial": macro("acc_beneficial"),
            "n_unnecessary": sum(int(r["n_unnecessary"]) for r in pair_rows),
            "n_unnecessary_evaluable": sum(int(r["n_unnecessary_evaluable"]) for r in pair_rows),
            "uer": macro("uer"),
            "acc_unnecessary": macro("acc_unnecessary"),
        },
        {
            "aggregation": "micro_instances",
            "model_benchmark_cells": len(pair_rows),
            "n_total": micro["n_total"],
            "n_evaluable": micro["n_evaluable"],
            "n_beneficial": micro["n_beneficial"],
            "n_beneficial_evaluable": micro["n_beneficial_evaluable"],
            "ber": micro["ber"],
            "ber_ci_low": micro["ber_ci_low"],
            "ber_ci_high": micro["ber_ci_high"],
            "acc_beneficial": micro["acc_beneficial"],
            "n_unnecessary": micro["n_unnecessary"],
            "n_unnecessary_evaluable": micro["n_unnecessary_evaluable"],
            "uer": micro["uer"],
            "uer_ci_low": micro["uer_ci_low"],
            "uer_ci_high": micro["uer_ci_high"],
            "acc_unnecessary": micro["acc_unnecessary"],
            "n_harmful": micro["n_harmful"],
            "n_unresolved": micro["n_unresolved"],
        },
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt_num(row.get(key)) for key in fields})


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def write_latex(path: Path, benchmark_rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Benchmark & \#Ben. & \#Ben. Eval. & BER $\uparrow$ & Acc\_Ben. $\uparrow$ & \#Unn. & \#Unn. Eval. & UER $\downarrow$ & Acc\_Unn. $\uparrow$ \\",
        r"\midrule",
    ]
    for row in benchmark_rows:
        lines.append(
            f"{latex_escape(str(row['benchmark']))} & "
            f"{int(row['n_beneficial'])} & "
            f"{int(row['n_beneficial_evaluable'])} & "
            f"{pct(float(row['ber_micro']))} & "
            f"{pct(float(row['acc_beneficial_micro']))} & "
            f"{int(row['n_unnecessary'])} & "
            f"{int(row['n_unnecessary_evaluable'])} & "
            f"{pct(float(row['uer_micro']))} & "
            f"{pct(float(row['acc_unnecessary_micro']))} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Selective procedural Skill exposure analysis of \textsc{SkillBrowser}. Skill-beneficial instances are defined separately for each model--benchmark pair as instances solved incorrectly without Skills but correctly with the gold Skill. Skill-unnecessary instances are solved correctly under both settings. BER measures exposure on beneficial instances, while UER measures unnecessary exposure on instances that do not require Skill content for correctness in this evaluation.}",
            r"\label{tab:selective-skill-exposure}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]]) -> str:
    header = "| Benchmark | #Ben. | #Ben. Eval. | BER | Acc_Ben. | #Unn. | #Unn. Eval. | UER | Acc_Unn. |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    body = [
        f"| {row['benchmark']} | {int(row['n_beneficial'])} | {int(row['n_beneficial_evaluable'])} | "
        f"{pct(float(row['ber_micro']))}% | {pct(float(row['acc_beneficial_micro']))}% | "
        f"{int(row['n_unnecessary'])} | {int(row['n_unnecessary_evaluable'])} | "
        f"{pct(float(row['uer_micro']))}% | {pct(float(row['acc_unnecessary_micro']))}% |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def write_summary(
    path: Path,
    *,
    sources: dict[tuple[str, str], SourcePaths],
    warnings: WarningLog,
    benchmark_rows: list[dict[str, Any]],
    overall: list[dict[str, Any]],
    plot_path: Path | None,
) -> None:
    lines = [
        "# Selective Procedural Skill Exposure Analysis",
        "",
        "This post-hoc analysis uses existing Direct/No-Skill, Oracle, and SkillBrowser per-instance outputs. No model inference was rerun.",
        "",
        "Exposure is derived from non-empty `skill_ids_used` or `meta.loaded_skill_ids` in SkillBrowser inference JSONL. For the direct solve engine, this means retrieved Skills were passed into the solver context.",
        "",
        "## Main Benchmark-Level Table",
        "",
        markdown_table(benchmark_rows),
        "",
        "## Overall Results",
        "",
    ]
    for row in overall:
        lines.append(
            f"- {row['aggregation']}: BER={pct(float(row['ber']))}%, "
            f"Acc_Ben.={pct(float(row['acc_beneficial']))}%, "
            f"UER={pct(float(row['uer']))}%, "
            f"Acc_Unn.={pct(float(row['acc_unnecessary']))}%"
        )
    lines.extend(["", "## Validation Status", ""])
    if warnings.messages:
        lines.append("Warnings were emitted and should be reviewed before direct paper inclusion:")
        for message in warnings.messages:
            lines.append(f"- {message}")
    else:
        lines.append("All alignment, category coverage, and exposure-field validation checks passed without warnings.")
    lines.extend(["", "## Sources", ""])
    for (model, benchmark), src in sorted(sources.items()):
        lines.append(f"- {model}/{benchmark}")
        lines.append(f"  - Direct: `{src.direct}`")
        lines.append(f"  - Oracle eval: `{src.oracle_eval}`")
        lines.append(f"  - SkillBrowser eval: `{src.skillbrowser_eval}`")
        lines.append(f"  - SkillBrowser inference: `{src.skillbrowser_inference}`")
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- High BER and low UER: SkillBrowser demonstrates well-calibrated selective procedural exposure.",
            "- High BER and high UER: SkillBrowser recognizes useful Skill needs but remains over-eager in exposing Skills when they are not necessary.",
            "- Low BER and low UER: SkillBrowser is conservative and avoids exposure, but under-invokes Skills on instances where external procedural support would help.",
            "- Low BER and high UER: The acquisition decision is poorly calibrated and should be presented as a limitation rather than a successful result.",
            "",
            "Exposure on Skill-Unnecessary instances is unnecessary for achieving correctness under this evaluation setting, not necessarily harmful.",
        ]
    )
    if plot_path:
        lines.extend(["", f"Scatter plot: `{plot_path}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, pair_rows: list[dict[str, Any]], warnings: WarningLog) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Could not import matplotlib; skipping scatter plot: {exc}")
        return None
    usable = [
        row for row in pair_rows
        if not math.isnan(float(row["ber"])) and not math.isnan(float(row["uer"]))
    ]
    if not usable:
        warnings.warn("No valid BER/UER pairs for scatter plot")
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter([row["uer"] for row in usable], [row["ber"] for row in usable], s=50)
    for row in usable:
        ax.annotate(
            f"{row['model']}:{row['benchmark']}",
            (row["uer"], row["ber"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("UER (lower is better)")
    ax.set_ylabel("BER (higher is better)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.set_title("Selective Procedural Skill Exposure")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def collect_records(args: argparse.Namespace, warnings: WarningLog) -> tuple[list[AnalysisRecord], dict[tuple[str, str], SourcePaths]]:
    records: list[AnalysisRecord] = []
    sources: dict[tuple[str, str], SourcePaths] = {}
    for model in args.models:
        for benchmark in args.benchmarks:
            direct, direct_source = load_direct_details(
                results_root=args.results_root,
                tarball=args.noskill_tarball,
                model=model,
                benchmark=benchmark,
                warnings=warnings,
            )
            oracle_path = args.results_root / "eval" / model / f"{benchmark}-oracle_direct.json"
            if not oracle_path.exists():
                warnings.warn(f"Missing Oracle eval for {model}/{benchmark}: {oracle_path}")
                continue
            skill_eval_path, skill_inf_path = resolve_skillbrowser_paths(
                args.results_root,
                model,
                benchmark,
                args.allow_fallback_methods,
                warnings,
                preferred_suffixes=args.skillbrowser_suffixes,
            )
            if not direct or skill_eval_path is None or skill_inf_path is None:
                continue
            oracle = load_eval_details(oracle_path, warnings, f"Oracle {model}/{benchmark}")
            skill_eval = load_eval_details(skill_eval_path, warnings, f"SkillBrowser eval {model}/{benchmark}")
            skill_inf = load_jsonl_records(skill_inf_path, warnings, f"SkillBrowser inference {model}/{benchmark}")
            pair_records = normalize_pair(
                model=model,
                benchmark=benchmark,
                direct=direct,
                oracle=oracle,
                skill_eval=skill_eval,
                skill_inf=skill_inf,
                warnings=warnings,
            )
            if not pair_records:
                warnings.warn(f"No aligned records for {model}/{benchmark}")
                continue
            records.extend(pair_records)
            sources[(model, benchmark)] = SourcePaths(
                direct=direct_source or str(args.noskill_tarball),
                oracle_eval=oracle_path,
                skillbrowser_eval=skill_eval_path,
                skillbrowser_inference=skill_inf_path,
            )
    return records, sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--noskill-tarball", type=Path, default=DEFAULT_NOSKILL_TARBALL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--benchmarks", nargs="+", default=list(DEFAULT_BENCHMARKS))
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=517)
    parser.add_argument("--allow-fallback-methods", action="store_true")
    parser.add_argument(
        "--skillbrowser-suffixes",
        nargs="+",
        help=(
            "Explicit SkillBrowser result suffix priority, e.g. "
            "skillbrowser_hybrid_top5_direct. Overrides default preferred agent suffixes."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    warnings = WarningLog()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, sources = collect_records(args, warnings)
    if not records:
        raise SystemExit("No aligned records found; see warnings above.")

    pair_rows = per_pair_rows(records, seed=args.seed, bootstrap_samples=args.bootstrap_samples)
    benchmark_rows = aggregate_by_benchmark(
        records, pair_rows, seed=args.seed, bootstrap_samples=args.bootstrap_samples
    )
    overall = overall_rows(records, pair_rows, seed=args.seed, bootstrap_samples=args.bootstrap_samples)

    write_csv(args.output_dir / "selective_exposure_by_model_benchmark.csv", pair_rows)
    write_csv(args.output_dir / "selective_exposure_by_benchmark.csv", benchmark_rows)
    write_csv(args.output_dir / "selective_exposure_overall.csv", overall)
    write_latex(args.output_dir / "selective_exposure_table.tex", benchmark_rows)
    plot_path = write_plot(args.output_dir / "selective_exposure_ber_uer_scatter.pdf", pair_rows, warnings)
    write_summary(
        args.output_dir / "selective_exposure_summary.md",
        sources=sources,
        warnings=warnings,
        benchmark_rows=benchmark_rows,
        overall=overall,
        plot_path=plot_path,
    )

    print(f"Wrote analysis outputs to {args.output_dir}")
    print(markdown_table(benchmark_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
