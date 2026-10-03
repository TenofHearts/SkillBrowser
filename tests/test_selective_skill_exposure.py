from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_selective_skill_exposure import (  # noqa: E402
    AnalysisRecord,
    aggregate_by_benchmark,
    category_for,
    derive_exposure,
    normalize_pair,
    per_pair_rows,
    summarize_group,
    WarningLog,
)


def _record(
    model: str,
    benchmark: str,
    instance_id: str,
    direct: bool,
    oracle: bool,
    skillbrowser: bool,
    exposed: bool,
) -> AnalysisRecord:
    return AnalysisRecord(
        model=model,
        benchmark=benchmark,
        instance_id=instance_id,
        direct_correct=direct,
        oracle_correct=oracle,
        skillbrowser_correct=skillbrowser,
        skillbrowser_search_invoked=exposed,
        skillbrowser_skill_exposed=exposed,
        skillbrowser_exposed_skill_count=1 if exposed else 0,
    )


def test_derive_exposure_uses_skill_ids_or_loaded_meta() -> None:
    search, exposed, loaded, count = derive_exposure(
        {"meta": {"search_call_count": 1, "loaded_skill_ids": ["s1"]}}
    )

    assert search is True
    assert exposed is True
    assert loaded == ["s1"]
    assert count == 1


def test_categories_are_direct_oracle_based() -> None:
    assert category_for(_record("m", "b", "1", False, True, True, True)) == "beneficial"
    assert category_for(_record("m", "b", "2", True, True, True, False)) == "unnecessary"
    assert category_for(_record("m", "b", "3", True, False, False, False)) == "harmful"
    assert category_for(_record("m", "b", "4", False, False, False, False)) == "unresolved"


def test_summarize_group_nan_for_empty_denominators() -> None:
    summary = summarize_group(
        [_record("m", "b", "1", True, False, False, False)],
        seed=1,
        bootstrap_samples=10,
    )

    assert summary["n_beneficial"] == 0
    assert math.isnan(summary["ber"])
    assert summary["n_unnecessary"] == 0
    assert math.isnan(summary["uer"])


def test_grouping_is_per_model_benchmark_and_macro_differs_from_micro() -> None:
    records = [
        _record("m1", "bench", "1", False, True, True, True),
        _record("m1", "bench", "2", False, True, False, False),
        _record("m2", "bench", "1", False, True, True, True),
        _record("m2", "bench", "2", False, True, True, True),
        _record("m2", "bench", "3", False, True, True, True),
    ]

    pairs = per_pair_rows(records, seed=1, bootstrap_samples=10)
    bench = aggregate_by_benchmark(records, pairs, seed=1, bootstrap_samples=10)[0]

    assert len(pairs) == 2
    assert bench["ber_micro"] == 4 / 5
    assert bench["ber_macro"] == (0.5 + 1.0) / 2


def test_normalize_pair_aligns_on_common_instance_ids() -> None:
    warnings = WarningLog()
    records = normalize_pair(
        model="m",
        benchmark="b",
        direct={"1": {"correct": False}, "2": {"correct": True}},
        oracle={"1": {"correct": True, "skill_annotations": ["gold"]}},
        skill_eval={"1": {"correct": True}},
        skill_inf={"1": {"skill_ids_used": ["loaded"], "meta": {"search_call_count": 1}}},
        warnings=warnings,
    )

    assert len(records) == 1
    assert records[0].instance_id == "1"
    assert records[0].gold_skill_names == ["gold"]
    assert records[0].skillbrowser_skill_exposed is True
    assert warnings.messages


def test_normalize_pair_keeps_direct_oracle_instances_without_skillbrowser() -> None:
    warnings = WarningLog()
    records = normalize_pair(
        model="m",
        benchmark="b",
        direct={"1": {"correct": False}, "2": {"correct": True}},
        oracle={
            "1": {"correct": True, "skill_annotations": ["gold1"]},
            "2": {"correct": True, "skill_annotations": ["gold2"]},
        },
        skill_eval={"1": {"correct": True}},
        skill_inf={"1": {"skill_ids_used": ["loaded"], "meta": {"search_call_count": 1}}},
        warnings=warnings,
    )
    summary = summarize_group(records, seed=1, bootstrap_samples=10)

    assert len(records) == 2
    assert summary["n_beneficial"] == 1
    assert summary["n_beneficial_evaluable"] == 1
    assert summary["n_unnecessary"] == 1
    assert summary["n_unnecessary_evaluable"] == 0
    assert math.isnan(summary["uer"])
