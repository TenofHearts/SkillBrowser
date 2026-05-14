from benchmarks.sra import compute_sra_retrieval_metrics, sra_skill_to_spec


def test_sra_skill_to_spec_preserves_benchmark_id():
    spec = sra_skill_to_spec(
        {
            "skill_id": "theoremqa_001",
            "name": "Apply Bayes rule",
            "description": "Use Bayes rule for conditional probability.",
            "content": "P(A|B) = P(B|A)P(A)/P(B).",
        }
    )

    assert spec.id == "theoremqa_001"
    assert spec.category.primary == "theoremqa"
    assert "Bayes" in spec.description.long


def test_compute_sra_retrieval_metrics_handles_multi_gold():
    records = [
        {
            "instance_id": "q1",
            "gold_skill_ids": ["a", "b"],
            "retrieved": [{"skill_id": "x", "score": 2.0}, {"skill_id": "a", "score": 1.0}],
        },
        {
            "instance_id": "q2",
            "gold_skill_ids": ["c"],
            "retrieved": [{"skill_id": "c", "score": 3.0}],
        },
    ]

    metrics = compute_sra_retrieval_metrics(records, top_k=5)

    assert metrics["Recall@1"] == 0.5
    assert metrics["Recall@5"] == 0.75
    assert 0.0 < metrics["nDCG@5"] <= 1.0
