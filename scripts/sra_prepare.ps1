param(
    [string]$Corpus = "benchmarks/SR-Agents/data/bench/corpus/corpus.json"
)

uv run python scripts/sra_bench.py prepare --corpus $Corpus
