param(
    [string]$Dataset = "theoremqa",
    [int]$TopK = 50,
    [int]$Limit = 0,
    [string]$Config = "config.toml"
)

$args = @("retrieve", "--dataset", $Dataset, "--top-k", $TopK, "--config", $Config)
if ($Limit -gt 0) {
    $args += @("--limit", $Limit)
}

uv run python scripts/sra_bench.py @args
