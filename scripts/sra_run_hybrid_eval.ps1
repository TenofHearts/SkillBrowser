param(
    [string]$Dataset = "theoremqa",
    [string]$Model,
    [string]$ApiBase = "",
    [int]$TopK = 50,
    [int]$ProviderK = 1,
    [string]$Engine = "direct",
    [int]$Workers = 8,
    [string]$Config = "config.toml"
)

if (-not $Model) {
    throw "Pass -Model with the served model name, for example -Model gpt-4o-mini"
}

$args = @(
    "run",
    "--dataset", $Dataset,
    "--top-k", $TopK,
    "--provider-k", $ProviderK,
    "--engine", $Engine,
    "--model", $Model,
    "--workers", $Workers,
    "--config", $Config
)

if ($ApiBase) {
    $args += @("--api-base", $ApiBase)
}

uv run python scripts/sra_bench.py @args
