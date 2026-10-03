param(
    [string[]]$Models = @("qwen3-8b", "qwen3.6-plus", "glm-5.1", "gpt-5.4-mini", "gemini-3.1-flash-lite"),
    [string[]]$Datasets = @("theoremqa", "logicbench", "champ", "medcalcbench"),
    [int]$Workers = 5
)

$ErrorActionPreference = "Stop"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = (Resolve-Path "src").Path
$env:UV_CACHE_DIR = (Resolve-Path ".uv-cache").Path

$configLines = Get-Content -Path "config.toml" -Encoding UTF8
$currentBase = ($configLines | Where-Object { $_ -match '^\s*base_url\s*=\s*"([^"]+)"' } | Select-Object -First 1) -replace '^\s*base_url\s*=\s*"([^"]+)".*$', '$1'
$currentKey = ($configLines | Where-Object { $_ -match '^\s*api_key\s*=\s*"([^"]+)"' } | Select-Object -First 1) -replace '^\s*api_key\s*=\s*"([^"]+)".*$', '$1'
$aliyunBase = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$aliyunKey = ($configLines | Where-Object { $_ -match '^\s*#\s*api_key\s*=\s*"([^"]+)"' } | Select-Object -First 1) -replace '^\s*#\s*api_key\s*=\s*"([^"]+)".*$', '$1'

$aliyunModels = @("qwen3-8b", "qwen3.6-plus", "glm-5.1")
$reducedCorpus = "data/eval/sra/results/retrieval/timing_first10_reduced_corpus.json"

function Invoke-Checked {
    param([string[]]$UvArgs)
    & uv @UvArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: uv $($UvArgs -join ' ')"
    }
}

foreach ($model in $Models) {
    if ($aliyunModels -contains $model) {
        $apiBase = $aliyunBase
        $env:OPENAI_API_KEY = $aliyunKey
    } else {
        $apiBase = $currentBase
        $env:OPENAI_API_KEY = $currentKey
    }
    $env:OPENAI_API_BASE = $apiBase

    $modelDir = "data/eval/sra/results/inference/$model"
    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

    foreach ($dataset in $Datasets) {
        $instances = "data/eval/sra/instances/first10/$dataset-first10.json"
        $retrieval = "data/eval/sra/results/retrieval/qwen3-8b/$dataset-sragents_bm25_top50_raw.json"

        $oracleOut = "$modelDir/$dataset-oracle_direct-timing-first10.jsonl"
        if (Test-Path $oracleOut) { Remove-Item -LiteralPath $oracleOut }
        Write-Host "==> $model / $dataset / oracle_direct"
        Invoke-Checked -UvArgs @(
            "run", "--project", "benchmarks/SR-Agents", "sragents", "infer",
            "--instances", $instances,
            "--output", $oracleOut,
            "--model", $model,
            "--provider", "oracle",
            "--provider-arg", "corpus_path=$reducedCorpus",
            "--engine", "direct",
            "--api-base", $apiBase,
            "--workers", "$Workers",
            "--temperature", "0.0",
            "--max-tokens", "4096",
            "--label", "oracle_direct_timing_first10"
        )

        $selectOut = "$modelDir/$dataset-bm25_select-timing-first10.jsonl"
        if (Test-Path $selectOut) { Remove-Item -LiteralPath $selectOut }
        Write-Host "==> $model / $dataset / bm25_select"
        Invoke-Checked -UvArgs @(
            "run", "--project", "benchmarks/SR-Agents", "sragents", "infer",
            "--instances", $instances,
            "--output", $selectOut,
            "--model", $model,
            "--provider", "llm_select",
            "--provider-arg", "source=$retrieval",
            "--provider-arg", "pool=50",
            "--provider-arg", "corpus_path=$reducedCorpus",
            "--engine", "direct",
            "--api-base", $apiBase,
            "--workers", "$Workers",
            "--temperature", "0.0",
            "--max-tokens", "4096",
            "--label", "bm25_select_timing_first10"
        )
    }
}
