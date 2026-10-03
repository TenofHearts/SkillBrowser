param(
    [int]$Workers = 5
)

$ErrorActionPreference = "Stop"

$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repo

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = (Resolve-Path "src").Path
$env:UV_CACHE_DIR = (Resolve-Path ".uv-cache").Path

$configText = Get-Content -Encoding UTF8 "config.toml"
$oldBase = ($configText | Select-String -Pattern '^\s*#\s*base_url\s*=\s*"https://dashscope\.aliyuncs\.com/compatible-mode/v1"\s*$' | Select-Object -First 1).Matches.Value
if (-not $oldBase) {
    throw "Could not find old DashScope base_url in config.toml comments."
}
$oldApiKeyLine = $configText | Select-String -Pattern '^\s*#\s*api_key\s*=\s*"([^"]+)"\s*$' | Select-Object -First 1
if (-not $oldApiKeyLine) {
    throw "Could not find old api_key in config.toml comments."
}

$apiBase = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:OPENAI_API_BASE = $apiBase
$env:OPENAI_API_KEY = $oldApiKeyLine.Matches[0].Groups[1].Value

$datasets = @("theoremqa", "logicbench", "champ", "medcalcbench")
$models = @("qwen3-8b", "qwen3.6-plus", "glm-5.1")
$corpus = "data/eval/sra/results/retrieval/llm_select_full_reduced_corpus.json"

foreach ($model in $models) {
    foreach ($dataset in $datasets) {
        $instances = "benchmarks/SR-Agents/data/bench/instances/$dataset.json"
        $retrieval = "data/eval/sra/results/retrieval/qwen3-8b/$dataset-sragents_bm25_top50_raw.json"
        $output = "data/eval/sra/results/inference/$model/$dataset-bm25_select-full.jsonl"

        Write-Host "[$(Get-Date -Format s)] $model $dataset -> $output"
        uv run --project benchmarks/SR-Agents sragents infer `
            --instances $instances `
            --output $output `
            --model $model `
            --provider llm_select `
            --provider-arg "source=$retrieval" `
            --provider-arg "pool=50" `
            --provider-arg "corpus_path=$corpus" `
            --engine direct `
            --api-base $apiBase `
            --workers $Workers `
            --temperature 0.0 `
            --max-tokens 4096 `
            --label bm25_select_first300
    }
}
