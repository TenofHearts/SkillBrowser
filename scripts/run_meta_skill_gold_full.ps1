$Group = if ($args.Count) { $args[0] } else { "all" }
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = (Resolve-Path src).Path
$env:UV_CACHE_DIR = (Resolve-Path .uv-cache).Path

$configText = Get-Content -LiteralPath config.toml -Raw
$keys = [regex]::Matches($configText, '(?m)^\s*#?\s*api_key\s*=\s*"([^"]+)"')
if ($keys.Count -lt 2) {
    throw "Expected both Aliyun and Laozhang API keys in config.toml"
}
$env:ALIYUN_API_KEY = $keys[0].Groups[1].Value
$env:LAOZHANG_API_KEY = $keys[$keys.Count - 1].Groups[1].Value

$runs = @(
    @{ Model = "qwen3-8b"; Dataset = "theoremqa"; ApiBase = "https://dashscope.aliyuncs.com/compatible-mode/v1"; KeyEnv = "ALIYUN_API_KEY" },
    @{ Model = "glm-5.1"; Dataset = "theoremqa"; ApiBase = "https://dashscope.aliyuncs.com/compatible-mode/v1"; KeyEnv = "ALIYUN_API_KEY" },
    @{ Model = "glm-5.1"; Dataset = "medcalcbench"; ApiBase = "https://dashscope.aliyuncs.com/compatible-mode/v1"; KeyEnv = "ALIYUN_API_KEY" },
    @{ Model = "gpt-5.4-mini"; Dataset = "theoremqa"; ApiBase = "https://api.laozhang.ai/v1"; KeyEnv = "LAOZHANG_API_KEY" },
    @{ Model = "gpt-5.4-mini"; Dataset = "champ"; ApiBase = "https://api.laozhang.ai/v1"; KeyEnv = "LAOZHANG_API_KEY" },
    @{ Model = "gemini-3.1-flash-lite"; Dataset = "champ"; ApiBase = "https://api.laozhang.ai/v1"; KeyEnv = "LAOZHANG_API_KEY" }
)

if ($Group -eq "aliyun-remaining") {
    $runs = @($runs | Where-Object { $_.KeyEnv -eq "ALIYUN_API_KEY" -and $_.Model -ne "qwen3-8b" })
} elseif ($Group -eq "aliyun-medcalc") {
    $runs = @($runs | Where-Object { $_.Model -eq "glm-5.1" -and $_.Dataset -eq "medcalcbench" })
} elseif ($Group -eq "aliyun-theorem") {
    $runs = @($runs | Where-Object { $_.Model -eq "glm-5.1" -and $_.Dataset -eq "theoremqa" })
} elseif ($Group -eq "laozhang") {
    $runs = @($runs | Where-Object { $_.KeyEnv -eq "LAOZHANG_API_KEY" })
} elseif ($Group -eq "laozhang-remaining") {
    $runs = @($runs | Where-Object { $_.KeyEnv -eq "LAOZHANG_API_KEY" -and $_.Dataset -eq "champ" })
} elseif ($Group -ne "all") {
    throw "Unknown run group: $Group"
}

foreach ($run in $runs) {
    $inference = "data/eval/sra/results/inference/$($run.Model)/$($run.Dataset)-meta_skill_gold_direct.jsonl"
    $evaluation = "data/eval/sra/results/eval/$($run.Model)/$($run.Dataset)-meta_skill_gold_direct.json"
    Write-Host "RUN model=$($run.Model) dataset=$($run.Dataset)"
    uv run python scripts/sra_bench.py run-decision-agent `
        --dataset $run.Dataset `
        --model $run.Model `
        --api-base $run.ApiBase `
        --api-key-env $run.KeyEnv `
        --exposure-mode gold_on_search `
        --agent-top-k 5 `
        --solve-engine direct `
        --workers 5 `
        --eval-workers 5 `
        --inference-output $inference `
        --eval-output $evaluation `
        --eval-force
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed: model=$($run.Model) dataset=$($run.Dataset)"
    }
}
