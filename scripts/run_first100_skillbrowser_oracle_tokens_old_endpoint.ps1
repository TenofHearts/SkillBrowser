param(
    [string[]]$Models = @("qwen3-8b", "qwen3.6-plus", "glm-5.1"),
    [string[]]$Datasets = @("theoremqa", "logicbench", "champ", "medcalcbench"),
    [switch]$Resume,
    [switch]$UseCurrentConfig,
    [int]$Workers = 5,
    [int]$EvalWorkers = 5
)

$ErrorActionPreference = "Stop"

$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repo

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = (Resolve-Path "src").Path
$env:UV_CACHE_DIR = (Resolve-Path ".uv-cache").Path

function Invoke-Checked {
    param([scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

$tmpConfig = "config.toml"
$apiBaseArgs = @()
if (-not $UseCurrentConfig) {
    $configText = Get-Content -Encoding UTF8 "config.toml"
    $oldApiKeyLine = $configText | Select-String -Pattern '^\s*#\s*api_key\s*=\s*"([^"]+)"\s*$' | Select-Object -First 1
    if (-not $oldApiKeyLine) {
        throw "Could not find old api_key in config.toml comments."
    }

    $tmpDir = "data/eval/sra/results/tmp"
    New-Item -ItemType Directory -Force $tmpDir | Out-Null
    $tmpConfig = Join-Path $tmpDir "old_endpoint_config.toml"

    $rewritten = foreach ($line in $configText) {
        if ($line -match '^\s*base_url\s*=') {
            'base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"'
        } elseif ($line -match '^\s*api_key\s*=') {
            "api_key = `"$($oldApiKeyLine.Matches[0].Groups[1].Value)`""
        } else {
            $line
        }
    }
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText((Resolve-Path $tmpDir).Path + "\old_endpoint_config.toml", ($rewritten -join [Environment]::NewLine), $utf8NoBom)
    $apiBaseArgs = @("--api-base", "https://dashscope.aliyuncs.com/compatible-mode/v1")
}

foreach ($model in $Models) {
    foreach ($dataset in $Datasets) {
        $instances = "data/eval/sra/instances/first100/$dataset-first100.json"
        $infDir = "data/eval/sra/results/inference/$model"
        $evalDir = "data/eval/sra/results/eval/$model"
        $summaryDir = "data/eval/sra/results/eval/$model"
        New-Item -ItemType Directory -Force $infDir | Out-Null
        New-Item -ItemType Directory -Force $evalDir | Out-Null

        $sbInf = "$infDir/$dataset-skillbrowser_agent_top5_direct-token-rerun-first100.jsonl"
        $sbEval = "$evalDir/$dataset-skillbrowser_agent_top5_direct-token-rerun-first100.json"
        $sbSummary = "$summaryDir/$dataset-skillbrowser_agent_top5_direct-token-rerun-first100-summary.json"

        Write-Host "[$(Get-Date -Format s)] SkillBrowser $model $dataset"
        $sbArgs = @(
            "run", "python", "scripts/sra_bench.py", "infer-decision-agent",
            "--dataset", $dataset,
            "--instances", $instances,
            "--inference-output", $sbInf,
            "--model", $model,
            "--config", $tmpConfig,
            "--agent-top-k", "5",
            "--solve-engine", "direct",
            "--workers", "$Workers",
            "--temperature", "0.0",
            "--max-tokens", "4096"
        )
        $sbArgs += $apiBaseArgs
        if (-not $Resume) {
            $sbArgs += "--force"
        }
        Invoke-Checked {
            uv @sbArgs
        }
        <#
                --dataset $dataset `
                --instances $instances `
                --inference-output $sbInf `
                --model $model `
                --config $tmpConfig `
                --api-base "https://dashscope.aliyuncs.com/compatible-mode/v1" `
                --agent-top-k 5 `
                --solve-engine direct `
                --workers $Workers `
                --temperature 0.0 `
                --max-tokens 4096 `
                --force
        #>

        Invoke-Checked {
            uv run python scripts/sra_bench.py evaluate `
                --dataset $dataset `
                --instances $instances `
                --inference-output $sbInf `
                --eval-output $sbEval `
                --eval-workers $EvalWorkers `
                --eval-force
        }

        Invoke-Checked {
            uv run python scripts/sra_bench.py summarize-agent `
                --dataset $dataset `
                --instances $instances `
                --inference-output $sbInf `
                --eval-output $sbEval `
                --summary-output $sbSummary
        }

        $oracleInf = "$infDir/$dataset-oracle_direct-token-rerun-first100.jsonl"
        $oracleEval = "$evalDir/$dataset-oracle_direct-token-rerun-first100.json"

        Write-Host "[$(Get-Date -Format s)] Oracle $model $dataset"
        $oracleArgs = @(
            "run", "python", "scripts/sra_bench.py", "infer",
            "--dataset", $dataset,
            "--instances", $instances,
            "--inference-output", $oracleInf,
            "--provider", "oracle",
            "--engine", "direct",
            "--model", $model,
            "--config", $tmpConfig,
            "--workers", "$Workers",
            "--temperature", "0.0",
            "--max-tokens", "4096"
        )
        $oracleArgs += $apiBaseArgs
        if (-not $Resume) {
            $oracleArgs += "--force"
        }
        Invoke-Checked {
            uv @oracleArgs
        }
        <#
                --dataset $dataset `
                --instances $instances `
                --inference-output $oracleInf `
                --provider oracle `
                --engine direct `
                --model $model `
                --config $tmpConfig `
                --api-base "https://dashscope.aliyuncs.com/compatible-mode/v1" `
                --workers $Workers `
                --temperature 0.0 `
                --max-tokens 4096 `
                --force
        #>

        Invoke-Checked {
            uv run python scripts/sra_bench.py evaluate `
                --dataset $dataset `
                --instances $instances `
                --inference-output $oracleInf `
                --eval-output $oracleEval `
                --eval-workers $EvalWorkers `
                --eval-force
        }
    }
}
