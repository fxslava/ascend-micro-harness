param([string]$Python = 'python')

$ErrorActionPreference = 'Stop'
$WorkspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$Runner = Join-Path $PSScriptRoot 'tq_cdf_vs_lloydmax_ablation.py'
$ReportRoot = Join-Path $WorkspaceRoot 'reports\tq_cdf_vs_lloydmax\extended'
New-Item -ItemType Directory -Force -Path $ReportRoot | Out-Null

# Run sources sequentially so the benchmark processes do not compete for CPUs.
& $Python -m unittest discover -s $PSScriptRoot -p 'test_tq_cdf_vs_lloydmax_ablation.py' -v 2>&1 |
    Tee-Object -FilePath (Join-Path $ReportRoot 'validation.txt')
if ($LASTEXITCODE -ne 0) { throw 'Quantizer verification failed' }

foreach ($Source in @('synthetic', 'niah')) {
    & $Python $Runner --source $Source --seeds '11,23,37,53,71' --bits '3,4,5' `
        --kappas '1,8,64' --vectors 8192 --context-lengths '8192,32768,131072' `
        --group-size 0 --chunk-tokens 8192 --sample-slots 4096 --threads 1 `
        --output-dir (Join-Path $ReportRoot $Source) | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "$Source ablation failed" }
}

& $Python (Join-Path $PSScriptRoot 'summarize_tq_cdf_extension.py') --report-root $ReportRoot
if ($LASTEXITCODE -ne 0) { throw 'Extended summary or baseline preservation audit failed' }

Write-Output "Reports: $ReportRoot"
