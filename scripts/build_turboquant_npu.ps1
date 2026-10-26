$ErrorActionPreference = 'Stop'
$scriptPath = Join-Path $PSScriptRoot 'build_turboquant_npu.sh'
$portableScript = $scriptPath.Replace('\', '/')
$wslScript = (wsl -- wslpath -a -u $portableScript).Trim()
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
wsl -- bash $wslScript
exit $LASTEXITCODE
