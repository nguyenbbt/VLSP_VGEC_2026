[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [ValidateSet('L4', 'A100', 'A100-40GB', 'A100-80GB')][string]$Gpu = 'L4',
    [string]$Profile = ''
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
$env:VGEC_GPU = $Gpu
python -X utf8 scripts/preflight.py
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed. No training submitted.' }
if ($CheckOnly) { return }
$modalArgs = @('-X', 'utf8', '-m', 'modal', 'run', '--detach')
if ($Profile) { $modalArgs += @('--profile', $Profile) }
$modalArgs += 'modal_train.py::workflow'
python @modalArgs
if ($LASTEXITCODE -ne 0) { throw 'Modal launch failed. Inspect app status before resubmitting.' }
