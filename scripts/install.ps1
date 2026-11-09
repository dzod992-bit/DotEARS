param(
    [string]$PythonVersion = '3.11',
    [switch]$WithModels
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & py "-$PythonVersion" -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
}
& .\.venv\Scripts\python.exe -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("pip") else 1)'
if ($LASTEXITCODE -ne 0) {
    & .\.venv\Scripts\python.exe -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { throw 'Could not prepare pip in the project environment.' }
}
$extras = if ($WithModels) { '.[download,mcp,dev,asr,yamnet]' } else { '.[download,mcp,dev]' }
& .\.venv\Scripts\python.exe -m pip install -e $extras
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
& .\.venv\Scripts\dotears.exe doctor
Write-Host 'Run: .\.venv\Scripts\dotears.exe analyze C:\Media\reference.mp4 --profile motion'

