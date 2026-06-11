# CardioCare — same as README 3-command flow (command 1 + 2)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

pip install -r requirements.txt
python scripts/run_pipeline.py

Write-Host "Optional command 3: docker build -t cardiocare:1.0 . ; docker run --rm cardiocare:1.0"
