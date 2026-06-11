# CardioCare full reproduction script (Windows PowerShell)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "Installing dependencies..."
pip install -r requirements.txt

Write-Host "Downloading data..."
python data/download_data.py

Write-Host "Training models..."
python src/train.py

Write-Host "Running monitor..."
python src/monitor.py

Write-Host "Running inference..."
python src/inference.py --input data/sample_batch.csv

Write-Host "Running tests..."
python -m unittest discover -s tests -v

Write-Host "Building report..."
python reports/build_report_pdf.py

Write-Host "Done. Artifacts: artifacts/, report.pdf, mlruns/"
