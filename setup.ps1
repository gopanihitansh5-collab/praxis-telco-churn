<#
    One-command setup and verification for Windows PowerShell.

    .\setup.ps1              Create the venv, install pinned deps, test, lint, predict
    .\setup.ps1 -Data        Also download the Kaggle dataset (needs Kaggle credentials)
    .\setup.ps1 -Reports     Also regenerate every report (implies -Data)
    .\setup.ps1 -SkipTests   Install only

    Finds a Python 3.10 or 3.11 interpreter automatically; the project pins that range
    because the saved model manifest refuses to load under a different scikit-learn.
#>
[CmdletBinding()]
param(
    [switch]$Data,
    [switch]$Reports,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Find-Python {
    # py launcher first: it is the reliable way to request an exact minor version.
    foreach ($v in @('3.11', '3.10')) {
        try {
            $probe = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $probe) { return $probe.Trim() }
        } catch {}
    }
    # uv-managed and common install locations.
    $globs = @(
        "$env:APPDATA\uv\python\cpython-3.11*\python.exe",
        "$env:APPDATA\uv\python\cpython-3.10*\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
    )
    foreach ($g in $globs) {
        $hit = Get-ChildItem -Path $g -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    # Last resort: whatever "python" is, but only if it is in the supported range.
    try {
        $ver = & python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $ver.Trim() -in @('3.10', '3.11')) {
            return (& python -c "import sys; print(sys.executable)").Trim()
        }
    } catch {}
    return $null
}

$python = Find-Python
if (-not $python) {
    Write-Host "No Python 3.10 or 3.11 found." -ForegroundColor Red
    Write-Host "This project pins that range: artifacts/model.manifest.json records"
    Write-Host "scikit-learn 1.7.2 and artifact.load_model fails closed on a mismatch,"
    Write-Host "so a newer Python with newer wheels cannot load the saved model."
    Write-Host ""
    Write-Host "Install one, then re-run:  winget install Python.Python.3.11"
    exit 1
}
Write-Host "Using interpreter: $python" -ForegroundColor Cyan

$venvPy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPy)) {
    Write-Host "Creating .venv ..." -ForegroundColor Cyan
    & $python -m venv .venv
}

Write-Host "Installing pinned dependencies ..." -ForegroundColor Cyan
& $venvPy -m pip install --upgrade pip --quiet
& $venvPy -m pip install -r requirements-dev.txt
& $venvPy -m pip install -e . --quiet
& $venvPy -m pip check

if ($Reports) { $Data = $true }

if ($Data) {
    if (Test-Path 'data\telco.csv') {
        Write-Host "data\telco.csv already present; skipping download." -ForegroundColor Yellow
    } else {
        Write-Host "Downloading dataset ..." -ForegroundColor Cyan
        & $venvPy download_data.py
    }
}

if (-not $SkipTests) {
    Write-Host "`nRunning tests ..." -ForegroundColor Cyan
    & $venvPy -m pytest -q
    Write-Host "`nLinting ..." -ForegroundColor Cyan
    & $venvPy -m ruff check src tests predict.py download_data.py
    & $venvPy -m ruff format --check src tests predict.py download_data.py
    Write-Host "`nSample prediction:" -ForegroundColor Cyan
    & $venvPy predict.py --input sample_customer.json
}

if ($Reports) {
    Write-Host "`nRegenerating reports (this takes a few minutes) ..." -ForegroundColor Cyan
    foreach ($m in @('calibrate', 'segments', 'cohort', 'policy')) {
        Write-Host "  telco_churn.$m" -ForegroundColor DarkGray
        & $venvPy -m "telco_churn.$m"
    }
}

Write-Host "`nReady." -ForegroundColor Green
Write-Host "  Predict:  .venv\Scripts\python.exe predict.py --input sample_customer.json"
Write-Host "  Serve:    .venv\Scripts\uvicorn.exe telco_churn.api:app --port 8000"
