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

function Invoke-Step {
    # 'Stop' does not trap nonzero exit codes from native executables in Windows
    # PowerShell, so every python/pip/pytest/ruff call runs through here instead.
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )
    & $Command
    # Only meaningful because every scriptblock below ends in a native command.
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "FAILED: $Label (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

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
    Invoke-Step 'create .venv' { & $python -m venv .venv }
}

Write-Host "Installing pinned dependencies ..." -ForegroundColor Cyan
Invoke-Step 'pip install --upgrade pip' { & $venvPy -m pip install --upgrade pip --quiet }
Invoke-Step 'pip install -r requirements-dev.txt' { & $venvPy -m pip install -r requirements-dev.txt }
Invoke-Step 'pip install -e .' { & $venvPy -m pip install -e . --quiet }
Invoke-Step 'pip check' { & $venvPy -m pip check }

if ($Reports) { $Data = $true }

if ($Data) {
    if (Test-Path 'data\telco.csv') {
        Write-Host "data\telco.csv already present; skipping download." -ForegroundColor Yellow
    } else {
        Write-Host "Downloading dataset ..." -ForegroundColor Cyan
        Invoke-Step 'download_data.py' { & $venvPy download_data.py }
    }
}

if (-not $SkipTests) {
    Write-Host "`nRunning tests ..." -ForegroundColor Cyan
    Invoke-Step 'pytest' { & $venvPy -m pytest -q }
    Write-Host "`nLinting ..." -ForegroundColor Cyan
    Invoke-Step 'ruff check' { & $venvPy -m ruff check src tests predict.py download_data.py }
    Invoke-Step 'ruff format --check' { & $venvPy -m ruff format --check src tests predict.py download_data.py }
    Write-Host "`nSample prediction:" -ForegroundColor Cyan
    Invoke-Step 'predict.py' { & $venvPy predict.py --input sample_customer.json }
}

if ($Reports) {
    Write-Host "`nRegenerating reports (this takes a few minutes) ..." -ForegroundColor Cyan
    # policy consumes the out-of-fold file calibrate writes, so order matters here.
    Invoke-Step 'telco_churn.calibrate' { & $venvPy -m telco_churn.calibrate }
    Invoke-Step 'telco_churn.segments' { & $venvPy -m telco_churn.segments }
    Invoke-Step 'telco_churn.cohort' { & $venvPy -m telco_churn.cohort }
    Invoke-Step 'telco_churn.policy' { & $venvPy -m telco_churn.policy --probabilities artifacts/oof_probabilities.csv }
    Invoke-Step 'telco_churn.score_batch' { & $venvPy -m telco_churn.score_batch }
}

Write-Host "`nReady." -ForegroundColor Green
Write-Host "  Predict:  .venv\Scripts\python.exe predict.py --input sample_customer.json"
Write-Host "  Serve:    .venv\Scripts\uvicorn.exe telco_churn.api:app --port 8000"
