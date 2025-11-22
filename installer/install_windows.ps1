Param(
    [switch]$ForceVenv,
    [switch]$ForceReinstall
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " InfoBot Installer - Windows" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

function Test-Command($Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

if (-not (Test-Command "python")) {
    Write-Error "Python 3.9+ not found. Install from https://www.python.org/downloads/windows/ and re-run this script."
}

if (-not (Test-Command "git")) {
    Write-Warning "Git not found. Install Git from https://git-scm.com/downloads to enable update checks."
}

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $projectRoot

Write-Host "Project root: $projectRoot"

$pythonVersion = (& python - <<'PYTHON'
import sys
print(".".join(map(str, sys.version_info[:3])))
PYTHON
)

if (-not $pythonVersion) {
    Write-Error "Unable to determine Python version."
}

$majorMinor = $pythonVersion.Split(".")[0..1] -join "."
if ([version]$majorMinor -lt [version]"3.9") {
    Write-Error "Python $pythonVersion detected. Python 3.9 or newer is required."
}

$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if ((-not (Test-Path $venvPython)) -or $ForceVenv) {
    if (Test-Path $venvPath) {
        Write-Host "Removing existing virtual environment (.venv)..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venvPath
    }
    Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Green
    & python -m venv ".venv"
}

if (-not (Test-Path $venvPython)) {
    Write-Error "Virtual environment creation failed. Check Python installation."
}

Write-Host "Upgrading pip/setuptools/wheel..." -ForegroundColor Green
& $venvPython -m pip install --upgrade pip setuptools wheel

if ($ForceReinstall) {
    Write-Host "Forcing reinstallation of dependencies..." -ForegroundColor Yellow
    & $venvPython -m pip install --force-reinstall -r requirements.txt
} else {
    Write-Host "Installing/updating project dependencies..." -ForegroundColor Green
    & $venvPython -m pip install -r requirements.txt
}

$managerCmd = Join-Path $projectRoot "start_infobot_manager.cmd"
if (-not (Test-Path $managerCmd)) {
    "@echo off
setlocal enabledelayedexpansion
cd /d %~dp0
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo [WARN] Virtual environment not found. Falling back to system python.
)
python launcher\infobot_manager.py %*
endlocal
" | Out-File -Encoding ascii $managerCmd
}

Write-Host ""
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run the manager: start_infobot_manager.cmd"
Write-Host "  2. Use the GUI to launch app.py, bots.py and ai.py"
Write-Host "  3. Check updates via the GUI (Git required)"
Write-Host ""
*** End Patch

