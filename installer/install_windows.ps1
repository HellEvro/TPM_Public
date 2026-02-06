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

if (-not (Test-Command "python") -and -not (Test-Command "py")) {
    Write-Error "Python 3.14 not found. Install from https://www.python.org/downloads/ or run: winget install Python.Python.3.14"
}

if (-not (Test-Command "git")) {
    Write-Warning "Git not found. Install Git from https://git-scm.com/downloads to enable update checks."
}

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $projectRoot

Write-Host "Project root: $projectRoot"

$pyCmd = $null
# Требуем Python 3.14 или выше
try {
    $null = & py -3.14 -c "import sys; exit(0 if sys.version_info[:2] >= (3,14) else 1)" 2>$null
    if ($LASTEXITCODE -eq 0) { $pyCmd = @('py','-3.14') }
} catch {}
if (-not $pyCmd) {
    try {
        $null = & python3.14 -c "import sys; exit(0 if sys.version_info[:2] >= (3,14) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { $pyCmd = @('python3.14') }
    } catch {}
}
if (-not $pyCmd) {
    try {
        $null = & python -c "import sys; exit(0 if sys.version_info[:2] >= (3,14) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { $pyCmd = @('python') }
    } catch {}
}
if (-not $pyCmd) {
    Write-Error "Python 3.14+ required. Install: https://www.python.org/downloads/ or: winget install Python.Python.3.14"
}
$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if ((-not (Test-Path $venvPython)) -or $ForceVenv) {
    if (Test-Path $venvPath) {
        Write-Host "Removing existing virtual environment (.venv)..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venvPath
    }
    Write-Host "Creating virtual environment (.venv) with Python 3.14..." -ForegroundColor Green
    $venvArgs = @($pyCmd[1..($pyCmd.Length-1)]) + @('-m','venv','.venv')
    & $pyCmd[0] $venvArgs
}

if (-not (Test-Path $venvPython)) {
    Write-Error "Virtual environment creation failed. Check Python installation."
}

Write-Host "Upgrading pip/setuptools/wheel..." -ForegroundColor Green
& $venvPython -m pip install --upgrade pip setuptools wheel --no-warn-script-location

if ($ForceReinstall) {
    Write-Host "Forcing reinstallation of dependencies..." -ForegroundColor Yellow
    & $venvPython -m pip install --force-reinstall -r requirements.txt --no-warn-script-location
} else {
    Write-Host "Installing/updating project dependencies..." -ForegroundColor Green
    & $venvPython -m pip install -r requirements.txt --no-warn-script-location
}

Write-Host "Compiling protected modules (.pyc files)..." -ForegroundColor Green
try {
    & $venvPython license_generator\compile_all.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] All protected modules compiled successfully!" -ForegroundColor Green
    } else {
        Write-Warning "Some modules failed to compile. This may be normal if source files are not available."
    }
} catch {
    Write-Warning "Failed to compile protected modules: $_"
}

$managerCmd = Join-Path $projectRoot "start_infobot_manager.bat"
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
Write-Host "  1. Run the manager: start_infobot_manager.bat"
Write-Host "  2. Use the GUI to launch app.py, bots.py and ai.py"
Write-Host "  3. Check updates via the GUI (Git required)"
Write-Host ""
