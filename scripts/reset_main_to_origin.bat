@echo off
chcp 65001 >nul
setlocal
if "%~1"=="" (
  for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
) else (
  set "REPO_ROOT=%~1"
)
cd /d "%REPO_ROOT%"
if errorlevel 1 (echo ERROR: dir not found: %REPO_ROOT% & exit /b 1)

echo Repo: %REPO_ROOT%
echo.
echo Fetching origin...
git fetch origin
if errorlevel 1 (echo ERROR: git fetch failed & exit /b 1)

echo Resetting main to origin/main (hard)...
git reset --hard origin/main
if errorlevel 1 (echo ERROR: git reset failed & exit /b 1)

echo.
echo main = origin/main
git log --oneline -1
echo.
echo Cleaning reflog and pruning...
git reflog expire --expire=now --all
git gc --prune=now
echo Done.
endlocal
