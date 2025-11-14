@echo off
setlocal enabledelayedexpansion
cd /d %~dp0
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    set "PYTHON_BIN=python"
) else (
    echo [WARN] Virtual environment not found. Falling back to system Python.
    if exist %SystemRoot%\py.exe (
        set "PYTHON_BIN=py -3"
    ) else (
        set "PYTHON_BIN=python"
    )
)
%PYTHON_BIN% launcher\infobot_manager.py %*
endlocal

