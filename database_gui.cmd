@echo off
REM Скрипт запуска GUI для работы с базами данных (Windows)

cd /d "%~dp0"

REM Проверяем наличие виртуального окружения
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe scripts\database_gui.py
) else (
    python scripts\database_gui.py
)

pause

