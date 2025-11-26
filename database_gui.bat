@echo off
chcp 65001 >nul
cd /d "%~dp0"
python scripts\database_gui.py
if errorlevel 1 (
    echo.
    echo Ошибка запуска. Нажмите любую клавишу для выхода...
    pause >nul
)

