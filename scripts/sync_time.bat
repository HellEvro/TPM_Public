@echo off
REM Скрипт для запуска синхронизации времени Windows 11
REM Автоматически запрашивает права администратора

:: Проверка прав администратора
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Запуск синхронизации времени...
    echo.
    python "%~dp0sync_time.py"
) else (
    echo Запрос прав администратора...
    powershell -Command "Start-Process python -ArgumentList '%~dp0sync_time.py' -Verb RunAs"
)

pause
