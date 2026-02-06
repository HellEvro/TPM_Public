@echo off
REM Скрипт для запуска постоянной синхронизации времени Windows 11
REM Автоматически запрашивает права администратора

set INTERVAL=60
if not "%1"=="" set INTERVAL=%1

echo ========================================
echo Постоянная синхронизация времени Windows 11
echo Интервал: %INTERVAL% минут
echo ========================================
echo.
echo Для остановки нажмите Ctrl+C
echo.

:: Проверка прав администратора
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Запуск в режиме постоянной работы...
    echo.
    python "%~dp0sync_time.py" --daemon --interval %INTERVAL% --log "%~dp0sync_time.log"
) else (
    echo Запрос прав администратора...
    powershell -Command "Start-Process python -ArgumentList '%~dp0sync_time.py --daemon --interval %INTERVAL% --log %~dp0sync_time.log' -Verb RunAs"
)

pause
