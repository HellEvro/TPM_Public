@echo off
REM Скрипт для установки зависимостей в .venv_gpu
echo ================================================================================
echo УСТАНОВКА ЗАВИСИМОСТЕЙ В .venv_gpu
echo ================================================================================
echo.

if not exist ".venv_gpu\Scripts\python.exe" (
    echo [ERROR] .venv_gpu не найден!
    echo [INFO] Запустите сначала: py scripts/setup_python_gpu.py
    echo [INFO] Или: python3 scripts/setup_python_gpu.py
    pause
    exit /b 1
)

echo [INFO] Используется Python из .venv_gpu
echo [INFO] Установка зависимостей из requirements.txt...
.venv_gpu\Scripts\python.exe -m pip install --upgrade pip
.venv_gpu\Scripts\python.exe -m pip install -r requirements.txt

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [OK] Зависимости установлены успешно!
) else (
    echo.
    echo [ERROR] Ошибка при установке зависимостей
)

pause
