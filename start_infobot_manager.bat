@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
cd /d %~dp0

REM Устанавливаем кодировку для Python
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM InfoBot требует Python 3.14.2+ или выше
set "PYTHON_FOUND=0"
set "PYTHON_CMD="

REM Проверяем py -3.14 (приоритет на Windows)
py -3.14 --version >nul 2>&1
if !errorlevel!==0 (
    set "PYTHON_FOUND=1"
    set "PYTHON_CMD=py -3.14"
)

REM Проверяем python3.14
if !PYTHON_FOUND!==0 (
    python3.14 --version >nul 2>&1
    if !errorlevel!==0 (
        set "PYTHON_FOUND=1"
        set "PYTHON_CMD=python3.14"
    )
)

REM Проверяем python (должна быть 3.14 или выше)
if !PYTHON_FOUND!==0 (
    python -c "import sys; exit(0 if sys.version_info[:2]>=(3,14) else 1)" >nul 2>&1
    if !errorlevel!==0 (
        set "PYTHON_FOUND=1"
        set "PYTHON_CMD=python"
    )
)

REM Если Python не найден — устанавливаем через winget
if !PYTHON_FOUND!==0 (
    winget --version >nul 2>&1
    if !errorlevel!==0 (
        echo [INFO] Установка Python 3.14.2+ через winget...
        winget install --id Python.Python.3.14 --silent --accept-package-agreements --accept-source-agreements
        timeout /t 8 /nobreak >nul
        REM Проверяем установку
        py -3.14 --version >nul 2>&1
        if !errorlevel!==0 (
            set "PYTHON_FOUND=1"
            set "PYTHON_CMD=py -3.14"
        ) else (
            REM Пробуем еще раз после небольшой задержки
            timeout /t 3 /nobreak >nul
            py -3.14 --version >nul 2>&1
            if !errorlevel!==0 (
                set "PYTHON_FOUND=1"
                set "PYTHON_CMD=py -3.14"
            )
        )
    )
    if !PYTHON_FOUND!==0 (
        echo [ERROR] Python 3.14.2+ не найден. Установите: https://www.python.org/downloads/
        start https://www.python.org/downloads/
        pause
        exit /b 1
    )
)

REM Проверка и обновление .venv для Python 3.14
if exist scripts\ensure_python314_venv.py (
    echo [INFO] Проверка и обновление .venv для Python 3.14...
    if not "!PYTHON_CMD!"=="" (
        !PYTHON_CMD! scripts\ensure_python314_venv.py >nul 2>&1
    ) else (
        python scripts\ensure_python314_venv.py >nul 2>&1
    )
)

REM Проверка наличия необходимых файлов
if not exist app\config.py (
    if exist app\config.example.py (
        echo [INFO] Creating app\config.py from example...
        copy /Y app\config.example.py app\config.py >nul 2>&1
        if exist app\config.py (
            echo [OK] File app\config.py created
        ) else (
            echo [ERROR] Failed to create app\config.py
            pause
            exit /b 1
        )
    ) else (
        echo [ERROR] File app\config.example.py not found!
        pause
        exit /b 1
    )
)

REM Определение Python для запуска: .venv > глобальный Python 3.14+
REM Если venv нет, используем глобальный Python
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    set "PYTHON_BIN=python"
    echo [INFO] Используется .venv
) else (
    REM Используем глобальный Python (venv не требуется)
    if not "!PYTHON_CMD!"=="" (
        set "PYTHON_BIN=!PYTHON_CMD!"
        echo [INFO] Используется глобальный Python (venv не найден)
    ) else (
        REM Пробуем найти любой доступный Python
        python --version >nul 2>&1
        if !errorlevel!==0 (
            set "PYTHON_BIN=python"
            echo [INFO] Используется глобальный Python
        ) else (
            echo [ERROR] Python не найден!
            pause
            exit /b 1
        )
    )
)

REM Проверка наличия Python
"%PYTHON_BIN%" --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python 3.14.2+ не найден!
    echo.
    echo Установите Python 3.14.2 или выше: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM Запуск менеджера
call "%PYTHON_BIN%" "launcher\infobot_manager.py" %*
if errorlevel 1 (
    echo.
    echo [ERROR] Error starting manager. Return code: %ERRORLEVEL%
    echo.
    echo Check:
    echo   1. Is Python 3.14.2+ installed?
    echo   2. Are dependencies installed: use GUI button "Create/Update Environment"
    echo   3. Does app\config.py exist?
    echo   4. Does app\keys.py exist?
    echo.
    pause
)
endlocal

