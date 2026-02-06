@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
cd /d %~dp0\..

REM Устанавливаем кодировку для Python
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

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

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    set "PYTHON_BIN=python"
) else (
    echo [WARN] Virtual environment not found. Falling back to system Python.
    REM Пробуем разные варианты Python
    if exist %SystemRoot%\py.exe (
        %SystemRoot%\py.exe -3 --version >nul 2>&1
        if errorlevel 1 (
            set "PYTHON_BIN=python"
        ) else (
            set "PYTHON_BIN=py -3"
        )
    ) else (
        set "PYTHON_BIN=python"
    )
    
    REM Проверяем, что Python действительно доступен
    "%PYTHON_BIN%" --version >nul 2>&1
    if errorlevel 1 (
        REM Пробуем python3
        python3 --version >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON_BIN=python3"
        ) else (
            REM Пробуем python3.exe
            python3.exe --version >nul 2>&1
            if not errorlevel 1 (
                set "PYTHON_BIN=python3.exe"
            )
        )
    )
    
    REM Проверяем наличие зависимостей - если нет, запускаем автонастройку
    "%PYTHON_BIN%" -c "import flask" >nul 2>&1
    if errorlevel 1 (
        echo.
        echo [INFO] Dependencies not found. Running automatic setup...
        echo.
        REM Запускаем скрипт настройки с полным подавлением вывода
        "%PYTHON_BIN%" "scripts\first_run_setup.py" >nul 2>&1
        if errorlevel 1 (
            echo [ERROR] Automatic setup failed.
            echo Run manually: python scripts\first_run_setup.py
            pause
            exit /b 1
        )
        echo.
        echo [OK] Automatic setup completed. Restarting manager...
        echo.
        REM Активируем .venv если он был создан
        if exist .venv\Scripts\activate.bat (
            call .venv\Scripts\activate.bat
            set "PYTHON_BIN=python"
        )
    )
)

REM Проверка наличия Python
"%PYTHON_BIN%" --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python not found!
    echo.
    echo Please install Python 3.8 or higher:
    echo   1. Download from: https://www.python.org/downloads/
    echo   2. During installation, check "Add Python to PATH"
    echo   3. Restart this script after installation
    echo.
    echo Or try running manually:
    echo   python --version
    echo   py --version
    echo   python3 --version
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
    echo Running diagnostics...
    call "%PYTHON_BIN%" "scripts\check_setup.py"
    echo.
    echo Check:
    echo   1. Is Python 3.8+ installed?
    echo   2. Are dependencies installed: pip install -r requirements.txt
    echo   3. Does app\config.py exist?
    echo   4. Does app\keys.py exist?
    echo.
    pause
)
endlocal

