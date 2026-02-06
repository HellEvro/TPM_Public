@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0\.."
echo ========================================
echo InfoBot - Install dependencies
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo Install Python 3.12 from https://www.python.org/downloads/release/python-3120/
    pause
    exit /b 1
)

echo [INFO] Python found
python --version

echo.
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel --no-warn-script-location

echo.
echo [INFO] Installing from requirements.txt...
python -m pip install -r requirements.txt --no-warn-script-location

if errorlevel 1 (
    echo.
    echo [ERROR] Installation failed.
    echo Try: 1) Python 3.12  2) Internet  3) Run as admin
    echo.
    pause
    exit /b 1
)

echo.
echo [OK] Done.
echo Run: python app.py  or  launcher\start_infobot_manager.bat
echo.
pause
