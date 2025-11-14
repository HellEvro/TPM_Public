@echo off
chcp 65001 >nul
title InfoBot - Trading System

echo.
echo ========================================
echo 🚀 InfoBot - Trading System
echo ========================================
echo.

cd /d "%~dp0"

echo 📁 Текущая директория: %CD%
echo.

REM Проверяем наличие Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python не найден! Установите Python и добавьте его в PATH
    pause
    exit /b 1
)

echo ✅ Python найден
echo.

REM Проверяем наличие основных файлов
if not exist "app.py" (
    echo ❌ Файл app.py не найден!
    pause
    exit /b 1
)

if not exist "bots.py" (
    echo ❌ Файл bots.py не найден!
    pause
    exit /b 1
)

echo ✅ Все файлы найдены
echo.

echo 🚀 Запуск InfoBot...
echo.
echo 📊 Основное приложение: http://localhost:5000
echo 🤖 API ботов: http://localhost:5001
echo.
echo Для остановки нажмите Ctrl+C
echo.

python start_all.py

echo.
echo 🔚 InfoBot остановлен
pause
