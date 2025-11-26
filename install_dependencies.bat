@echo off
chcp 65001 >nul 2>&1
echo ========================================
echo Установка зависимостей InfoBot
echo ========================================
echo.

REM Проверка Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python не найден!
    echo Установите Python 3.8 или выше с https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [INFO] Python найден
python --version

echo.
echo [INFO] Обновление pip...
python -m pip install --upgrade pip setuptools wheel

echo.
echo [INFO] Установка зависимостей из requirements.txt...
echo Это может занять несколько минут...
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Ошибка при установке зависимостей!
    echo.
    echo Попробуйте:
    echo   1. Убедитесь, что у вас установлен Python 3.8 или выше
    echo   2. Проверьте подключение к интернету
    echo   3. Попробуйте запустить от имени администратора
    echo.
    pause
    exit /b 1
)

echo.
echo [OK] Зависимости успешно установлены!
echo.
echo Теперь вы можете запустить:
echo   python app.py
echo   или
echo   start_infobot_manager.bat
echo.
pause

