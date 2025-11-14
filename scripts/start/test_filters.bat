@echo off
chcp 65001 > nul
if "%1"=="" (
    echo Использование: test_filters.bat SYMBOL
    echo Пример: test_filters.bat 1000000CHEEMS
    exit /b 1
)
python test_coin_filters.py %1
pause

