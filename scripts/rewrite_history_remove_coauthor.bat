@echo off
chcp 65001 >nul
echo ================================================================================
echo Удаление "Co-authored-by: Cursor" из ВСЕХ коммитов (переписывание истории)
echo Займёт примерно 5-10 минут. Не закрывайте окно.
echo ================================================================================
cd /d "%~dp0.."

set FILTER_BRANCH_SQUELCH_WARNING=1
git filter-branch -f --msg-filter "python \"%~dp0remove_coauthor_from_msg.py\"" main

if %ERRORLEVEL% neq 0 (
    echo.
    echo Ошибка переписывания. Проверьте, что Python доступен и скрипт существует.
    pause
    exit /b 1
)

echo.
echo Готово. Выполните вручную:
echo   1. git update-ref -d refs/original/refs/heads/main
echo   2. git push --force origin main
echo После force push история на GitHub будет перезаписана (старые коммиты исчезнут).
echo.
pause
