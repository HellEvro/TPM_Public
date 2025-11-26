@echo off
REM Скрипт для открытия старого workspace Cursor с сессиями

echo ================================================================================
echo ОТКРЫТИЕ СТАРОГО WORKSPACE CURSOR С СЕССИЯМИ
echo ================================================================================
echo.
echo Этот скрипт откроет старый workspace Cursor, где сохранены ваши сессии.
echo.
echo Workspace ID: 872fa8ec5ac17c960b0d21a7e7a0af40
echo Путь: E:\Drive\TRADEBOT\InfoBot
echo.

REM Получаем путь к Cursor
set CURSOR_PATH=%LOCALAPPDATA%\Programs\cursor\Cursor.exe

if not exist "%CURSOR_PATH%" (
    echo [!] Cursor не найден по пути: %CURSOR_PATH%
    echo Попробуйте найти Cursor.exe вручную
    pause
    exit /b 1
)

echo [*] Открываю старый workspace в Cursor...
echo.

REM Открываем проект напрямую
start "" "%CURSOR_PATH%" "E:\Drive\TRADEBOT\InfoBot"

echo [*] Cursor должен открыться с проектом
echo.
echo ДЕЙСТВИЯ:
echo 1. В Cursor нажмите Alt+Ctrl+' для просмотра истории чатов
echo 2. Или Ctrl+E для фоновых агентов
echo.
echo Если сессии не появились, попробуйте:
echo - File -^> Open Recent
echo - Найдите старый проект InfoBot в списке
echo.
pause

