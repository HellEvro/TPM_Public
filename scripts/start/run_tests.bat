@echo off
chcp 65001 > nul
title InfoBot - Комплексное тестирование системы

echo.
echo ========================================
echo 🧪 КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ INFOBOT
echo ========================================
echo.
echo 📋 Этот скрипт проверит:
echo    ✅ Сервисы (app.py и bots.py)
echo    ✅ Конфигурацию Auto Bot
echo    ✅ RSI данные и фильтры
echo    ✅ Управление ботами
echo    ✅ Историю торговли
echo    ✅ Защитные механизмы
echo    ✅ API endpoints
echo    ✅ Файлы данных
echo.
echo ⚠️  ВАЖНО: Убедитесь что запущены:
echo    - app.py (порт 5000)
echo    - bots.py (порт 5001)
echo.
pause

echo.
echo 🚀 Запуск тестов...
echo.

python test_full_system.py

echo.
echo.
echo ========================================
echo 📊 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО
echo ========================================
echo.
echo 📄 Отчет сохранен: logs\test_report.json
echo.
pause

