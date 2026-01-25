#!/bin/bash
# Скрипт для установки зависимостей в .venv_gpu (для Git Bash)

echo "================================================================================"
echo "УСТАНОВКА ЗАВИСИМОСТЕЙ В .venv_gpu"
echo "================================================================================"
echo ""

if [ ! -f ".venv_gpu/Scripts/python.exe" ]; then
    echo "[ERROR] .venv_gpu не найден!"
    echo "[INFO] Запустите сначала: python scripts/setup_python_gpu.py"
    exit 1
fi

echo "[INFO] Установка зависимостей из requirements.txt..."
.venv_gpu/Scripts/python.exe -m pip install --upgrade pip
.venv_gpu/Scripts/python.exe -m pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "[OK] Зависимости установлены успешно!"
else
    echo ""
    echo "[ERROR] Ошибка при установке зависимостей"
    exit 1
fi
