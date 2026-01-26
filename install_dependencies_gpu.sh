#!/bin/bash
# Скрипт для установки PyTorch с GPU поддержкой в текущее окружение Python

echo "================================================================================"
echo "УСТАНОВКА PYTORCH С GPU ПОДДЕРЖКОЙ"
echo "================================================================================"
echo ""

echo "[INFO] Используется текущий Python"
python3 --version
echo ""

echo "[INFO] Проверка наличия GPU..."

# Проверяем наличие NVIDIA GPU
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "[INFO] NVIDIA GPU обнаружен, устанавливаю PyTorch с CUDA поддержкой..."
    python3 -m pip install --upgrade pip
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if [ $? -ne 0 ]; then
        echo "[WARNING] Не удалось установить PyTorch с CUDA 12.1, пробую CUDA 11.8..."
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
    if [ $? -ne 0 ]; then
        echo "[WARNING] Не удалось установить PyTorch с CUDA, устанавливаю CPU версию..."
        python3 -m pip install torch torchvision torchaudio
    fi
else
    echo "[INFO] GPU не обнаружен, устанавливаю PyTorch (CPU версия)..."
    python3 -m pip install --upgrade pip
    python3 -m pip install torch torchvision torchaudio
fi

echo ""
echo "[INFO] Установка остальных зависимостей из requirements.txt..."
python3 -m pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "[OK] Зависимости установлены успешно!"
    echo ""
    echo "[INFO] Проверка PyTorch:"
    python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
else
    echo ""
    echo "[ERROR] Ошибка при установке зависимостей"
    exit 1
fi
