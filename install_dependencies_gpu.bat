@echo off
REM Скрипт для установки PyTorch с GPU поддержкой в текущее окружение Python
echo ================================================================================
echo УСТАНОВКА PYTORCH С GPU ПОДДЕРЖКОЙ
echo ================================================================================
echo.

echo [INFO] Используется текущий Python
python --version
echo.

echo [INFO] Проверка наличия GPU...

REM Проверяем наличие NVIDIA GPU
nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [INFO] NVIDIA GPU обнаружен, устанавливаю PyTorch с CUDA поддержкой...
    python -m pip install --upgrade pip
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] Не удалось установить PyTorch с CUDA 12.1, пробую CUDA 11.8...
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    )
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] Не удалось установить PyTorch с CUDA, устанавливаю CPU версию...
        python -m pip install torch torchvision torchaudio
    )
) else (
    echo [INFO] GPU не обнаружен, устанавливаю PyTorch (CPU версия)...
    python -m pip install --upgrade pip
    python -m pip install torch torchvision torchaudio
)

echo.
echo [INFO] Установка остальных зависимостей из requirements.txt...
python -m pip install -r requirements.txt

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [OK] Зависимости установлены успешно!
    echo.
    echo [INFO] Проверка PyTorch:
    python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
) else (
    echo.
    echo [ERROR] Ошибка при установке зависимостей
)

pause
