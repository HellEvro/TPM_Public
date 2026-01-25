@echo off
REM Скрипт для установки зависимостей в .venv_gpu (включая PyTorch)
echo ================================================================================
echo УСТАНОВКА ЗАВИСИМОСТЕЙ В .venv_gpu
echo ================================================================================
echo.

if not exist ".venv_gpu\Scripts\python.exe" (
    echo [ERROR] .venv_gpu не найден!
    echo [INFO] Запустите сначала: py scripts/setup_python_gpu.py
    echo [INFO] Или: python3 scripts/setup_python_gpu.py
    pause
    exit /b 1
)

echo [INFO] Используется Python из .venv_gpu
echo [INFO] Проверка наличия GPU...

REM Проверяем наличие NVIDIA GPU
nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [INFO] NVIDIA GPU обнаружен, устанавливаю PyTorch с CUDA поддержкой...
    .venv_gpu\Scripts\python.exe -m pip install --upgrade pip
    .venv_gpu\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] Не удалось установить PyTorch с CUDA 12.1, пробую CUDA 11.8...
        .venv_gpu\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    )
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] Не удалось установить PyTorch с CUDA, устанавливаю CPU версию...
        .venv_gpu\Scripts\python.exe -m pip install torch torchvision torchaudio
    )
) else (
    echo [INFO] GPU не обнаружен, устанавливаю PyTorch (CPU версия)...
    .venv_gpu\Scripts\python.exe -m pip install --upgrade pip
    .venv_gpu\Scripts\python.exe -m pip install torch torchvision torchaudio
)

echo [INFO] Установка остальных зависимостей из requirements.txt...
.venv_gpu\Scripts\python.exe -m pip install -r requirements.txt

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [OK] Зависимости установлены успешно!
) else (
    echo.
    echo [ERROR] Ошибка при установке зависимостей
)

pause
