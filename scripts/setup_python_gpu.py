#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Установка PyTorch с GPU поддержкой для InfoBot.
Устанавливает PyTorch в текущее окружение Python (Python 3.14+).
"""

import sys
import os
import subprocess
import platform

if platform.system() == 'Windows':
    try:
        if getattr(sys.stdout, 'encoding', None) != 'utf-8':
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


def check_gpu_available():
    """Проверяет наличие NVIDIA GPU"""
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def install_pytorch():
    """Устанавливает PyTorch с GPU поддержкой (если доступен) или CPU версию"""
    print("=" * 80)
    print("УСТАНОВКА PYTORCH ДЛЯ INFOBOT")
    print("=" * 80)
    print(f"Текущий Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print()
    
    # Проверяем наличие GPU
    has_gpu = check_gpu_available()
    
    if has_gpu:
        print("[INFO] NVIDIA GPU обнаружен, устанавливаю PyTorch с CUDA поддержкой...")
        print("[INFO] Пробую CUDA 12.1...")
        
        # Пробуем установить PyTorch с CUDA 12.1
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio',
             '--index-url', 'https://download.pytorch.org/whl/cu121'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("[WARNING] Не удалось установить PyTorch с CUDA 12.1, пробую CUDA 11.8...")
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio',
                 '--index-url', 'https://download.pytorch.org/whl/cu118'],
                capture_output=True,
                text=True
            )
        
        if result.returncode != 0:
            print("[WARNING] Не удалось установить PyTorch с CUDA, устанавливаю CPU версию...")
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio'],
                capture_output=True,
                text=True
            )
    else:
        print("[INFO] GPU не обнаружен, устанавливаю PyTorch (CPU версия)...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio'],
            capture_output=True,
            text=True
        )
    
    if result.returncode != 0:
        print("[ERROR] Ошибка установки PyTorch:")
        print(result.stderr)
        return False
    
    print("[OK] PyTorch установлен")
    
    # Проверяем установку
    try:
        import torch
        print(f"[OK] PyTorch версия: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"[OK] CUDA доступна: {torch.version.cuda}")
            print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[INFO] CUDA недоступна, используется CPU")
    except ImportError:
        print("[ERROR] PyTorch не импортируется после установки")
        return False
    
    return True


def main():
    print("=" * 80)
    print("НАСТРОЙКА PYTORCH ДЛЯ INFOBOT")
    print("=" * 80)
    
    # Проверяем версию Python
    if sys.version_info < (3, 8):
        print("[ERROR] Требуется Python 3.8 или выше")
        print(f"Текущая версия: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return 1
    
    if sys.version_info < (3, 14):
        print("[WARNING] Рекомендуется Python 3.14+, но текущая версия поддерживается")
    
    # Обновляем pip
    print("[INFO] Обновление pip...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                   capture_output=True)
    
    # Устанавливаем PyTorch
    if not install_pytorch():
        return 1
    
    print("\n" + "=" * 80)
    print("ГОТОВО")
    print("=" * 80)
    print("PyTorch установлен в текущее окружение Python")
    print("Теперь можно запускать: python ai.py")
    print("=" * 80)
    return 0


if __name__ == '__main__':
    sys.exit(main())
