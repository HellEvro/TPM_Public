#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ручная установка CUDA Toolkit и настройка TensorFlow для GPU
Инструкции для пользователя
"""

import sys
import subprocess
import os

def check_nvidia_smi():
    """Проверяет наличие nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("[OK] nvidia-smi работает")
            print(result.stdout[:500])
            return True
    except:
        pass
    print("[ERROR] nvidia-smi не найден")
    return False

def check_cuda_path():
    """Проверяет наличие CUDA в PATH"""
    cuda_paths = [
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA',
        r'C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA',
    ]
    
    for base_path in cuda_paths:
        if os.path.exists(base_path):
            print(f"[OK] CUDA найден: {base_path}")
            # Ищем версии
            for item in os.listdir(base_path):
                version_path = os.path.join(base_path, item)
                if os.path.isdir(version_path) and item.startswith('v'):
                    print(f"  Версия: {item}")
            return True
    
    print("[WARNING] CUDA Toolkit не найден в стандартных путях")
    return False

def main():
    print("="*60)
    print("НАСТРОЙКА CUDA ДЛЯ TENSORFLOW GPU")
    print("="*60)
    
    print("\n[STEP 1] Проверка NVIDIA драйверов...")
    has_nvidia = check_nvidia_smi()
    
    print("\n[STEP 2] Проверка CUDA Toolkit...")
    has_cuda = check_cuda_path()
    
    if not has_nvidia:
        print("\n[ERROR] NVIDIA драйверы не установлены!")
        print("Установите драйверы NVIDIA с сайта: https://www.nvidia.com/drivers")
        return False
    
    if not has_cuda:
        print("\n[INFO] CUDA Toolkit не установлен")
        print("\nИНСТРУКЦИИ ПО УСТАНОВКЕ CUDA:")
        print("1. Скачайте CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
        print("2. Выберите версию CUDA 12.x (совместима с TensorFlow 2.16+)")
        print("3. При установке отметьте 'Add to PATH'")
        print("4. После установки перезапустите терминал")
        print("5. Запустите этот скрипт снова для проверки")
        return False
    
    print("\n[STEP 3] Установка TensorFlow с CUDA библиотеками...")
    print("Пробую установить tensorflow[and-cuda]...")
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'tensorflow[and-cuda]', '--upgrade', '--no-warn-script-location'],
            capture_output=True,
            text=True,
            timeout=1200
        )
        
        if result.returncode == 0:
            print("[OK] TensorFlow с CUDA установлен!")
            
            # Проверяем GPU
            print("\n[STEP 4] Проверка GPU...")
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                if len(gpus) > 0:
                    print(f"[SUCCESS] GPU найден: {len(gpus)} устройств")
                    for i, gpu in enumerate(gpus):
                        print(f"  GPU {i}: {gpu.name}")
                    return True
                else:
                    print("[WARNING] GPU не найден TensorFlow")
                    print("Возможно, требуется перезагрузка системы после установки CUDA")
            except Exception as e:
                print(f"[ERROR] Ошибка проверки: {e}")
        else:
            print("[ERROR] Не удалось установить tensorflow[and-cuda]")
            print(result.stderr[-500:])
            
    except subprocess.TimeoutExpired:
        print("[TIMEOUT] Установка заняла слишком много времени")
    except Exception as e:
        print(f"[ERROR] Ошибка: {e}")
    
    return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
