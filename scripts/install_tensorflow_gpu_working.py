#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
РАБОЧАЯ установка TensorFlow с GPU поддержкой
Обходит конфликты зависимостей через пошаговую установку
"""

import sys
import subprocess
import time

def run(cmd, desc, timeout=600):
    print(f"\n[STEP] {desc}")
    print(f"Команда: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
        print(f"[OK] {desc}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {desc}")
        if e.stderr:
            print(e.stderr[-300:])
        return False
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {desc}")
        return False

def check_gpu():
    try:
        import tensorflow as tf
        print(f"\nTensorFlow: {tf.__version__}")
        print(f"CUDA built: {tf.test.is_built_with_cuda()}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPU devices: {len(gpus)}")
        if len(gpus) > 0:
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            return True
        return False
    except Exception as e:
        print(f"[ERROR] Check failed: {e}")
        return False

def main():
    print("="*60)
    print("УСТАНОВКА TENSORFLOW GPU (РАБОЧИЙ МЕТОД)")
    print("="*60)
    
    # Шаг 1: Удаляем старый TensorFlow
    print("\n[STEP 1] Удаление старого TensorFlow...")
    run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'tensorflow', 'tensorflow-cpu', 'tensorflow-gpu'], 
        "Удаление TensorFlow")
    
    # Шаг 2: Устанавливаем CUDA библиотеки ПЕРВЫМИ
    print("\n[STEP 2] Установка CUDA библиотек...")
    cuda_packages = [
        'nvidia-cudnn-cu12>=8.9',
        'nvidia-cublas-cu12>=12.0',
        'nvidia-cuda-runtime-cu12>=12.0',
        'nvidia-cuda-nvrtc-cu12>=12.0',
        'nvidia-cufft-cu12>=11.0',
        'nvidia-curand-cu12>=10.3',
        'nvidia-cusolver-cu12>=11.4',
        'nvidia-cusparse-cu12>=12.0',
    ]
    
    for pkg in cuda_packages:
        run([sys.executable, '-m', 'pip', 'install', '--upgrade', '--no-warn-script-location', pkg],
            f"Установка {pkg.split('>=')[0]}")
    
    # Шаг 3: Устанавливаем TensorFlow
    print("\n[STEP 3] Установка TensorFlow...")
    run([sys.executable, '-m', 'pip', 'install', 'tensorflow==2.16.1', '--no-warn-script-location'],
        "Установка TensorFlow 2.16.1", timeout=900)
    
    # Шаг 4: Проверяем
    print("\n[STEP 4] Проверка GPU...")
    time.sleep(3)
    if check_gpu():
        print("\n[SUCCESS] GPU работает!")
        return True
    else:
        print("\n[WARNING] GPU не найден, но TensorFlow установлен")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
