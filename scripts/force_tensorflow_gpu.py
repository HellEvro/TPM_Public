#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Принудительная установка TensorFlow с GPU через обход конфликтов зависимостей
Использует --no-deps и ручную установку зависимостей
"""

import sys
import subprocess
import sys

def run(cmd, desc):
    print(f"\n[STEP] {desc}")
    print(f"Команда: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        print(f"[OK] {desc}")
        if result.stdout:
            print(result.stdout[-200:])
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {desc}")
        if e.stderr:
            print(e.stderr[-500:])
        return False

def main():
    print("="*60)
    print("ПРИНУДИТЕЛЬНАЯ УСТАНОВКА TENSORFLOW GPU")
    print("="*60)
    
    # Удаляем все
    print("\n[STEP 1] Очистка...")
    run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'tensorflow', 'tensorflow-cpu', 'tensorflow-gpu', 'tf-nightly'],
        "Удаление TensorFlow")
    
    # Устанавливаем CUDA библиотеки ПЕРВЫМИ (без версий)
    print("\n[STEP 2] Установка CUDA библиотек...")
    cuda_packages = [
        'nvidia-cudnn-cu12',
        'nvidia-cublas-cu12', 
        'nvidia-cuda-runtime-cu12',
        'nvidia-cuda-nvrtc-cu12',
        'nvidia-cufft-cu12',
        'nvidia-curand-cu12',
        'nvidia-cusolver-cu12',
        'nvidia-cusparse-cu12',
    ]
    
    for pkg in cuda_packages:
        run([sys.executable, '-m', 'pip', 'install', '--upgrade', '--no-warn-script-location', pkg],
            f"Установка {pkg}")
    
    # Устанавливаем TensorFlow
    print("\n[STEP 3] Установка TensorFlow...")
    run([sys.executable, '-m', 'pip', 'install', 'tensorflow==2.16.1', '--no-warn-script-location'],
        "Установка TensorFlow 2.16.1")
    
    # Проверяем
    print("\n[STEP 4] Проверка...")
    try:
        import tensorflow as tf
        print(f"TensorFlow: {tf.__version__}")
        print(f"CUDA built: {tf.test.is_built_with_cuda()}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPU devices: {len(gpus)}")
        if len(gpus) > 0:
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            print("\n[SUCCESS] GPU работает!")
            return True
        else:
            print("\n[WARNING] GPU не найден, но TensorFlow установлен")
            return False
    except Exception as e:
        print(f"[ERROR] Проверка не удалась: {e}")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
