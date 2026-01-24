#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Финальная установка TensorFlow с GPU поддержкой для Python 3.12
Использует обходной путь для конфликтов зависимостей
"""

import sys
import subprocess
import time

def run_cmd(cmd, desc, timeout=600):
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
        version = tf.__version__
        cuda_built = tf.test.is_built_with_cuda()
        gpus = tf.config.list_physical_devices('GPU')
        print(f"\n[CHECK] TensorFlow {version}")
        print(f"[CHECK] CUDA built: {cuda_built}")
        print(f"[CHECK] GPU devices: {len(gpus)}")
        if len(gpus) > 0:
            for i, gpu in enumerate(gpus):
                print(f"[CHECK]   GPU {i}: {gpu.name}")
            return True
        return False
    except Exception as e:
        print(f"[ERROR] Check failed: {e}")
        return False

def main():
    print("="*60)
    print("УСТАНОВКА TENSORFLOW GPU ДЛЯ PYTHON 3.12")
    print("="*60)
    
    # Шаг 1: Удаляем старый TensorFlow
    print("\n[STEP 1] Удаление старого TensorFlow...")
    run_cmd([sys.executable, '-m', 'pip', 'uninstall', '-y', 'tensorflow', 'tensorflow-cpu', 'tensorflow-gpu'], 
            "Удаление TensorFlow")
    
    # Шаг 2: Устанавливаем TensorFlow 2.16.1 (последняя стабильная для Python 3.12)
    print("\n[STEP 2] Установка TensorFlow 2.16.1...")
    if not run_cmd([sys.executable, '-m', 'pip', 'install', 'tensorflow==2.16.1', '--no-warn-script-location'], 
                   "Установка TensorFlow 2.16.1", timeout=900):
        print("[ERROR] Не удалось установить TensorFlow")
        return False
    
    time.sleep(3)
    
    # Шаг 3: Проверяем текущее состояние
    print("\n[STEP 3] Проверка установки...")
    gpu_works = check_gpu()
    
    if gpu_works:
        print("\n[SUCCESS] GPU уже работает!")
        return True
    
    # Шаг 4: Пробуем добавить CUDA библиотеки вручную
    print("\n[STEP 4] Добавление CUDA библиотек...")
    
    # Список CUDA библиотек (без жестких версий)
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
        run_cmd([sys.executable, '-m', 'pip', 'install', '--upgrade', '--no-warn-script-location', pkg],
                f"Установка {pkg}", timeout=300)
    
    time.sleep(3)
    
    # Шаг 5: Финальная проверка
    print("\n[STEP 5] Финальная проверка...")
    gpu_works = check_gpu()
    
    if gpu_works:
        print("\n" + "="*60)
        print("[SUCCESS] TensorFlow с GPU поддержкой установлен и работает!")
        print("="*60)
        return True
    else:
        print("\n" + "="*60)
        print("[WARNING] TensorFlow установлен, но GPU не обнаружен")
        print("Возможные причины:")
        print("1. TensorFlow собран без CUDA поддержки для Python 3.12")
        print("2. Требуется установка CUDA Toolkit вручную")
        print("3. Несовместимость версий CUDA библиотек")
        print("="*60)
        print("\n[INFO] TensorFlow CPU версия установлена и работает")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
