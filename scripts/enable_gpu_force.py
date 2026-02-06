#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Принудительное включение GPU для TensorFlow
Использует системные CUDA библиотеки
"""

import os
import sys

# Настройка переменных окружения для CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Добавляем пути к CUDA в PATH
cuda_paths = [
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin',
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin',
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\libnvvp',
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp',
]

for path in cuda_paths:
    if os.path.exists(path) and path not in os.environ.get('PATH', ''):
        os.environ['PATH'] = path + os.pathsep + os.environ.get('PATH', '')

try:
    import tensorflow as tf
    
    print("="*60)
    print("ПРИНУДИТЕЛЬНАЯ НАСТРОЙКА GPU")
    print("="*60)
    
    # Принудительно настраиваем GPU
    gpus = tf.config.list_physical_devices('GPU')
    
    if len(gpus) == 0:
        # Пробуем использовать логические устройства
        logical_gpus = tf.config.list_logical_devices('GPU')
        if len(logical_gpus) > 0:
            print(f"[INFO] Найдены логические GPU: {len(logical_gpus)}")
            for i, gpu in enumerate(logical_gpus):
                print(f"  Logical GPU {i}: {gpu.name}")
        else:
            # Пробуем создать виртуальное GPU устройство
            print("[INFO] Пробую создать виртуальное GPU устройство...")
            try:
                tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
            except:
                pass
    
    # Проверяем снова
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nTensorFlow версия: {tf.__version__}")
    print(f"CUDA built: {tf.test.is_built_with_cuda()}")
    print(f"GPU devices: {len(gpus)}")
    
    if len(gpus) > 0:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            # Настраиваем рост памяти
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except:
                pass
        
        # Тест вычислений на GPU
        print("\n[TEST] Тест вычислений на GPU...")
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            result = c.numpy()
            print(f"Результат: {result}")
            print("\n[SUCCESS] GPU РАБОТАЕТ!")
            sys.exit(0)
    else:
        print("\n[WARNING] GPU устройства не найдены")
        print("TensorFlow собран без CUDA поддержки для Python 3.12")
        sys.exit(1)
        
except Exception as e:
    print(f"[ERROR] Ошибка: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
