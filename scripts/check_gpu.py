#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для проверки доступности GPU для TensorFlow
"""

import sys
import subprocess

print("=" * 80)
print("ДИАГНОСТИКА GPU ДЛЯ TENSORFLOW")
print("=" * 80)

# 1. Проверка наличия NVIDIA GPU в системе
print("\n[1] Проверка NVIDIA GPU в системе...")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✅ NVIDIA GPU обнаружен в системе:")
        print(result.stdout)
    else:
        print("❌ nvidia-smi не работает или GPU не найден")
        print("   Вывод:", result.stderr)
except FileNotFoundError:
    print("❌ nvidia-smi не найден - возможно, драйверы NVIDIA не установлены")
except subprocess.TimeoutExpired:
    print("⚠️ nvidia-smi не отвечает")
except Exception as e:
    print(f"❌ Ошибка при проверке nvidia-smi: {e}")

# 2. Проверка установки TensorFlow
print("\n[2] Проверка установки TensorFlow...")
try:
    import tensorflow as tf
    print(f"✅ TensorFlow установлен: версия {tf.__version__}")
    
    # Проверяем, какая версия TensorFlow (CPU или GPU)
    try:
        # Пытаемся импортировать модули GPU
        from tensorflow.python.client import device_lib
        print("✅ TensorFlow импортирован успешно")
    except Exception as e:
        print(f"⚠️ Ошибка импорта модулей TensorFlow: {e}")
        
except ImportError:
    print("❌ TensorFlow не установлен")
    print("   Установите: pip install tensorflow[and-cuda]")
    sys.exit(1)

# 3. Проверка доступности GPU в TensorFlow
print("\n[3] Проверка доступности GPU в TensorFlow...")
try:
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"✅ Найдено GPU устройств: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
            
            # Получаем детальную информацию о GPU
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                if gpu_details:
                    print(f"      Детали: {gpu_details}")
            except:
                pass
    else:
        print("❌ GPU устройства не найдены в TensorFlow")
        
        # Проверяем логические устройства
        logical_gpus = tf.config.list_logical_devices('GPU')
        if logical_gpus:
            print(f"⚠️ Найдены логические GPU устройства: {logical_gpus}")
        
except Exception as e:
    print(f"❌ Ошибка при проверке GPU: {e}")

# 4. Проверка CUDA
print("\n[4] Проверка CUDA...")
try:
    # Проверяем переменные окружения CUDA
    import os
    cuda_path = os.environ.get('CUDA_PATH')
    cuda_home = os.environ.get('CUDA_HOME')
    
    if cuda_path:
        print(f"✅ CUDA_PATH: {cuda_path}")
    if cuda_home:
        print(f"✅ CUDA_HOME: {cuda_home}")
    if not cuda_path and not cuda_home:
        print("⚠️ Переменные окружения CUDA не установлены")
    
    # Проверяем версию CUDA через TensorFlow
    try:
        cuda_version = tf.sysconfig.get_build_info()['cuda_version']
        cudnn_version = tf.sysconfig.get_build_info()['cudnn_version']
        print(f"✅ TensorFlow скомпилирован с CUDA {cuda_version}, cuDNN {cudnn_version}")
    except:
        print("⚠️ Не удалось определить версию CUDA из TensorFlow")
        print("   Возможно, установлена CPU версия TensorFlow")
        
except Exception as e:
    print(f"⚠️ Ошибка при проверке CUDA: {e}")

# 5. Тестовая операция на GPU
print("\n[5] Тестовая операция на GPU...")
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Настраиваем GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Пытаемся выполнить простую операцию на GPU
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"✅ Тестовая операция выполнена успешно на GPU")
            print(f"   Результат: {c.numpy()}")
    else:
        print("⚠️ GPU не доступен для тестовой операции")
        
except Exception as e:
    print(f"❌ Ошибка при выполнении тестовой операции на GPU: {e}")
    print("   Это может означать, что GPU не настроен правильно")

# 6. Рекомендации
print("\n" + "=" * 80)
print("РЕКОМЕНДАЦИИ")
print("=" * 80)

gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("\n❌ GPU не найден. Для использования GPU:")
    print("\n1. Убедитесь, что установлены драйверы NVIDIA:")
    print("   - Проверьте: nvidia-smi")
    print("   - Скачайте с: https://www.nvidia.com/drivers")
    print("\n2. Установите CUDA toolkit:")
    print("   - TensorFlow 2.15+ требует CUDA 11.8 или 12.x")
    print("   - Скачайте с: https://developer.nvidia.com/cuda-downloads")
    print("\n3. Установите cuDNN:")
    print("   - Скачайте с: https://developer.nvidia.com/cudnn")
    print("\n4. Переустановите TensorFlow с поддержкой GPU:")
    print("   pip uninstall tensorflow")
    print("   pip install tensorflow[and-cuda]")
    print("   или")
    print("   pip install tensorflow-gpu")
else:
    print("\n✅ GPU настроен правильно и готов к использованию!")

print("\n" + "=" * 80)
