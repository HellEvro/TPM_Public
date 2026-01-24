#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для автоматической установки TensorFlow с поддержкой GPU
Проверяет наличие GPU и устанавливает правильную версию TensorFlow

Использует ту же логику, что и bot_engine/ai/tensorflow_setup.py
"""

import sys
import os

# Добавляем путь к проекту
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Используем общую логику из tensorflow_setup
try:
    from bot_engine.ai.tensorflow_setup import (
        check_python_version,
        check_gpu_available,
        check_tensorflow_installation,
        install_tensorflow_with_gpu,
        ensure_tensorflow_setup
    )
    USE_SHARED_LOGIC = True
except ImportError:
    USE_SHARED_LOGIC = False
    import subprocess

def check_gpu():
    """Проверяет наличие NVIDIA GPU в системе"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return False

def check_tensorflow():
    """Проверяет установлен ли TensorFlow"""
    try:
        import tensorflow as tf
        return True, tf.__version__
    except ImportError:
        return False, None

def check_cuda_support():
    """Проверяет, скомпилирован ли TensorFlow с поддержкой CUDA"""
    try:
        import tensorflow as tf
        return tf.test.is_built_with_cuda()
    except:
        return False

def install_tensorflow_gpu():
    """Устанавливает TensorFlow с поддержкой GPU"""
    print("=" * 80)
    print("УСТАНОВКА TENSORFLOW С ПОДДЕРЖКОЙ GPU")
    print("=" * 80)
    
    if USE_SHARED_LOGIC:
        # Используем общую логику
        return ensure_tensorflow_setup()
    
    # Fallback на старую логику, если общий модуль недоступен
    print("\n[1] Проверка GPU...")
    has_gpu = check_gpu()
    if has_gpu:
        print("✅ NVIDIA GPU обнаружен в системе")
    else:
        print("⚠️ NVIDIA GPU не обнаружен")
        response = input("Продолжить установку TensorFlow с GPU поддержкой? (y/n): ")
        if response.lower() != 'y':
            print("Установка отменена")
            return False
    
    # ... остальная логика
    return install_tensorflow_with_gpu()[0]

def verify_installation():
    """Проверяет установку TensorFlow и GPU"""
    print("\n[4] Проверка установки...")
    if USE_SHARED_LOGIC:
        tf_info = check_tensorflow_installation()
        if tf_info['installed']:
            print(f"✅ TensorFlow версия: {tf_info['version']}")
            print(f"CUDA поддержка: {'✅ Да' if tf_info['cuda_built'] else '❌ Нет'}")
            if tf_info['gpus_found'] > 0:
                print(f"✅ Найдено GPU устройств: {tf_info['gpus_found']}")
                for i, gpu in enumerate(tf_info['gpu_devices']):
                    print(f"   GPU {i}: {gpu.name}")
            else:
                print("⚠️ GPU устройства не найдены")
            return True
        else:
            print("❌ TensorFlow не установлен")
            return False
    
    # Fallback логика
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow версия: {tf.__version__}")
        cuda_built = tf.test.is_built_with_cuda()
        print(f"CUDA поддержка: {'✅ Да' if cuda_built else '❌ Нет'}")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ Найдено GPU устройств: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("⚠️ GPU устройства не найдены")
        return True
    except ImportError:
        print("❌ TensorFlow не установлен")
        return False

if __name__ == "__main__":
    success = install_tensorflow_gpu()
    if success:
        verify_installation()
        print("\n" + "=" * 80)
        print("УСТАНОВКА ЗАВЕРШЕНА")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("УСТАНОВКА НЕ ЗАВЕРШЕНА")
        print("=" * 80)
        sys.exit(1)
