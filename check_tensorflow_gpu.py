#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка установки TensorFlow и GPU"""

try:
    import tensorflow as tf
    print(f"TensorFlow версия: {tf.__version__}")
    print(f"CUDA поддержка: {tf.test.is_built_with_cuda()}")
    
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Найдено GPU устройств: {len(gpus)}")
    
    if len(gpus) > 0:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        print("\n✅ GPU доступен для использования!")
    else:
        print("\n⚠️ GPU устройства не найдены TensorFlow")
        
except ImportError:
    print("[ERROR] TensorFlow NOT INSTALLED")
except Exception as e:
    print(f"[ERROR] {e}")
