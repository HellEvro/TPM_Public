#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПРИНУДИТЕЛЬНАЯ установка TensorFlow с GPU - работает до успеха
Пробует все возможные методы установки
"""

import sys
import subprocess
import time

def run(cmd, desc, timeout=900):
    print(f"\n{'='*60}")
    print(f"[STEP] {desc}")
    print(f"{'='*60}")
    print(f"Команда: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
        print(f"[OK] {desc}")
        if result.stdout:
            # Показываем последние строки вывода
            lines = result.stdout.strip().split('\n')
            for line in lines[-5:]:
                if line.strip():
                    print(f"  {line}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {desc}")
        if e.stderr:
            err_lines = e.stderr.strip().split('\n')
            for line in err_lines[-10:]:
                if line.strip() and 'error' in line.lower():
                    print(f"  {line}")
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
        
        print(f"\n{'='*60}")
        print("ПРОВЕРКА TENSORFLOW GPU")
        print(f"{'='*60}")
        print(f"TensorFlow версия: {version}")
        print(f"CUDA built: {cuda_built}")
        print(f"GPU devices: {len(gpus)}")
        
        if len(gpus) > 0:
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            print("\n[SUCCESS] GPU РАБОТАЕТ!")
            return True
        else:
            print("\n[WARNING] GPU не найден")
            return False
    except Exception as e:
        print(f"[ERROR] Проверка не удалась: {e}")
        return False

def main():
    print("="*60)
    print("ПРИНУДИТЕЛЬНАЯ УСТАНОВКА TENSORFLOW GPU")
    print("Пробует все методы до успеха")
    print("="*60)
    
    # Методы установки
    methods = [
        # Метод 1: tensorflow[and-cuda] с --no-cache-dir
        (['tensorflow[and-cuda]'], '--upgrade', '--no-cache-dir', '--no-warn-script-location'),
        
        # Метод 2: tensorflow[and-cuda] с --use-deprecated=legacy-resolver
        (['tensorflow[and-cuda]'], '--upgrade', '--use-deprecated=legacy-resolver', '--no-warn-script-location'),
        
        # Метод 3: tensorflow[and-cuda] без версии
        (['tensorflow[and-cuda]'], '--upgrade', '--no-warn-script-location'),
        
        # Метод 4: Установка CUDA библиотек отдельно, затем tensorflow
        (['nvidia-cudnn-cu12', 'nvidia-cublas-cu12', 'nvidia-cuda-runtime-cu12'], '--upgrade', '--no-warn-script-location'),
        (['tensorflow==2.16.1'], '--no-warn-script-location'),
        
        # Метод 5: tf-nightly[and-cuda]
        (['tf-nightly[and-cuda]'], '--upgrade', '--no-warn-script-location'),
    ]
    
    # Удаляем старый TensorFlow
    print("\n[STEP 0] Удаление старого TensorFlow...")
    run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'tensorflow', 'tensorflow-cpu', 'tensorflow-gpu', 'tf-nightly'],
        "Удаление TensorFlow")
    
    # Проверяем текущее состояние
    print("\n[CHECK] Проверка текущего состояния...")
    if check_gpu():
        print("\n[SUCCESS] GPU уже работает!")
        return True
    
    # Пробуем методы установки
    for method_idx, method in enumerate(methods, 1):
        print(f"\n{'='*60}")
        print(f"МЕТОД {method_idx}/{len(methods)}")
        print(f"{'='*60}")
        
        packages = method[0]
        flags = list(method[1:])
        
        cmd = [sys.executable, '-m', 'pip', 'install'] + flags + packages
        
        if run(cmd, f"Установка через метод {method_idx}", timeout=1200):
            # Ждем немного и проверяем
            time.sleep(5)
            if check_gpu():
                print(f"\n[SUCCESS] Метод {method_idx} сработал! GPU работает!")
                return True
            else:
                print(f"\n[WARNING] Метод {method_idx} установил TensorFlow, но GPU не работает")
                print("Пробую следующий метод...")
        else:
            print(f"\n[WARNING] Метод {method_idx} не сработал, пробую следующий...")
    
    print("\n" + "="*60)
    print("НЕ УДАЛОСЬ УСТАНОВИТЬ TENSORFLOW С GPU")
    print("="*60)
    print("Установлена CPU версия TensorFlow")
    print("Система будет работать, но медленнее на CPU")
    return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
