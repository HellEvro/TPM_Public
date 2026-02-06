#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Автоматическое исправление установки TensorFlow с GPU поддержкой
Проверяет и исправляет до тех пор, пока GPU не заработает
"""

import sys
import subprocess
import time
import os

def run_command(cmd, description, timeout=300):
    """Выполняет команду и выводит результат"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Выполняю: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        print(f"[OK] {description} - успешно")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} - ошибка")
        if e.stderr:
            print(e.stderr[-500:])
        return False, e.stderr if e.stderr else ""
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {description} - таймаут")
        return False, ""

def check_gpu_available():
    """Проверяет наличие GPU через nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except:
        return False

def check_tensorflow_gpu():
    """Проверяет установку TensorFlow и доступность GPU"""
    try:
        import tensorflow as tf
        version = tf.__version__
        cuda_built = tf.test.is_built_with_cuda()
        gpus = tf.config.list_physical_devices('GPU')
        
        print(f"\n[CHECK] TensorFlow версия: {version}")
        print(f"[CHECK] CUDA поддержка: {cuda_built}")
        print(f"[CHECK] Найдено GPU устройств: {len(gpus)}")
        
        if len(gpus) > 0:
            for i, gpu in enumerate(gpus):
                print(f"[CHECK]   GPU {i}: {gpu.name}")
            return True, version, cuda_built, len(gpus)
        else:
            return False, version, cuda_built, 0
    except ImportError:
        print("[CHECK] TensorFlow не установлен")
        return False, None, False, 0
    except Exception as e:
        print(f"[CHECK] Ошибка при проверке: {e}")
        return False, None, False, 0

def install_tensorflow_with_cuda():
    """Устанавливает TensorFlow с CUDA поддержкой"""
    
    # Методы установки в порядке приоритета (для Python 3.12 доступны только версии 2.16+)
    methods = [
        # Метод 1: tensorflow 2.16.1 + CUDA библиотеки (стабильная для Python 3.12)
        (['tensorflow==2.16.1'], "tensorflow 2.16.1"),
        
        # Метод 2: tensorflow 2.17.0 + CUDA библиотеки
        (['tensorflow==2.17.0'], "tensorflow 2.17.0"),
        
        # Метод 3: tensorflow 2.18.0 + CUDA библиотеки
        (['tensorflow==2.18.0'], "tensorflow 2.18.0"),
        
        # Метод 4: tensorflow 2.19.0 + CUDA библиотеки
        (['tensorflow==2.19.0'], "tensorflow 2.19.0"),
        
        # Метод 5: tensorflow 2.20.0 + CUDA библиотеки
        (['tensorflow==2.20.0'], "tensorflow 2.20.0"),
    ]
    
    for packages, method_name in methods:
        print(f"\n[TRY] Пробую метод: {method_name}")
        
        # Удаляем старый TensorFlow
        print("[STEP] Удаляю старый TensorFlow...")
        subprocess.run(
            [sys.executable, '-m', 'pip', 'uninstall', '-y', 'tensorflow', 'tensorflow-cpu', 'tensorflow-gpu'],
            capture_output=True
        )
        
        # Устанавливаем новый
        cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade', '--no-warn-script-location'] + packages
        success, output = run_command(cmd, f"Установка {method_name}", timeout=600)
        
        if success:
            # Проверяем результат
            time.sleep(2)
            gpu_works, version, cuda_built, gpu_count = check_tensorflow_gpu()
            
            if gpu_works and gpu_count > 0:
                print(f"\n[SUCCESS] GPU работает! TensorFlow {version}, GPU устройств: {gpu_count}")
                return True
            
            # Если TensorFlow установлен, но GPU не работает - пробуем добавить CUDA библиотеки
            if version:
                print(f"\n[INFO] TensorFlow установлен ({version}), но GPU не работает")
                print("[STEP] Пробую добавить CUDA библиотеки...")
                
                cuda_packages = [
                    'nvidia-cudnn-cu12',
                    'nvidia-cublas-cu12',
                    'nvidia-cuda-runtime-cu12',
                    'nvidia-cuda-nvrtc-cu12',
                ]
                
                for cuda_pkg in cuda_packages:
                    run_command(
                        [sys.executable, '-m', 'pip', 'install', '--upgrade', '--no-warn-script-location', cuda_pkg],
                        f"Установка {cuda_pkg}",
                        timeout=300
                    )
                
                # Проверяем снова
                time.sleep(2)
                gpu_works, version, cuda_built, gpu_count = check_tensorflow_gpu()
                if gpu_works and gpu_count > 0:
                    print(f"\n[SUCCESS] GPU работает после установки CUDA библиотек!")
                    return True
    
    return False

def main():
    print("="*60)
    print("АВТОМАТИЧЕСКОЕ ИСПРАВЛЕНИЕ TENSORFLOW GPU")
    print("="*60)
    
    # Проверяем наличие GPU
    print("\n[STEP 1] Проверка наличия GPU в системе...")
    if not check_gpu_available():
        print("[ERROR] GPU не обнаружен в системе (nvidia-smi не работает)")
        print("[INFO] Установка TensorFlow CPU версии...")
        run_command(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', 'tensorflow', '--no-warn-script-location'],
            "Установка TensorFlow (CPU версия)"
        )
        return False
    
    print("[OK] GPU обнаружен в системе")
    
    # Проверяем текущее состояние
    print("\n[STEP 2] Проверка текущей установки TensorFlow...")
    gpu_works, version, cuda_built, gpu_count = check_tensorflow_gpu()
    
    if gpu_works and gpu_count > 0:
        print(f"\n[SUCCESS] GPU уже работает! TensorFlow {version}, GPU устройств: {gpu_count}")
        return True
    
    # Пробуем исправить
    print("\n[STEP 3] Попытка исправления установки...")
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        print(f"\n[ATTEMPT {attempt}/{max_attempts}]")
        
        if install_tensorflow_with_cuda():
            print("\n" + "="*60)
            print("УСПЕХ! TensorFlow с GPU поддержкой установлен и работает!")
            print("="*60)
            return True
        
        print(f"\n[WARNING] Попытка {attempt} не удалась, пробую следующий метод...")
        time.sleep(3)
    
    print("\n" + "="*60)
    print("НЕ УДАЛОСЬ УСТАНОВИТЬ TENSORFLOW С GPU ПОДДЕРЖКОЙ")
    print("="*60)
    print("Установлена CPU версия TensorFlow")
    return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
