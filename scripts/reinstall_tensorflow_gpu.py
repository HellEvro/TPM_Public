#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полная переустановка TensorFlow с GPU поддержкой
Удаляет все конфликтующие пакеты и устанавливает заново
"""

import sys
import subprocess
import time

def run_command(cmd, description):
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
            timeout=600
        )
        print(f"[OK] {description} - успешно")
        if result.stdout:
            print(result.stdout[-500:])  # Последние 500 символов
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} - ошибка")
        if e.stderr:
            print(e.stderr[-500:])
        return False
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {description} - таймаут")
        return False

def main():
    print("="*60)
    print("ПОЛНАЯ ПЕРЕУСТАНОВКА TENSORFLOW С GPU")
    print("="*60)
    print()
    
    # Список пакетов для удаления
    packages_to_remove = [
        'tensorflow',
        'tensorflow-cpu',
        'tensorflow-gpu',
        'tensorflow-intel',
        'tensorflow-estimator',
        'tensorflow-io',
        'keras',
        'keras-nightly',
        'nvidia-cublas-cu12',
        'nvidia-cuda-cupti-cu12',
        'nvidia-cuda-nvcc-cu12',
        'nvidia-cuda-nvrtc-cu12',
        'nvidia-cuda-runtime-cu12',
        'nvidia-cudnn-cu12',
        'nvidia-cufft-cu12',
        'nvidia-curand-cu12',
        'nvidia-cusolver-cu12',
        'nvidia-cusparse-cu12',
        'nvidia-nccl-cu12',
    ]
    
    # Шаг 1: Удаление всех TensorFlow и NVIDIA пакетов
    print("\nШАГ 1: Удаление всех TensorFlow и NVIDIA пакетов...")
    for package in packages_to_remove:
        run_command(
            [sys.executable, '-m', 'pip', 'uninstall', '-y', package],
            f"Удаление {package}"
        )
    
    # Шаг 2: Очистка кэша pip
    print("\nШАГ 2: Очистка кэша pip...")
    run_command(
        [sys.executable, '-m', 'pip', 'cache', 'purge'],
        "Очистка кэша pip"
    )
    
    # Шаг 3: Установка TensorFlow с GPU
    print("\nШАГ 3: Установка TensorFlow с GPU поддержкой...")
    print("Это может занять 10-15 минут...")
    
    # Методы установки в порядке приоритета (согласно документации pip)
    installation_methods = [
        # Метод 1: tensorflow[and-cuda] с --upgrade и --force-reinstall
        ([sys.executable, '-m', 'pip', 'install', '--upgrade', '--force-reinstall', 'tensorflow[and-cuda]', '--no-warn-script-location', '--no-cache-dir'], 
         "tensorflow[and-cuda] с --upgrade --force-reinstall"),
        
        # Метод 2: tensorflow[and-cuda] без версии (ослабленные требования)
        ([sys.executable, '-m', 'pip', 'install', '--upgrade', 'tensorflow[and-cuda]', '--no-warn-script-location', '--no-cache-dir'],
         "tensorflow[and-cuda] с --upgrade"),
        
        # Метод 3: tensorflow 2.15.0 (стабильная версия)
        ([sys.executable, '-m', 'pip', 'install', '--upgrade', 'tensorflow==2.15.0', '--no-warn-script-location', '--no-cache-dir'],
         "tensorflow 2.15.0"),
        
        # Метод 4: базовый tensorflow (может автоматически подхватить GPU если CUDA установлен)
        ([sys.executable, '-m', 'pip', 'install', '--upgrade', 'tensorflow', '--no-warn-script-location', '--no-cache-dir'],
         "tensorflow (базовая версия)")
    ]
    
    success = False
    for cmd, method_name in installation_methods:
        print(f"\nПробую метод: {method_name}...")
        success = run_command(cmd, f"Установка через {method_name}")
        if success:
            print(f"[OK] Успешно установлено через: {method_name}")
            break
        else:
            print(f"[SKIP] Метод {method_name} не сработал, пробую следующий...")
    
    # Если tensorflow установлен, но без CUDA - пробуем добавить CUDA библиотеки
    if success:
        try:
            import tensorflow as tf
            if not tf.test.is_built_with_cuda():
                print("\nTensorFlow установлен, но без CUDA. Пробую добавить CUDA библиотеки...")
                run_command(
                    [sys.executable, '-m', 'pip', 'install', '--upgrade', 'nvidia-cudnn-cu12', 'nvidia-cublas-cu12', '--no-warn-script-location', '--no-cache-dir'],
                    "Добавление CUDA библиотек"
                )
        except ImportError:
            pass
    
    # Шаг 4: Проверка установки
    print("\nШАГ 4: Проверка установки...")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow версия: {tf.__version__}")
        print(f"✅ CUDA поддержка: {tf.test.is_built_with_cuda()}")
        
        gpus = tf.config.list_physical_devices('GPU')
        print(f"✅ Найдено GPU устройств: {len(gpus)}")
        
        if len(gpus) > 0:
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            print("\n[OK] TensorFlow с GPU успешно установлен и готов к использованию!")
        else:
            print("\n[WARNING] TensorFlow установлен, но GPU не найден")
            print("   Возможно, требуется установка CUDA драйверов")
            
    except ImportError:
        print("[ERROR] TensorFlow не установлен")
        return False
    except Exception as e:
        print(f"[ERROR] Ошибка при проверке: {e}")
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
