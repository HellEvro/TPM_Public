#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Установка TensorFlow с GPU поддержкой через Python 3.11
Создает venv с Python 3.11 и устанавливает TensorFlow GPU
"""

import sys
import subprocess
import os
from pathlib import Path

def run(cmd, desc):
    print(f"\n[STEP] {desc}")
    print(f"Команда: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=900, shell=True)
        print(f"[OK] {desc}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {desc}")
        if e.stderr:
            print(e.stderr[-300:])
        return False

def main():
    print("="*60)
    print("УСТАНОВКА TENSORFLOW GPU ЧЕРЕЗ PYTHON 3.11")
    print("="*60)
    
    # Проверяем Python 3.11
    print("\n[STEP 1] Проверка Python 3.11...")
    result = subprocess.run(['py', '-3.11', '--version'], capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR] Python 3.11 не найден!")
        print("Установите Python 3.11: https://www.python.org/downloads/release/python-3110/")
        return False
    
    print(f"[OK] Python 3.11 найден: {result.stdout.strip()}")
    
    # Создаем venv с Python 3.11
    project_root = Path(__file__).resolve().parents[1]
    venv_path = project_root / '.venv_gpu'
    
    print(f"\n[STEP 2] Создание venv с Python 3.11...")
    if venv_path.exists():
        print("[INFO] .venv_gpu уже существует, удаляю...")
        import shutil
        shutil.rmtree(venv_path)
    
    if not run(['py', '-3.11', '-m', 'venv', str(venv_path)], "Создание venv"):
        return False
    
    # Устанавливаем TensorFlow с GPU
    venv_python = venv_path / 'Scripts' / 'python.exe'
    if not venv_python.exists():
        print("[ERROR] venv Python не найден")
        return False
    
    print(f"\n[STEP 3] Установка TensorFlow с GPU...")
    if not run([str(venv_python), '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'], "Обновление pip"):
        return False
    
    if not run([str(venv_python), '-m', 'pip', 'install', 'tensorflow[and-cuda]', '--no-warn-script-location'], "Установка tensorflow[and-cuda]"):
        print("[WARNING] Не удалось установить tensorflow[and-cuda], пробую tensorflow...")
        run([str(venv_python), '-m', 'pip', 'install', 'tensorflow', '--no-warn-script-location'], "Установка tensorflow")
    
    # Проверяем GPU
    print(f"\n[STEP 4] Проверка GPU...")
    check_cmd = f'"{venv_python}" -c "import tensorflow as tf; print(\'TF:\', tf.__version__); print(\'CUDA:\', tf.test.is_built_with_cuda()); gpus = tf.config.list_physical_devices(\'GPU\'); print(\'GPUs:\', len(gpus)); [print(f\'  GPU {{i}}: {{g.name}}\') for i, g in enumerate(gpus)]"'
    result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    print("\n" + "="*60)
    print("ГОТОВО!")
    print("="*60)
    print(f"Используйте: {venv_python} ai.py")
    print("="*60)
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
