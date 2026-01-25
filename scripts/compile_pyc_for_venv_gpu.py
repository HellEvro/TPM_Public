#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для компиляции .pyc файлов для Python 3.12 через .venv_gpu
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent.parent
    
    # Определяем путь к Python в .venv_gpu
    if platform.system() == 'Windows':
        venv_python = project_root / '.venv_gpu' / 'Scripts' / 'python.exe'
    else:
        venv_python = project_root / '.venv_gpu' / 'bin' / 'python'
    
    if not venv_python.exists():
        print("[ERROR] .venv_gpu не найден или Python в нем недоступен")
        print("[INFO] Создайте .venv_gpu: python scripts/setup_python_gpu.py")
        return 1
    
    # Проверяем версию Python
    try:
        result = subprocess.run(
            [str(venv_python), '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"[OK] Python в .venv_gpu: {result.stdout.strip()}")
            if '3.12' not in result.stdout:
                print("[WARNING] Python в .venv_gpu не версия 3.12!")
        else:
            print("[ERROR] Python в .venv_gpu не отвечает")
            return 1
    except Exception as e:
        print(f"[ERROR] Ошибка проверки Python: {e}")
        return 1
    
    # Запускаем компиляцию
    compile_script = project_root / 'license_generator' / 'compile_all.py'
    
    print("=" * 80)
    print("КОМПИЛЯЦИЯ .pyc ФАЙЛОВ ДЛЯ PYTHON 3.12")
    print("=" * 80)
    print()
    print(f"[INFO] Используется: {venv_python}")
    print(f"[INFO] Скрипт: {compile_script}")
    print()
    
    result = subprocess.run(
        [str(venv_python), str(compile_script), '--version-only'],
        cwd=str(project_root),
        timeout=600
    )
    
    if result.returncode == 0:
        print()
        print("=" * 80)
        print("[OK] Компиляция завершена успешно!")
        print("=" * 80)
        return 0
    else:
        print()
        print("=" * 80)
        print("[ERROR] Ошибка компиляции!")
        print("=" * 80)
        return 1

if __name__ == '__main__':
    sys.exit(main())
