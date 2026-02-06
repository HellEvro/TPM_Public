#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для компиляции .pyc файлов для Python 3.12
Использует .venv_gpu если доступен, иначе ищет глобальный Python 3.12
"""

import sys
import subprocess
import os
import platform
from pathlib import Path

def find_python312():
    """Находит Python 3.12"""
    project_root = Path(__file__).resolve().parent.parent
    
    # Сначала проверяем .venv_gpu
    venv_gpu_path = project_root / '.venv_gpu'
    if venv_gpu_path.exists():
        if platform.system() == 'Windows':
            venv_python = venv_gpu_path / 'Scripts' / 'python.exe'
        else:
            venv_python = venv_gpu_path / 'bin' / 'python'
        
        if venv_python.exists():
            try:
                result = subprocess.run(
                    [str(venv_python), '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and '3.12' in (result.stdout or '') + (result.stderr or ''):
                    print(f"[OK] Найден Python 3.12 в .venv_gpu: {venv_python}")
                    return str(venv_python)
            except Exception as e:
                print(f"[WARNING] Ошибка проверки Python в .venv_gpu: {e}")
    
    # Пробуем глобальные команды
    commands = []
    if platform.system() == 'Windows':
        commands = [['py', '-3.12'], ['python3.12'], ['python312']]
    else:
        commands = [['python3.12'], ['python312']]
    
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd + ['--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and '3.12' in (result.stdout or '') + (result.stderr or ''):
                print(f"[OK] Найден Python 3.12: {' '.join(cmd)}")
                return cmd
        except:
            continue
    
    return None

def main():
    print("=" * 80)
    print("КОМПИЛЯЦИЯ .pyc ФАЙЛОВ ДЛЯ PYTHON 3.12")
    print("=" * 80)
    print()
    
    python312 = find_python312()
    if not python312:
        print("[ERROR] Python 3.12 не найден!")
        print("[INFO] Попробуйте:")
        print("  1. Пересоздать .venv_gpu: python scripts/setup_python_gpu.py")
        print("  2. Установить Python 3.12 вручную")
        return 1
    
    project_root = Path(__file__).resolve().parent.parent
    compile_script = project_root / 'license_generator' / 'compile_all.py'
    
    # Запускаем компиляцию
    if isinstance(python312, list):
        cmd = python312 + [str(compile_script), '--version-only']
    else:
        cmd = [python312, str(compile_script), '--version-only']
    
    print(f"[INFO] Запуск компиляции: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(
        cmd,
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
