#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для компиляции всех .pyc файлов под Python 3.14
Запускает компиляцию через Python 3.14 явно
"""

import sys
import subprocess
import os
from pathlib import Path

def find_python314():
    """Находит Python 3.14 в системе"""
    # Варианты команд для поиска Python 3.14
    commands = [
        ['py', '-3.14'],
        ['python3.14'],
        ['python314'],
        ['python', '--version'],  # Проверяем текущий Python
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd + ['--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version_output = result.stdout.strip()
                if '3.14' in version_output:
                    print(f"[OK] Найден Python 3.14: {' '.join(cmd)}")
                    print(f"     Версия: {version_output}")
                    return cmd
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            continue
    
    return None

def main():
    """Главная функция"""
    print("=" * 60)
    print("КОМПИЛЯЦИЯ ВСЕХ МОДУЛЕЙ ПОД PYTHON 3.14")
    print("=" * 60)
    print()
    
    # Находим Python 3.14
    python314 = find_python314()
    if not python314:
        print("[ERROR] Python 3.14 не найден!")
        print()
        print("Пожалуйста, установите Python 3.14:")
        print("  https://www.python.org/downloads/")
        print()
        print("Или используйте py launcher:")
        print("  py -3.14 --version")
        return False
    
    # Переходим в корень проекта
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    print(f"[INFO] Рабочая директория: {project_root}")
    print()
    
    # Компилируем все модули через Python 3.14
    compile_script = project_root / 'license_generator' / 'compile_all.py'
    
    print("[STEP 1] Компиляция hardware_id_source...")
    result1 = subprocess.run(
        python314 + [str(compile_script), '--no-python314'],
        cwd=project_root
    )
    
    if result1.returncode != 0:
        print("[ERROR] Ошибка компиляции hardware_id_source")
        return False
    
    print()
    print("[STEP 2] Компиляция license_checker...")
    # license_checker компилируется в том же скрипте
    
    print()
    print("[STEP 3] Компиляция ai_manager...")
    # ai_manager компилируется в том же скрипте
    
    print()
    print("[STEP 4] Компиляция _ai_launcher...")
    # _ai_launcher компилируется в том же скрипте
    
    print()
    print("=" * 60)
    print("[OK] ВСЕ МОДУЛИ СКОМПИЛИРОВАНЫ ПОД PYTHON 3.14")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
