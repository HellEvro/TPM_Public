#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для проверки наличия и совместимости .pyc файлов.
Автоматически перекомпилирует файлы, если они отсутствуют или несовместимы.
"""

import sys
import os
from pathlib import Path

def check_pyc_files():
    """Проверяет наличие и совместимость .pyc файлов"""
    project_root = Path(__file__).resolve().parent.parent
    ai_dir = project_root / 'bot_engine' / 'ai'
    target_dir = ai_dir
    
    required_files = [
        '_ai_launcher.pyc',
        'ai_manager.pyc',
        'license_checker.pyc',
        'hardware_id_source.pyc'
    ]
    
    missing_files = []
    incompatible_files = []
    
    print("Проверка .pyc файлов...")
    print(f"Целевая директория: {target_dir}")
    print()
    
    for filename in required_files:
        filepath = target_dir / filename
        if not filepath.exists():
            missing_files.append(filename)
            print(f"[ERROR] Отсутствует: {filename}")
        else:
            # Проверяем совместимость
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location('test', str(filepath))
                if spec is None:
                    incompatible_files.append(filename)
                    print(f"[WARNING] Несовместим: {filename}")
                else:
                    print(f"[OK] Найден: {filename}")
            except Exception as e:
                err_msg = str(e).lower()
                if "bad magic number" in err_msg or "bad magic" in err_msg:
                    incompatible_files.append(filename)
                    print(f"[WARNING] Несовместим (bad magic): {filename}")
                else:
                    print(f"[OK] Найден: {filename}")
    
    print()
    
    if missing_files or incompatible_files:
        print("=" * 60)
        print("ОБНАРУЖЕНЫ ПРОБЛЕМЫ С .pyc ФАЙЛАМИ")
        print("=" * 60)
        
        if missing_files:
            print(f"Отсутствующие файлы: {', '.join(missing_files)}")
        
        if incompatible_files:
            print(f"Несовместимые файлы: {', '.join(incompatible_files)}")
        
        print()
        print("Попытка автоматической перекомпиляции...")
        print()
        
        # Запускаем компиляцию
        compile_script = project_root / 'license_generator' / 'compile_all.py'
        if compile_script.exists():
            import subprocess
            try:
                result = subprocess.run(
                    [sys.executable, str(compile_script)],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0:
                    print("[OK] Компиляция завершена успешно")
                    return True
                else:
                    print(f"[ERROR] Ошибка компиляции:")
                    print(result.stderr)
                    return False
            except Exception as e:
                print(f"[ERROR] Ошибка при запуске компиляции: {e}")
                return False
        else:
            print(f"[ERROR] Скрипт компиляции не найден: {compile_script}")
            return False
    else:
        print("=" * 60)
        print("[OK] ВСЕ .pyc ФАЙЛЫ НА МЕСТЕ И СОВМЕСТИМЫ")
        print("=" * 60)
        return True

if __name__ == '__main__':
    success = check_pyc_files()
    sys.exit(0 if success else 1)
