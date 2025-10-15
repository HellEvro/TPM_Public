#!/usr/bin/env python3
"""
Скрипт для автоматического форматирования Python файлов
Использует autopep8 для исправления отступов и других проблем PEP8
"""

import subprocess
import sys

def format_file(filename):
    """Форматирует файл с помощью autopep8"""
    try:
        print(f"[FORMAT] Processing {filename}...")
        
        # Используем autopep8 с агрессивным режимом для исправления отступов
        result = subprocess.run(
            ['autopep8', '--in-place', '--aggressive', '--aggressive', filename],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"[OK] {filename} formatted successfully")
            return True
        else:
            print(f"[ERROR] Failed to format {filename}")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

if __name__ == '__main__':
    files_to_format = ['bots.py']
    
    print("=" * 60)
    print("АВТОФОРМАТИРОВАНИЕ PYTHON КОДА")
    print("=" * 60)
    
    success_count = 0
    for filename in files_to_format:
        if format_file(filename):
            success_count += 1
    
    print("=" * 60)
    print(f"[DONE] Formatted: {success_count}/{len(files_to_format)} files")
    print("=" * 60)

