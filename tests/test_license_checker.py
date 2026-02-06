#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест проверки лицензии для ai.py
"""

import sys
import os
from pathlib import Path

# Настройка кодировки для Windows
if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("ТЕСТ ПРОВЕРКИ ЛИЦЕНЗИИ")
print("=" * 80)
print()

# 1. Проверяем наличие файла лицензии
root = Path.cwd()
lic_files = [f for f in os.listdir(root) if f.endswith('.lic')]
print(f"1. Корень проекта: {root}")
print(f"2. Файлы лицензии: {lic_files}")
if not lic_files:
    print("[ERROR] Файл лицензии не найден!")
    sys.exit(1)
print(f"[OK] Файл лицензии найден: {lic_files[0]}")
print()

# 2. Тестируем license_checker
print("3. Тестируем license_checker...")
try:
    from bot_engine.ai.license_checker import get_license_checker
    checker = get_license_checker(project_root=root)
    print(f"   [OK] LicenseChecker создан")
    print(f"   [INFO] Project root: {checker.project_root}")
    
    # Проверяем лицензию
    valid, info = checker.check_license()
    print(f"   [OK] check_license() вызван")
    print(f"   [INFO] Valid: {valid}")
    print(f"   [INFO] Info: {info}")
    
    if valid:
        print(f"   [OK] ЛИЦЕНЗИЯ ВАЛИДНА!")
        print(f"   [INFO] Тип: {info.get('type', 'N/A')}")
        print(f"   [INFO] Действительна до: {info.get('expires_at', 'N/A')}")
        print(f"   [INFO] Функции: {info.get('features', {})}")
    else:
        print(f"   [ERROR] ЛИЦЕНЗИЯ НЕ ВАЛИДНА!")
        print(f"   [WARN] Проверьте файл лицензии и HWID")
    
    # Проверяем методы
    is_valid = checker.is_valid()
    info_dict = checker.get_info()
    print(f"   [OK] is_valid(): {is_valid}")
    print(f"   [OK] get_info(): {info_dict}")
    
except Exception as e:
    print(f"   [ERROR] ОШИБКА: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 80)
print("ТЕСТ ЗАВЕРШЕН")
print("=" * 80)

