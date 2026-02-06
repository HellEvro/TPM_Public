#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест проверки HWID
"""

import sys
import os
from pathlib import Path

if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("ТЕСТ ПРОВЕРКИ HWID")
print("=" * 80)
print()

# 1. Получаем HWID через activate_premium.py способ
print("1. HWID через scripts/hardware_id.pyc (как activate_premium.py):")
try:
    import importlib.util
    hw_path = PROJECT_ROOT / 'scripts' / 'hardware_id.pyc'
    if hw_path.exists():
        spec = importlib.util.spec_from_file_location("hardware_id", hw_path)
        hw_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hw_module)
        short_hwid = hw_module.get_short_hardware_id()
        full_hwid = hw_module.get_hardware_id()
        print(f"   Short HWID: {short_hwid}")
        print(f"   Full HWID:  {full_hwid[:64]}...")
    else:
        print(f"   [ERROR] Файл не найден: {hw_path}")
except Exception as e:
    print(f"   [ERROR] Ошибка: {e}")
    import traceback
    traceback.print_exc()

print()

# 2. Проверяем через license_checker
print("2. HWID через license_checker:")
try:
    from bot_engine.ai.license_checker import get_license_checker
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    
    checker = get_license_checker(project_root=PROJECT_ROOT)
    valid, info = checker.check_license()
    
    print(f"   Valid: {valid}")
    if info:
        print(f"   Type: {info.get('type')}")
        print(f"   Expires: {info.get('expires_at')}")
except Exception as e:
    print(f"   [ERROR] Ошибка: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)

