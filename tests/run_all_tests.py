#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Запуск всех тестов
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Настройка кодировки для Windows
if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

print("=" * 80)
print("ЗАПУСК ВСЕХ ТЕСТОВ")
print("=" * 80)
print()

tests = [
    ("test_license_checker.py", "Тест проверки лицензии"),
    ("test_ai_system.py", "Тест запуска AI системы (15 сек)"),
    ("test_ui_config.py", "Тест изменения конфигов через API"),
]

results = []

for test_file, description in tests:
    print(f"\n{'=' * 80}")
    print(f"ТЕСТ: {description}")
    print(f"Файл: {test_file}")
    print(f"{'=' * 80}\n")
    
    if not Path(test_file).exists():
        print(f"[SKIP] Файл {test_file} не найден, пропускаем")
        results.append((test_file, "SKIPPED", "File not found"))
        continue
    
    try:
        # Запускаем тест с таймаутом
        if "ai_system" in test_file:
            # Для AI системы ограничиваем время
            result = subprocess.run(
                ["python", test_file],
                timeout=20,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
        else:
            result = subprocess.run(
                ["python", test_file],
                timeout=30,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"\n[OK] Тест {test_file} прошел успешно")
            results.append((test_file, "PASSED", ""))
        else:
            print(f"\n[FAIL] Тест {test_file} завершился с ошибкой (код: {result.returncode})")
            results.append((test_file, "FAILED", f"Exit code: {result.returncode}"))
    
    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT] Тест {test_file} превысил время ожидания")
        results.append((test_file, "TIMEOUT", ""))
    except Exception as e:
        print(f"\n[ERROR] Ошибка при запуске теста {test_file}: {e}")
        results.append((test_file, "ERROR", str(e)))

print()
print("=" * 80)
print("РЕЗУЛЬТАТЫ ТЕСТОВ")
print("=" * 80)
print()

for test_file, status, details in results:
    status_symbol = {
        "PASSED": "[OK]",
        "FAILED": "[FAIL]",
        "ERROR": "[ERROR]",
        "TIMEOUT": "[TIMEOUT]",
        "SKIPPED": "[SKIP]"
    }.get(status, "[?]")
    
    print(f"{status_symbol} {test_file}: {status}")
    if details:
        print(f"    {details}")

print()
passed = sum(1 for _, s, _ in results if s == "PASSED")
total = len(results)
print(f"Пройдено: {passed}/{total}")

if passed == total:
    print("[OK] ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
    sys.exit(0)
else:
    print("[FAIL] НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ")
    sys.exit(1)

