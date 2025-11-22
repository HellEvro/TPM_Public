#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Автотест запуска и остановки AISystem без тяжёлых фоновых задач.
"""

import os
import sys
import time
from pathlib import Path

if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

print('=' * 80)
print('ТЕСТ ЗАПУСКА/ОСТАНОВКИ AI СИСТЕМЫ (LIGHT MODE)')
print('=' * 80)
print()

try:
    from ai import AISystem
except ImportError as err:
    print(f'[ERROR] Не удалось импортировать AISystem: {err}')
    sys.exit(1)

minimal_config = {
    'enabled': True,
    'enable_data_service': False,
    'enable_training': False,
    'enable_backtest': False,
    'enable_optimizer': False,
}

ai_system = None

try:
    print('1. Создаем AI систему в легком режиме...')
    ai_system = AISystem(config=minimal_config)
    print('   [OK] Экземпляр создан')
    print(f'   [INFO] Лицензия валидна: {ai_system.license_valid}')

    if not ai_system.license_valid:
        print('   [ERROR] Лицензия не валидна, тест прерван')
        sys.exit(1)

    print('\n2. Запускаем систему...')
    ai_system.start()
    print('   [OK] Старт выполнен')

    print('\n3. Проверяем, что система активна...')
    if not ai_system.running:
        print('   [ERROR] Флаг running не установлен после старта')
        sys.exit(1)
    print('   [OK] Флаг running установлен')

    print('\n4. Ждем 2 секунды (имитация работы)...')
    time.sleep(2)

    print('\n5. Останавливаем систему...')
    ai_system.stop()
    print('   [OK] Система остановлена')

    if ai_system.running:
        print('   [ERROR] Флаг running остался True после остановки')
        sys.exit(1)

    print('\n[OK] Тест успешно завершен')
except Exception as exc:
    print(f'   [ERROR] Исключение: {exc}')
    raise
finally:
    if ai_system and ai_system.running:
        ai_system.stop()

print('\n' + '=' * 80)
print('ТЕСТ ЗАВЕРШЕН')
print('=' * 80)
