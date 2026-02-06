#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест изменения конфигурации Auto Bot через REST API бота.
"""

import os
import sys
import time
import requests

if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

APP_URL = os.getenv('APP_URL', 'http://127.0.0.1:5000')
BOTS_URL = os.getenv('BOTS_SERVICE_URL', 'http://127.0.0.1:5001')
HEALTH_ENDPOINT = f"{BOTS_URL}/api/bots/health"
CONFIG_ENDPOINT = f"{BOTS_URL}/api/bots/auto-bot"

print('=' * 80)
print('ТЕСТ ИЗМЕНЕНИЯ КОНФИГА ЧЕРЕЗ API БОТА')
print('=' * 80)
print()

print(f'1. Проверяем доступность сервиса бота: {HEALTH_ENDPOINT}')
try:
    response = requests.get(HEALTH_ENDPOINT, timeout=5)
    if response.status_code == 200:
        print('   [OK] Bots service доступен')
    else:
        print(f'   [WARN] health ответил кодом {response.status_code}: {response.text}')
except requests.RequestException as exc:
    print(f'   [ERROR] Не удалось подключиться к bots service: {exc}')
    sys.exit(1)

print()
print('2. Загружаем текущую конфигурацию Auto Bot...')
try:
    response = requests.get(CONFIG_ENDPOINT, timeout=5)
    if response.status_code != 200:
        print(f'   [ERROR] Не удалось получить конфигурацию: {response.status_code} {response.text}')
        sys.exit(1)
    payload = response.json()
    auto_config = payload.get('config') or payload.get('autoBot') or {}
    if not auto_config:
        print('   [ERROR] Конфигурация пуста в ответе')
        sys.exit(1)
    print('   [OK] Конфигурация получена')
    original_break_even = auto_config.get('break_even_trigger_percent', 20.0)
    print(f'   [INFO] Текущее значение break_even_trigger_percent = {original_break_even}')
except Exception as exc:
    print(f'   [ERROR] Ошибка чтения конфигурации: {exc}')
    sys.exit(1)

new_value = 25.0 if abs(original_break_even - 25.0) > 1e-3 else 30.0

print()
print(f'3. Отправляем обновление break_even_trigger_percent → {new_value}...')
try:
    response = requests.post(
        CONFIG_ENDPOINT,
        json={'break_even_trigger_percent': new_value},
        headers={'Content-Type': 'application/json'},
        timeout=5
    )
    if response.status_code != 200:
        print(f'   [ERROR] Не удалось обновить конфигурацию: {response.status_code} {response.text}')
        sys.exit(1)
    print('   [OK] Конфигурация обновлена')
except Exception as exc:
    print(f'   [ERROR] Ошибка при отправке обновления: {exc}')
    sys.exit(1)

print()
print('4. Проверяем, что изменение применилось...')
try:
    time.sleep(1)
    response = requests.get(CONFIG_ENDPOINT, timeout=5)
    if response.status_code != 200:
        print(f'   [ERROR] Не удалось перечитать конфигурацию: {response.status_code} {response.text}')
        sys.exit(1)
    auto_config = (response.json().get('config') or response.json().get('autoBot') or {})
    updated_value = auto_config.get('break_even_trigger_percent')
    print(f'   [INFO] Значение после обновления: {updated_value}')
    if abs(updated_value - new_value) > 1e-3:
        print('   [ERROR] Значение не соответствует ожидаемому')
        sys.exit(1)
    print('   [OK] Изменение подтверждено')
except Exception as exc:
    print(f'   [ERROR] Ошибка проверки конфигурации: {exc}')
    sys.exit(1)

print()
print('5. Восстанавливаем исходное значение...')
try:
    response = requests.post(
        CONFIG_ENDPOINT,
        json={'break_even_trigger_percent': original_break_even},
        headers={'Content-Type': 'application/json'},
        timeout=5
    )
    if response.status_code == 200:
        print('   [OK] Исходное значение восстановлено')
    else:
        print(f'   [WARN] Не удалось восстановить: {response.status_code} {response.text}')
except Exception as exc:
    print(f'   [WARN] Ошибка при восстановлении: {exc}')

print('\nИТОГ: тест завершен успешно')
print('=' * 80)
