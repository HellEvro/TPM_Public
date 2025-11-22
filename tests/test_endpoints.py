#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тест новых эндпоинтов"""

import requests
import time

# Ждем 5 секунд для запуска сервера
print("⏳ Ожидание запуска сервера (5 секунд)...")
time.sleep(5)

endpoints = {
    'health': 'http://127.0.0.1:5001/api/bots/health',
    'status': 'http://127.0.0.1:5001/api/bots/status',
    'pairs': 'http://127.0.0.1:5001/api/bots/pairs',
    'sync-positions (GET)': 'http://127.0.0.1:5001/api/bots/sync-positions'
}

print('\n=== ТЕСТ НОВЫХ ЭНДПОИНТОВ ===')
for name, url in endpoints.items():
    try:
        response = requests.get(url, timeout=4)
        status = '✅ OK' if response.status_code == 200 else f'❌ FAILED ({response.status_code})'
        print(f'{name}: {status}')
        if response.status_code == 200:
            data = response.json()
            print(f'  └─ Response: {str(data)[:100]}...')
    except Exception as e:
        print(f'{name}: ❌ ERROR ({e})')

print('\n✅ Тест завершен!')

