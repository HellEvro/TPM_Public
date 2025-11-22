#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест детального логирования AI конфигурации
"""

import requests
import json

def test_ai_config_logging():
    """Тестирует детальное логирование изменений AI конфигурации"""
    
    url = "http://localhost:5001/api/ai/config"
    
    print("Тестирование детального логирования AI конфигурации...")
    print("=" * 60)
    
    # Тест 1: Изменение порога блокировки аномалий
    print("Тест 1: Изменение порога блокировки аномалий")
    data = {
        "anomaly_block_threshold": 0.75
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"Успешно: {result.get('message', 'OK')}")
        else:
            print(f"Ошибка: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Исключение: {e}")
    
    print()
    
    # Тест 2: Изменение интервала обновления рисков
    print("Тест 2: Изменение интервала обновления рисков")
    data = {
        "risk_update_interval": 600
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"Успешно: {result.get('message', 'OK')}")
        else:
            print(f"Ошибка: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Исключение: {e}")
    
    print()
    
    # Тест 3: Изменение нескольких параметров одновременно
    print("Тест 3: Изменение нескольких параметров одновременно")
    data = {
        "anomaly_block_threshold": 0.7,
        "risk_update_interval": 300,
        "retrain_hour": 4
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"Успешно: {result.get('message', 'OK')}")
        else:
            print(f"Ошибка: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Исключение: {e}")
    
    print()
    print("Проверьте логи ботов на предмет детального логирования изменений!")
    print("   Должны появиться сообщения вида:")
    print("   [AI_CONFIG] Порог блокировки аномалий: 0.7 -> 0.75")
    print("   [AI_CONFIG] Интервал обновления рисков (сек): 300 -> 600")

if __name__ == "__main__":
    test_ai_config_logging()
