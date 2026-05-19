#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Автоматизированное тестирование системы торговых ботов
"""

import requests
import time
import json
import sys
import io
from datetime import datetime

# Исправление кодировки для Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_URL = "http://localhost:5001"
COLORS = {
    'green': '\033[92m',
    'red': '\033[91m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'end': '\033[0m'
}

def log_test(name, status, message=""):
    """Логирование результата теста"""
    color = COLORS['green'] if status else COLORS['red']
    symbol = "✅" if status else "❌"
    print(f"{color}{symbol} Test {name}: {'PASSED' if status else 'FAILED'}{COLORS['end']}")
    if message:
        print(f"   {message}")

def test_1_api_status():
    """Тест 1: Проверка статуса API"""
    try:
        r = requests.get(f"{BASE_URL}/api/status", timeout=5)
        assert r.status_code == 200, "Status code not 200"
        
        data = r.json()
        assert data.get('status') == 'online', "Service not online"
        assert data.get('success') == True, "Success flag not True"
        
        log_test("1", True, f"Status: {data.get('status')}, Bots: {data.get('bots_count')}")
        return True
    except Exception as e:
        log_test("1", False, str(e))
        return False

def test_2_auto_bot_disabled():
    """Тест 2: Автобот выключен при запуске"""
    try:
        r = requests.get(f"{BASE_URL}/api/bots/auto-bot", timeout=5)
        assert r.status_code == 200
        
        data = r.json()
        assert data.get('success') == True
        assert data['data'].get('enabled') == False, "Auto Bot should be disabled on startup"
        
        log_test("2", True, "Auto Bot отключен при запуске")
        return True
    except Exception as e:
        log_test("2", False, str(e))
        return False

def test_3_bots_list():
    """Тест 3: Получение списка ботов"""
    try:
        r = requests.get(f"{BASE_URL}/api/bots/list", timeout=5)
        assert r.status_code == 200
        
        data = r.json()
        assert 'data' in data
        assert isinstance(data['data'], list)
        
        log_test("3", True, f"Активных ботов: {len(data['data'])}")
        return True
    except Exception as e:
        log_test("3", False, str(e))
        return False

def test_4_coins_with_rsi():
    """Тест 4: Монеты с RSI данными"""
    try:
        r = requests.get(f"{BASE_URL}/api/bots/coins-with-rsi", timeout=10)
        assert r.status_code == 200
        
        data = r.json()
        assert 'coins' in data
        assert 'count' in data
        
        log_test("4", True, f"Монет с RSI: {data['count']}")
        return True
    except Exception as e:
        log_test("4", False, str(e))
        return False

def test_5_system_initialized():
    """Тест 5: Система полностью инициализирована"""
    try:
        r = requests.get(f"{BASE_URL}/api/status", timeout=5)
        data = r.json()
        
        # Проверяем что система работает и отвечает
        assert data.get('status') == 'online'
        
        # Проверяем что есть timestamp (система инициализирована)
        assert 'timestamp' in data
        
        log_test("5", True, "Система полностью инициализирована")
        return True
    except Exception as e:
        log_test("5", False, str(e))
        return False

def test_6_maturity_check_enabled():
    """Тест 6: Проверка зрелости включена"""
    try:
        r = requests.get(f"{BASE_URL}/api/bots/auto-bot", timeout=5)
        data = r.json()
        
        config = data['data']
        enable_maturity = config.get('enable_maturity_check', False)
        
        assert enable_maturity == True, "Maturity check should be enabled"
        
        log_test("6", True, "Проверка зрелости включена")
        return True
    except Exception as e:
        log_test("6", False, str(e))
        return False

def test_7_sync_positions():
    """Тест 7: Синхронизация позиций"""
    try:
        r = requests.post(f"{BASE_URL}/api/bots/sync-positions", timeout=10)
        assert r.status_code == 200
        
        data = r.json()
        assert data.get('success') == True
        
        log_test("7", True, "Синхронизация позиций работает")
        return True
    except Exception as e:
        log_test("7", False, str(e))
        return False

def test_8_process_state_endpoint():
    """Тест 8: Дополнительная проверка process-state"""
    try:
        r = requests.get(f"{BASE_URL}/api/bots/process-state", timeout=5)
        assert r.status_code == 200
        
        data = r.json()
        log_test("8", True, f"Process state success: {data.get('success', False)}")
        return True
    except Exception as e:
        log_test("8", False, str(e))
        return False

def run_all_tests():
    """Запуск всех тестов"""
    print(f"\n{COLORS['blue']}{'='*60}{COLORS['end']}")
    print(f"{COLORS['blue']}🧪 АВТОМАТИЗИРОВАННОЕ ТЕСТИРОВАНИЕ СИСТЕМЫ{COLORS['end']}")
    print(f"{COLORS['blue']}{'='*60}{COLORS['end']}\n")
    
    print(f"⏰ Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 URL: {BASE_URL}\n")
    
    tests = [
        ("API Status", test_1_api_status),
        ("Auto Bot Disabled", test_2_auto_bot_disabled),
        ("Bots List", test_3_bots_list),
        ("Coins with RSI", test_4_coins_with_rsi),
        ("System Initialized", test_5_system_initialized),
        ("Maturity Check Enabled", test_6_maturity_check_enabled),
        ("Sync Positions", test_7_sync_positions),
        ("Process State Endpoint", test_8_process_state_endpoint),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"{COLORS['red']}❌ Exception in {name}: {e}{COLORS['end']}")
            results.append(False)
        
        time.sleep(0.5)  # Небольшая задержка между тестами
    
    # Итоги
    print(f"\n{COLORS['blue']}{'='*60}{COLORS['end']}")
    passed = sum(results)
    total = len(results)
    percentage = (passed / total * 100) if total > 0 else 0
    
    if passed == total:
        print(f"{COLORS['green']}🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! {passed}/{total} (100%){COLORS['end']}")
        return 0
    else:
        print(f"{COLORS['yellow']}⚠️ Пройдено: {passed}/{total} ({percentage:.1f}%){COLORS['end']}")
        print(f"{COLORS['red']}❌ Провалено: {total - passed}{COLORS['end']}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = run_all_tests()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{COLORS['yellow']}⏸️ Тестирование прервано пользователем{COLORS['end']}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{COLORS['red']}💥 Критическая ошибка: {e}{COLORS['end']}")
        sys.exit(1)

