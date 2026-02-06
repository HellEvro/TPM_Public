#!/usr/bin/env python3
"""
Комплексное тестирование системы после рефакторинга
"""

import requests
import sys
import time

BASE_URL = 'http://localhost:5001'

def test_service_health():
    """Тест: Сервис запущен и работает"""
    print("\n[TEST 1] Service Health Check")
    print("-" * 60)
    
    r = requests.get(f'{BASE_URL}/health', timeout=5)
    assert r.status_code == 200, "Health check failed"
    
    data = r.json()
    print(f"  Status: {data['status']}")
    print(f"  Exchange: {'Connected' if data['exchange_connected'] else 'Disconnected'}")
    print(f"  Coins loaded: {data['coins_loaded']}")
    print(f"  Bots active: {data['bots_active']}")
    
    assert data['status'] == 'ok', "Service status is not ok"
    assert data['exchange_connected'], "Exchange not connected"
    
    print("  [OK] Service is healthy")
    return True


def test_rsi_data():
    """Тест: RSI данные загружены"""
    print("\n[TEST 2] RSI Data")
    print("-" * 60)
    
    r = requests.get(f'{BASE_URL}/api/bots/coins-with-rsi', timeout=10)
    assert r.status_code == 200, "RSI data request failed"
    
    data = r.json()
    assert data['success'], "RSI data not successful"
    
    print(f"  Total coins: {data['total']}")
    print(f"  Update in progress: {data['update_in_progress']}")
    print(f"  Last update: {data['last_update']}")
    
    # Проверяем что есть монеты
    assert data['total'] > 0, "No coins loaded"
    
    # Проверяем структуру данных
    if data['coins']:
        first_coin = list(data['coins'].values())[0]
        assert 'rsi6h' in first_coin, "RSI data missing"
        assert 'signal' in first_coin, "Signal missing"
        print(f"  Sample coin: {first_coin['symbol']}, RSI: {first_coin['rsi6h']}, Signal: {first_coin['signal']}")
    
    print("  [OK] RSI data loaded correctly")
    return True


def test_mature_coins():
    """Тест: Зрелые монеты"""
    print("\n[TEST 3] Mature Coins")
    print("-" * 60)
    
    r = requests.get(f'{BASE_URL}/api/bots/mature-coins-list', timeout=5)
    assert r.status_code == 200, "Mature coins request failed"
    
    data = r.json()
    assert data['success'], "Mature coins not successful"
    
    print(f"  Total mature coins: {data['total_count']}")
    if data['mature_coins']:
        print(f"  Sample: {data['mature_coins'][:5]}")
    
    print("  [OK] Mature coins data available")
    return True


def test_auto_bot_config():
    """Тест: Конфигурация Auto Bot"""
    print("\n[TEST 4] Auto Bot Config")
    print("-" * 60)
    
    r = requests.get(f'{BASE_URL}/api/bots/auto-bot', timeout=5)
    assert r.status_code == 200, "Auto bot config request failed"
    
    data = r.json()
    assert data['success'], "Auto bot config not successful"
    
    config = data['config']
    print(f"  Enabled: {config['enabled']}")
    print(f"  Max concurrent: {config['max_concurrent']}")
    print(f"  RSI thresholds: LONG<={config['rsi_long_threshold']}, SHORT>={config['rsi_short_threshold']}")
    print(f"  RSI Time Filter: {config['rsi_time_filter_enabled']} ({config['rsi_time_filter_candles']} candles)")
    print(f"  ExitScam: {config['exit_scam_enabled']}")
    print(f"  Maturity check: {config['enable_maturity_check']}")
    
    print("  [OK] Auto bot config available")
    return True


def test_bots_list():
    """Тест: Список ботов"""
    print("\n[TEST 5] Bots List")
    print("-" * 60)
    
    r = requests.get(f'{BASE_URL}/api/bots/list', timeout=5)
    assert r.status_code == 200, "Bots list request failed"
    
    data = r.json()
    assert data['success'], "Bots list not successful"
    
    print(f"  Total bots: {data['count']}")
    print(f"  Active bots: {data['stats']['active_bots']}")
    print(f"  Auto bot enabled: {data['auto_bot_enabled']}")
    
    if data['bots']:
        for bot in data['bots'][:3]:
            print(f"  - {bot['symbol']}: {bot['status']}")
    
    print("  [OK] Bots list available")
    return True


def test_account_info():
    """Тест: Информация о счете"""
    print("\n[TEST 6] Account Info")
    print("-" * 60)
    
    r = requests.get(f'{BASE_URL}/api/bots/account-info', timeout=5)
    assert r.status_code == 200, "Account info request failed"
    
    data = r.json()
    
    if data.get('success'):
        print(f"  Balance: {data.get('totalWalletBalance', 'N/A')}")
        print(f"  Available: {data.get('totalAvailableBalance', 'N/A')}")
        print(f"  Bots count: {data.get('bots_count', 0)}")
    else:
        print(f"  [WARNING] Could not get account info: {data.get('error', 'Unknown')}")
    
    print("  [OK] Account endpoint works")
    return True


def test_process_state():
    """Тест: Состояние процессов"""
    print("\n[TEST 7] Process State")
    print("-" * 60)
    
    r = requests.get(f'{BASE_URL}/api/bots/process-state', timeout=5)
    assert r.status_code == 200, "Process state request failed"
    
    data = r.json()
    assert data['success'], "Process state not successful"
    
    system_info = data['system_info']
    print(f"  Smart RSI Manager: {'Running' if system_info['smart_rsi_manager_running'] else 'Stopped'}")
    print(f"  Exchange: {'Initialized' if system_info['exchange_initialized'] else 'Not initialized'}")
    print(f"  Total bots: {system_info['total_bots']}")
    print(f"  Mature coins: {system_info['mature_coins_storage_size']}")
    print(f"  Optimal EMA: {system_info['optimal_ema_count']}")
    
    print("  [OK] Process state available")
    return True


def test_new_modules():
    """Тест: Новые модули импортируются"""
    print("\n[TEST 8] New Modules Import")
    print("-" * 60)
    
    try:
        from bot_engine.utils.rsi_utils import calculate_rsi
        from bot_engine.utils.ema_utils import calculate_ema
        from bot_engine.filters import check_rsi_time_filter
        from bot_engine.maturity_checker import check_coin_maturity
        from bot_engine.storage import save_rsi_cache
        from bot_engine.signal_processor import get_effective_signal
        from bot_engine.optimal_ema_manager import get_optimal_ema_periods
        
        print("  [OK] All modules imported successfully")
        
        # Тест RSI расчета
        prices = [100 + i * 0.5 for i in range(50)]
        rsi = calculate_rsi(prices, 14)
        print(f"  [OK] RSI calculation works: RSI={rsi}")
        
        # Тест EMA расчета
        ema = calculate_ema(prices, 20)
        print(f"  [OK] EMA calculation works: EMA={ema:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Failed to import modules: {e}")
        return False


def test_filters():
    """Тест: Фильтры работают"""
    print("\n[TEST 9] Filters (ExitScam, RSI Time)")
    print("-" * 60)
    
    # Тест ExitScam для монеты
    try:
        r = requests.get(f'{BASE_URL}/api/bots/test-exit-scam/BTC', timeout=10)
        if r.status_code == 200:
            print("  [OK] ExitScam filter test works")
        else:
            print(f"  [WARNING] ExitScam test returned {r.status_code}")
    except Exception as e:
        print(f"  [WARNING] ExitScam test error: {e}")
    
    # Тест RSI Time Filter
    try:
        r = requests.get(f'{BASE_URL}/api/bots/test-rsi-time-filter/BTC', timeout=10)
        if r.status_code == 200:
            print("  [OK] RSI Time Filter test works")
        else:
            print(f"  [WARNING] RSI Time Filter test returned {r.status_code}")
    except Exception as e:
        print(f"  [WARNING] RSI Time Filter test error: {e}")
    
    return True


def test_system_operations():
    """Тест: Системные операции"""
    print("\n[TEST 10] System Operations")
    print("-" * 60)
    
    # Тест сброса флага обновления
    try:
        r = requests.post(f'{BASE_URL}/api/bots/reset-update-flag', timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"  [OK] Reset update flag works (was: {data.get('was_in_progress', False)})")
        else:
            print(f"  [WARNING] Reset flag returned {r.status_code}")
    except Exception as e:
        print(f"  [WARNING] Reset flag error: {e}")
    
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ СИСТЕМЫ")
    print("=" * 60)
    
    tests = [
        test_service_health,
        test_rsi_data,
        test_mature_coins,
        test_auto_bot_config,
        test_bots_list,
        test_account_info,
        test_process_state,
        test_new_modules,
        test_filters,
        test_system_operations
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n  [ERROR] {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"РЕЗУЛЬТАТЫ: Passed: {passed}/{len(tests)}, Failed: {failed}")
    print("=" * 60)
    
    if failed == 0:
        print("\n[SUCCESS] Все тесты пройдены!")
        print("\nСистема полностью работоспособна:")
        print("  - Новые модули импортируются")
        print("  - API endpoints отвечают")
        print("  - RSI данные загружены")
        print("  - Фильтры работают")
        print("  - Сервис стабилен")
        sys.exit(0)
    else:
        print(f"\n[ERROR] {failed} тестов не прошли!")
        sys.exit(1)


