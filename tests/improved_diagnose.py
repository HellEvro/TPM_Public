#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшенная диагностика с таймаутами
Запуск: python improved_diagnose.py
Цель: дать серверу время на запуск, затем проверить в течение 5 минут
"""

import json
import socket
import sys
import time
from typing import Tuple, Optional

import requests

UI = "http://127.0.0.1:5000"
API = "http://127.0.0.1:5001"  # сервис ботов (bots.py)

TIMEOUT = 4
STARTUP_WAIT = 60  # Ждем 1 минуту на запуск сервера
TEST_DURATION = 300  # Тестируем 5 минут


def ping(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def get(url: str, expect_json: bool = True) -> Tuple[int, Optional[dict], Optional[str]]:
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if expect_json:
            try:
                return r.status_code, r.json(), None
            except Exception as je:
                return r.status_code, None, f"JSON decode error: {je}"
        return r.status_code, None, r.text
    except Exception as e:
        return -1, None, str(e)


def check_endpoint(name: str, url: str, expect_json: bool = True):
    code, js, txt = get(url, expect_json=expect_json)
    status = "OK" if code == 200 else f"ERR({code})"
    print(f"[{name}] {status} → {url}")
    if js is not None:
        sample = json.dumps(js, ensure_ascii=False)[:400]
        print(f"  JSON: {sample}{'...' if len(sample)==400 else ''}")
    if txt:
        clip = (txt or "")[:200]
        print(f"  TEXT: {clip}{'...' if len(clip)==200 else ''}")
    print()


def wait_for_server_startup():
    """Ждем пока сервер запустится"""
    print(f"=== Ожидание запуска сервера ({STARTUP_WAIT} секунд) ===")
    
    for i in range(STARTUP_WAIT):
        ui_up = ping('127.0.0.1', 5000)
        api_up = ping('127.0.0.1', 5001)
        
        if ui_up and api_up:
            print(f"✅ Серверы запущены через {i+1} секунд")
            return True
            
        if i % 10 == 0:  # Каждые 10 секунд показываем статус
            print(f"⏳ {i+1}/{STARTUP_WAIT}с - UI:{'UP' if ui_up else 'DOWN'}, API:{'UP' if api_up else 'DOWN'}")
        
        time.sleep(1)
    
    print(f"⚠️ Серверы не запустились за {STARTUP_WAIT} секунд")
    return False


def check_direct_api():
    """Проверяем прямые вызовы к API сервиса ботов"""
    print("=== Bots Service API (прямые вызовы на порт 5001) ===")
    api_checks = {
        "account-info": f"{API}/api/bots/account-info",
        "coins-with-rsi": f"{API}/api/bots/coins-with-rsi",
        "health": f"{API}/api/bots/health",
        "status": f"{API}/api/bots/status",
        "pairs": f"{API}/api/bots/pairs",
    }
    for name, url in api_checks.items():
        check_endpoint(name, url)


def check_ui_proxies():
    """Проверяем прокси через UI"""
    print("=== UI proxy endpoints (через app.py → порт 5000) ===")
    ui_checks = {
        "account-info (proxy)": f"{UI}/api/bots/account-info",
        "coins-with-rsi (proxy)": f"{UI}/api/bots/coins-with-rsi",
        "sync-positions (proxy)": f"{UI}/api/bots/sync-positions",
    }
    for name, url in ui_checks.items():
        check_endpoint(name, url)


def check_stochastic_presence():
    """Проверяем наличие стохастика у монет"""
    print("=== Проверка наличия стохастика у монет (stoch_rsi_k / stoch_rsi_d) ===")
    code, js, txt = get(f"{API}/api/bots/coins-with-rsi")
    if code != 200 or not isinstance(js, dict):
        print(f"НЕТ данных с {API}/api/bots/coins-with-rsi → code={code}, err={txt}")
        return
    
    coins = js.get("coins") or {}
    if not coins:
        print("coins пустой — сервис вернул 0 монет.")
        return
    
    total = len(coins)
    with_stoch = 0
    examples_missing = []
    examples_with = []
    
    for i, (symbol, data) in enumerate(coins.items()):
        if data.get("stoch_rsi_k") is not None or data.get("stoch_rsi_d") is not None:
            with_stoch += 1
            if len(examples_with) < 3:
                examples_with.append(symbol)
        else:
            if len(examples_missing) < 3:
                examples_missing.append(symbol)
        
        if i > 500:  # достаточно сэмпла
            break
    
    print(f"Монет всего: {total}")
    print(f"С StochRSI: {with_stoch} (примеры: {examples_with})")
    print(f"Без StochRSI: {total - with_stoch} (примеры: {examples_missing})")
    
    # Проверяем оптимальные EMA
    with_ema = 0
    examples_ema_missing = []
    examples_ema_with = []
    
    for i, (symbol, data) in enumerate(coins.items()):
        if data.get("ema_short") is not None and data.get("ema_long") is not None:
            with_ema += 1
            if len(examples_ema_with) < 3:
                examples_ema_with.append(symbol)
        else:
            if len(examples_ema_missing) < 3:
                examples_ema_missing.append(symbol)
        
        if i > 500:
            break
    
    print(f"С Optimal EMA: {with_ema} (примеры: {examples_ema_with})")
    print(f"Без Optimal EMA: {total - with_ema} (примеры: {examples_ema_missing})")


def run_diagnosis():
    """Запускаем диагностику в течение ограниченного времени"""
    print("=== Диагностика системы торговых ботов ===")
    print(f"⏰ Время тестирования: {TEST_DURATION} секунд")
    print(f"⏰ Ожидание запуска: {STARTUP_WAIT} секунд")
    print()
    
    # 1) Ждем запуска сервера
    if not wait_for_server_startup():
        print("❌ Серверы не запустились, завершаем диагностику")
        return
    
    # 2) Проводим диагностику в течение ограниченного времени
    start_time = time.time()
    test_count = 0
    
    while time.time() - start_time < TEST_DURATION:
        test_count += 1
        elapsed = int(time.time() - start_time)
        remaining = TEST_DURATION - elapsed
        
        print(f"\n=== Тест #{test_count} (прошло: {elapsed}с, осталось: {remaining}с) ===")
        
        # Проверяем порты
        ui_up = ping('127.0.0.1', 5000)
        api_up = ping('127.0.0.1', 5001)
        print(f"Порты: UI(5000): {'UP' if ui_up else 'DOWN'}, API(5001): {'UP' if api_up else 'DOWN'}")
        
        if not (ui_up and api_up):
            print("⚠️ Серверы недоступны, ждем...")
            time.sleep(10)
            continue
        
        # Проверяем API
        check_direct_api()
        
        # Проверяем UI прокси
        check_ui_proxies()
        
        # Проверяем стохастик
        check_stochastic_presence()
        
        # Проверяем типичные проблемы
        print("=== Диагностика причин пустого UI ===")
        hints = []
        
        # a) Нет прокси эндпоинтов
        code, _, _ = get(f"{UI}/api/bots/account-info")
        if code != 200:
            hints.append("В app.py отсутствует или падает прокси /api/bots/account-info")
        
        code, _, _ = get(f"{UI}/api/bots/coins-with-rsi")
        if code != 200:
            hints.append("В app.py отсутствует или падает прокси /api/bots/coins-with-rsi")
        
        # b) У самого сервиса нет эндпоинтов
        code, _, _ = get(f"{API}/api/bots/coins-with-rsi")
        if code != 200:
            hints.append("В сервисе ботов отсутствует /api/bots/coins-with-rsi")
        
        code, js, _ = get(f"{API}/api/bots/status")
        last_update = None
        if isinstance(js, dict):
            last_update = js.get("last_update") or js.get("lastUpdate") or js.get("timestamp")
        if not last_update:
            hints.append("Сервис ботов не публикует время последнего обновления")
        
        # c) Стохастик
        code, js, _ = get(f"{API}/api/bots/coins-with-rsi")
        if isinstance(js, dict) and js.get("coins"):
            coins_with_stoch = sum(1 for v in js["coins"].values() 
                                 if v.get("stoch_rsi_k") is not None or v.get("stoch_rsi_d") is not None)
            if coins_with_stoch == 0:
                hints.append("Стохастик не вычисляется (нет полей stoch_rsi_k/stoch_rsi_d)")
            else:
                print(f"✅ Стохастик работает: {coins_with_stoch} монет с данными")
        
        if hints:
            print("* Найдены проблемы:")
            for h in hints:
                print(f"  - {h}")
        else:
            print("✅ Проблем не обнаружено — UI должен работать корректно")
        
        # Ждем перед следующим тестом
        if remaining > 30:  # Если осталось больше 30 секунд
            print(f"\n⏳ Ждем 30 секунд перед следующим тестом...")
            time.sleep(30)
        else:
            print(f"\n⏳ Осталось {remaining} секунд, завершаем тестирование")
            break
    
    print(f"\n=== Диагностика завершена ===")
    print(f"Проведено тестов: {test_count}")
    print(f"Общее время: {int(time.time() - start_time)} секунд")


def main():
    try:
        run_diagnosis()
    except KeyboardInterrupt:
        print("\n🛑 Диагностика прервана пользователем")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Ошибка диагностики: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
