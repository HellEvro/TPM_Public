#!/usr/bin/env python3
"""
Быстрая диагностика: почему UI пустой и где рвётся коннект
Запуск: python quick_front_backend_diagnose.py
"""

import json
import socket
import sys
from typing import Tuple, Optional

import requests

UI = "http://127.0.0.1:5000"
API = "http://127.0.0.1:5001"  # сервис ботов (bots.py)

TIMEOUT = 4


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


def check_required_ui_proxies():
    print("=== UI proxy endpoints (через app.py → порт 5000) ===")
    ui_checks = {
        "account-info (proxy)": f"{UI}/api/bots/account-info",
        "coins-with-rsi (proxy)": f"{UI}/api/bots/coins-with-rsi",
        "sync-positions (proxy, POST)": f"{UI}/api/bots/sync-positions",
    }
    for name, url in ui_checks.items():
        check_endpoint(name, url)


def check_direct_api():
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


def check_stochastic_presence():
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
    for i, (symbol, data) in enumerate(coins.items()):
        if data.get("stoch_rsi_k") is not None or data.get("stoch_rsi_d") is not None:
            with_stoch += 1
        else:
            if len(examples_missing) < 5:
                examples_missing.append(symbol)
        if i > 500:  # достаточно сэмпла
            break
    print(f"Монет всего: {total}, c StochRSI: {with_stoch}, без StochRSI (пример): {examples_missing}")


def main():
    print("=== ДИАГНОСТИКА UI ↔ API КОННЕКТА ===\n")
    
    # 1) Порты слушают?
    print("=== Порты ===")
    print(f"UI(5000): {'UP' if ping('127.0.0.1', 5000) else 'DOWN'}")
    print(f"API(5001): {'UP' if ping('127.0.0.1', 5001) else 'DOWN'}")
    print()

    # 2) Прямые вызовы к API сервиса ботов
    check_direct_api()

    # 3) Прокси через UI (важно: если прокси висит → UI пустой)
    check_required_ui_proxies()

    # 4) Стохастик
    check_stochastic_presence()

    # 5) Быстрая диагностика типичных причин «UI пусто»
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
        hints.append("В сервисе ботов отсутствует /api/bots/coins-with-rsi (bots_modules/api_endpoints.py)")

    code, js, _ = get(f"{API}/api/bots/status")
    last_update = None
    if isinstance(js, dict):
        last_update = js.get("last_update") or js.get("lastUpdate") or js.get("timestamp")
    if not last_update:
        hints.append("Сервис ботов не публикует время последнего обновления (/api/bots/status)")

    # c) Стохастик
    code, js, _ = get(f"{API}/api/bots/coins-with-rsi")
    if isinstance(js, dict) and js.get("coins"):
        if not any(v.get("stoch_rsi_k") is not None or v.get("stoch_rsi_d") is not None for v in js["coins"].values()):
            hints.append("Стохастик не вычисляется (нет полей stoch_rsi_k/stoch_rsi_d в coins-with-rsi)")

    if hints:
        print("* Найдено:")
        for h in hints:
            print(f"  - {h}")
        print("\n> Что проверить в коде (быстро):")
        print("  - app.py: есть ли прокси маршруты:")
        print("      /api/bots/account-info → call_bots_service('/api/bots/account-info')")
        print("      /api/bots/coins-with-rsi → call_bots_service('/api/bots/coins-with-rsi')")
        print("  - bots_modules/api_endpoints.py: есть ли эндпоинт /api/bots/coins-with-rsi, возвращающий coins со stoch_rsi_k/stoch_rsi_d")
        print("  - static/js/managers/bots_manager.js: this.BOTS_SERVICE_URL указывает на :5001 (уже ок), UI читает прокси :5000")
    else:
        print("Проблем не обнаружено — UI должен заполняться.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
